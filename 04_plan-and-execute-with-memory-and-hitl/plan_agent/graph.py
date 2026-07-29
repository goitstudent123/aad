"""Plan-and-Execute граф у LangGraph.

    START → planner → executor → [approval → act] → replanner → executor | respond → END

Чому executor розрізаний на три вузли (executor / approval / act), а не один, як у
прикладі з методички: interrupt() перезапускає вузол з початку після resume. Якби виклик
LLM і interrupt жили в одному вузлі, після кожного «approve» модель викликалася б удруге
й могла б попросити інші аргументи, ніж ті, які підтвердила людина. Тому LLM обирає
інструменти в executor, підтвердження питає approval (він читає лише стан), а виконує act.
"""

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .config import MAX_ITERATIONS, make_llm
from .logs import short, trace
from .schemas import Plan, ReplanDecision, TravelAnswer
from .tools import RISKY_TOOLS, TOOLS, TOOLS_BY_NAME
from .trajectory import signature

SYSTEM_PROMPT = """
You are "Weekend Escape", a travel planner that works in Plan-and-Execute mode.

## Language
Reply in the language the user writes in. Default to Ukrainian when it's unclear.

## Tools
- geocode(city)                 -> coordinates; weather needs them, so call it first
- weather(lat, lon, days)       -> live forecast
- currency(amount, code)        -> today's UAH value at the NBU rate
- search_knowledge(query)       -> knowledge base: visa rules, insurance, booking and
                                   cancellation policy, city tax, luggage, transport,
                                   typical prices, seasonality, safety
- book_hotel(...)               -> RISKY: books a hotel and charges money. A human must
                                   approve it before it runs.

## Rules
1. Reference facts (rules, requirements, typical prices) come from search_knowledge.
   Live data (weather, exchange rate) comes from weather/currency. Never mix them up.
2. Never invent numbers. Every figure comes from a tool.
3. Chain properly: geocode before weather.
4. Only call book_hotel when the user explicitly asked to book something — but when they did,
   CALL IT. Never ask the user for confirmation in text and never describe the booking instead
   of performing it: the graph itself pauses before the tool runs and a human approves or
   rejects it. A step that says "book X" is done by calling book_hotel, nothing else.
5. If a tool returns an error, say so plainly and continue with what you have.
6. Copy numbers from the user's request into tool arguments unchanged: "100 dollars" means
   currency(amount=100, code="USD"), never amount=1.
"""


class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]  # людиночитний журнал ходу роботи
    goal: str
    plan: list[str]  # список кроків
    current_step: int  # індекс наступного кроку (0-based)
    results: list[str]  # результати виконаних кроків
    completed: bool
    iterations: int  # скільки разів відпрацював executor
    max_iterations: int
    pending: list[dict]  # tool calls, які чекають на підтвердження/виконання
    decisions: list[dict]  # рішення людини щодо pending (по одному на виклик)
    log: Annotated[list[dict], operator.add]  # плаский журнал для артефактів
    stop_reason: Optional[str]
    answer: Optional[dict]


def initial_state(query: str, max_iterations: int = MAX_ITERATIONS) -> dict:
    return {
        "messages": [HumanMessage(query)],
        "goal": "",
        "plan": [],
        "current_step": 0,
        "results": [],
        "completed": False,
        "iterations": 0,
        "max_iterations": max_iterations,
        "pending": [],
        "decisions": [],
        "log": [],
        "stop_reason": None,
        "answer": None,
    }


def _decision_from(reply, call: dict) -> dict:
    """Відповідь людини (Command(resume=...)) → нормалізоване рішення.

    Приймає і dict, і True/False, і рядок 'approve'/'reject' — людина відповідає руками,
    тож форма буває різна. args у dict дозволяє сценарій edit: підтвердити, але з іншими
    параметрами.
    """
    if isinstance(reply, dict):
        approved = bool(reply.get("approved"))
        args = reply.get("args") or call["args"]
        reason = reply.get("reason", "")
    elif isinstance(reply, str):
        approved = reply.strip().lower() in {"approve", "approved", "yes", "y", "так"}
        args, reason = call["args"], reply
    else:
        approved, args, reason = bool(reply), call["args"], ""
    return {"tool": call["name"], "approved": approved, "args": args, "reason": reason}


def build_graph(llm=None, checkpointer=None):
    """Збирає та компілює граф. llm можна підмінити фейком — цим користуються тести."""
    llm = llm or make_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    planner_llm = llm.with_structured_output(Plan, method="function_calling")
    replanner_llm = llm.with_structured_output(ReplanDecision, method="function_calling")
    answer_llm = llm.with_structured_output(TravelAnswer, method="function_calling")

    def planner_node(state: PlanExecuteState) -> dict:
        """Складає ПОВНИЙ план заздалегідь — головна відмінність від ReAct."""
        query = state["messages"][0].content if state["messages"] else ""
        trace("planner", f"складаю план для: {short(query, 120)}")
        plan = planner_llm.invoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(
                    f"Задача користувача: {query}\n\n"
                    "Склади план з 2-5 конкретних кроків. Кожен крок — одна дія, яку можна "
                    "виконати одним із доступних інструментів. Не вигадуй кроків, для яких "
                    "немає інструмента, і не плануй бронювання, якщо про це не просили. "
                    "Не нумеруй кроки в тексті — нумерація додається автоматично."
                ),
            ]
        )
        # with_structured_output віддає None, якщо провайдер відповів текстом замість
        # function call (у OpenRouter це трапляється під навантаженням). Порожній план
        # буває на дуже коротких запитах. В обох випадках крок один: сама задача.
        goal = plan.goal if plan else query
        steps = (plan.steps if plan else None) or [query]
        if plan is None:
            trace("planner", "⚠ провайдер не віддав structured output — план з одного кроку")
        trace("planner", f"ціль: {short(goal, 160)}")
        for number, step in enumerate(steps, start=1):
            trace("planner", f"  {number}. {short(step, 160)}")
        return {
            "goal": goal,
            "plan": steps,
            "current_step": 0,
            "results": [],
            "messages": [AIMessage(content=f"План ({goal}): " + "; ".join(steps))],
            "log": [{"type": "plan", "goal": goal, "steps": steps}],
        }

    def executor_node(state: PlanExecuteState) -> dict:
        """Виконує ОДИН крок плану: просить LLM обрати інструменти для цього кроку."""
        step_idx = state["current_step"]
        plan = state["plan"]
        if step_idx >= len(plan):
            trace("executor", "кроків більше немає — завершую")
            return {"completed": True}
        if state["iterations"] >= state["max_iterations"]:
            trace("executor", f"⛔ ліміт {state['max_iterations']} ітерацій — зупиняюся")
            return {
                "completed": True,
                "stop_reason": f"max_iterations: ліміт {state['max_iterations']} ітерацій executor-а",
            }

        step = plan[step_idx]
        trace("executor", f"крок {step_idx + 1}/{len(plan)}: {short(step, 160)}")
        response = llm_with_tools.invoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(
                    f"Ціль: {state['goal']}\n"
                    f"Повний план: {plan}\n"
                    f"Уже зроблено: {state['results'] or 'нічого'}\n\n"
                    f"Виконай ЛИШЕ цей крок: {step}\n"
                    "Викликай інструменти, потрібні саме для цього кроку. Не забігай наперед."
                ),
            ]
        )
        calls = list(getattr(response, "tool_calls", None) or [])
        if calls:
            for call in calls:
                risky = " ⚠ РИЗИКОВИЙ" if call["name"] in RISKY_TOOLS else ""
                trace("executor", f"обрано {short(signature(call))}{risky}")
            return {
                "pending": calls,
                "iterations": state["iterations"] + 1,
                "log": [
                    {"type": "action", "step": step_idx + 1, "tool": c["name"],
                     "args": c["args"], "signature": signature(c)}
                    for c in calls
                ],
            }

        # Крок закрито міркуванням без інструментів — теж легітимно (наприклад, підсумок).
        text = response.content or "(без результату)"
        trace("executor", f"без інструментів: {short(text)}")
        return {
            "current_step": step_idx + 1,
            "iterations": state["iterations"] + 1,
            "results": [*state["results"], f"Крок {step_idx + 1} ({step}): {text}"],
            "messages": [AIMessage(content=f"Крок {step_idx + 1}: {text}")],
            "log": [{"type": "reasoning", "step": step_idx + 1, "content": text}],
        }

    def approval_node(state: PlanExecuteState) -> dict:
        """HITL-шлюз: на ризиковому інструменті зупиняє граф через interrupt()."""
        decisions, log = [], []
        for call in state["pending"]:
            if call["name"] not in RISKY_TOOLS:
                decisions.append({"tool": call["name"], "approved": True, "args": call["args"],
                                  "reason": "безпечний інструмент"})
                continue
            # Граф спиняється тут. Стан лежить у checkpointer-і, поки людина не відповість.
            trace("approval", f"⚠ {call['name']}({call['args']}) — чекаю рішення людини (interrupt)")
            reply = interrupt(
                {
                    "type": "approval_request",
                    "tool": call["name"],
                    "args": call["args"],
                    "step": state["plan"][state["current_step"]],
                    "message": (
                        "Підтвердіть ризикову дію:\n"
                        f"Інструмент: {call['name']}\n"
                        f"Параметри: {call['args']}\n"
                        "Відповідь: {'approved': true} / {'approved': false, 'reason': '...'} / "
                        "{'approved': true, 'args': {...}} для зміни параметрів"
                    ),
                }
            )
            decision = _decision_from(reply, call)
            verdict = "APPROVE" if decision["approved"] else "REJECT"
            edited = " (аргументи змінено людиною)" if decision["args"] != call["args"] else ""
            trace("approval", f"{verdict}{edited} {decision['reason'] and '— ' + decision['reason']}")
            decisions.append(decision)
            log.append({"type": "approval", "tool": call["name"], "approved": decision["approved"],
                        "args": decision["args"], "reason": decision["reason"]})
        return {"decisions": decisions, "log": log}

    def act_node(state: PlanExecuteState) -> dict:
        """Виконує підтверджені tool calls; відхилені не виконуються взагалі."""
        step_idx = state["current_step"]
        step = state["plan"][step_idx] if step_idx < len(state["plan"]) else "—"
        parts, log = [], []
        for call, decision in zip(state["pending"], state["decisions"]):
            if not decision["approved"]:
                text = (
                    f"{call['name']}: ВІДХИЛЕНО людиною"
                    f"{' — ' + decision['reason'] if decision['reason'] else ''}. Дію не виконано."
                )
                log.append({"type": "rejected", "tool": call["name"], "reason": decision["reason"]})
                trace("act", f"{call['name']}: НЕ виконано (відхилено людиною)")
            else:
                tool = TOOLS_BY_NAME[call["name"]]
                trace("act", f"викликаю {call['name']}({short(decision['args'], 160)})")
                try:
                    output = tool.invoke(decision["args"])
                except Exception as exc:  # noqa: BLE001 — валідація Pydantic або збій інструменту
                    output = {"error": f"{type(exc).__name__}: {exc}"}
                trace("act", f"{call['name']} → {short(output)}")
                text = f"{call['name']}: {output}"
                log.append({"type": "observation", "tool": call["name"], "content": str(output)})
            parts.append(text)

        joined = " | ".join(parts)
        return {
            "current_step": step_idx + 1,
            "results": [*state["results"], f"Крок {step_idx + 1} ({step}): {joined}"],
            "messages": [AIMessage(content=f"Крок {step_idx + 1}: {joined}")],
            "pending": [],
            "decisions": [],
            "log": log,
        }

    def replanner_node(state: PlanExecuteState) -> dict:
        """Після кожного кроку вирішує: continue / replan / finish."""
        if state.get("completed") or state.get("stop_reason"):
            return {"completed": True}
        step_idx, plan = state["current_step"], state["plan"]
        if step_idx >= len(plan):
            trace("replanner", "finish — план виконано повністю")
            return {"completed": True, "log": [{"type": "decision", "action": "finish",
                                               "reasoning": "план виконано повністю"}]}

        remaining = plan[step_idx:]
        trace("replanner", f"оцінюю прогрес {step_idx}/{len(plan)}, залишилось {len(remaining)}")
        decision = replanner_llm.invoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(
                    f"Ціль: {state['goal']}\n"
                    f"План: {plan}\n"
                    f"Виконано кроків: {step_idx}/{len(plan)}\n"
                    f"Результати: {state['results']}\n"
                    f"Залишилось: {remaining}\n\n"
                    "Що робити далі? continue — якщо решта кроків досі має сенс; "
                    "replan — якщо результати показали, що решту треба переписати "
                    "(тоді дай нові кроки замість тих, що залишилися); "
                    "finish — якщо ціль уже досягнута або решта кроків нічого не додасть."
                ),
            ]
        )
        if decision is None:
            # Провайдер не віддав structured output — безпечніше продовжити наявний план,
            # ніж вважати роботу завершеною.
            trace("replanner", "⚠ structured output не прийшов — продовжую план")
            return {"log": [{"type": "decision", "action": "continue",
                             "reasoning": "structured output не прийшов, продовжуємо план"}]}

        entry = {"type": "decision", "action": decision.action, "reasoning": decision.reasoning}
        trace("replanner", f"{decision.action} — {short(decision.reasoning, 160)}")

        if decision.action == "finish":
            return {"completed": True, "log": [entry],
                    "messages": [AIMessage(content=f"Завершено: {decision.reasoning}")]}
        if decision.action == "replan" and decision.updated_steps:
            # Виконані кроки залишаємо як є — переписуємо лише хвіст плану.
            new_plan = plan[:step_idx] + decision.updated_steps
            for number, step in enumerate(decision.updated_steps, start=step_idx + 1):
                trace("replanner", f"  новий крок {number}. {short(step, 160)}")
            entry["updated_steps"] = decision.updated_steps
            return {"plan": new_plan, "log": [entry],
                    "messages": [AIMessage(content=f"Перепланування: {decision.updated_steps}")]}
        return {"log": [entry]}  # continue — стан не змінюємо

    def respond_node(state: PlanExecuteState) -> dict:
        """Збирає фінальну структуровану відповідь з результатів кроків."""
        trace("respond", f"збираю фінальну відповідь із {len(state['results'])} результатів")
        prompt = (
            f"Ціль: {state['goal']}\n"
            f"План: {state['plan']}\n"
            f"Результати кроків:\n" + "\n".join(state["results"]) + "\n\n"
            "Сформуй фінальну відповідь користувачу лише з цих результатів."
        )
        if state.get("stop_reason"):
            prompt += (
                f"\n\nУВАГА: агента примусово зупинено ({state['stop_reason']}). "
                "Познач у warnings, чого не встиг зробити."
            )
        answer = answer_llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
        if answer is None:
            # Краще віддати сирі результати кроків, ніж втратити всю роботу агента.
            trace("respond", "⚠ structured output не прийшов — віддаю сирі результати кроків")
            answer = TravelAnswer(
                summary="\n".join(state["results"]),
                warnings=["Провайдер не віддав структуровану відповідь — це сирі результати кроків"],
            )
        return {"answer": answer.model_dump(),
                "messages": [AIMessage(content=answer.summary)]}

    def after_executor(state: PlanExecuteState) -> str:
        return "approval" if state.get("pending") else "replanner"

    def after_replanner(state: PlanExecuteState) -> str:
        return "respond" if state.get("completed") else "executor"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("approval", approval_node)
    graph.add_node("act", act_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", after_executor,
                               {"approval": "approval", "replanner": "replanner"})
    graph.add_edge("approval", "act")
    graph.add_edge("act", "replanner")
    graph.add_conditional_edges("replanner", after_replanner,
                               {"executor": "executor", "respond": "respond"})
    graph.add_edge("respond", END)

    # Запобіжник самого LangGraph — на випадок, якщо наш лічильник ітерацій підведе.
    return graph.compile(checkpointer=checkpointer).with_config(
        {"recursion_limit": 6 * MAX_ITERATIONS}
    )


def summarise(state: dict) -> dict:
    """Стан графа → компактний словник для друку та артефактів."""
    return {
        "goal": state.get("goal"),
        "plan": state.get("plan", []),
        "current_step": state.get("current_step", 0),
        "results": state.get("results", []),
        "completed": state.get("completed", False),
        "iterations": state.get("iterations", 0),
        "stop_reason": state.get("stop_reason"),
        "answer": state.get("answer"),
        "interrupt": [i.value for i in state.get("__interrupt__", ())],
        "log": state.get("log", []),
    }
