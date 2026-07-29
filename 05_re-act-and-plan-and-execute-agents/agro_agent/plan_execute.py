"""Plan-and-Execute граф: planner → executor → [risky_act] → replanner → … → respond.

Кроки плану виконує вкладений ReAct-агент (react.py) — план задає «що робити»,
ReAct вирішує «якими інструментами». Ризиковий виклик вкладений цикл не виконує:
він повертає його як pending, граф зупиняється перед вузлом risky_act
(interrupt_before) і чекає рішення людини у стані.
"""

import operator
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import (
    MAX_ITERATIONS,
    STEP_MAX_STEPS,
    STEP_TIMEOUT_SECONDS,
    make_llm,
)
from .logs import short, trace
from .prompts import SYSTEM_PROMPT
from .react import build_react_graph, run_react
from .schemas import AgroAnswer, Plan, ReplanDecision
from .tools import call_tool
from .trajectory import signature


class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    goal: str
    plan: list[str]
    current_step: int
    results: list[str]
    completed: bool
    iterations: int  # скільки разів відпрацював executor
    max_iterations: int
    react_steps: int  # сумарно кроків вкладених ReAct-агентів
    pending: list[dict]  # ризиковий виклик, що чекає на рішення людини
    approval: Optional[dict]  # рішення людини, покладене у стан через update_state
    log: Annotated[list[dict], operator.add]
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
        "react_steps": 0,
        "pending": [],
        "approval": None,
        "log": [],
        "stop_reason": None,
        "answer": None,
    }


def build_plan_graph(llm=None, checkpointer=None, react_graph=None):
    """Збирає та компілює граф. llm можна підмінити фейком — цим користуються тести."""
    llm = llm or make_llm()
    planner_llm = llm.with_structured_output(Plan, method="function_calling")
    replanner_llm = llm.with_structured_output(ReplanDecision, method="function_calling")
    answer_llm = llm.with_structured_output(AgroAnswer, method="function_calling")
    # Вкладеному агенту структурована відповідь не потрібна: його результат — вхід
    # для replanner-а, а не текст для користувача.
    react_graph = react_graph or build_react_graph(llm, structured=False)

    def planner_node(state: PlanExecuteState) -> dict:
        # Останнє питання людини, а не перше: у тому ж thread_id запитів може бути кілька,
        # і план потрібен для свіжого, з оглядом на попередню переписку.
        messages = state["messages"]
        query = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        context = "\n".join(short(m.content, 200) for m in messages[:-1] if m.content)
        trace("planner", f"складаю план: {short(query, 120)}")
        plan = planner_llm.invoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(
                    (f"Попередня переписка:\n{context}\n\n" if context else "")
                    + f"Задача агронома: {query}\n\n"
                    "Склади план з 2-5 конкретних кроків. Кожен крок — одна дія, яку можна "
                    "виконати доступними інструментами. Не вигадуй кроків без інструмента, "
                    "не плануй обробку поля, якщо про це не просили, і не плануй кроків, у "
                    "яких треба щось питати в користувача — його немає поруч. "
                    "Не нумеруй кроки в тексті."
                ),
            ]
        )
        # with_structured_output віддає None, якщо провайдер відповів текстом замість
        # function call. Тоді план — сама задача одним кроком.
        goal = plan.goal if plan else query
        steps = (plan.steps if plan else None) or [query]
        if plan is None:
            trace("planner", "⚠ структурованого плану не прийшло — план з одного кроку")
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
        """Виконує ОДИН крок плану вкладеним ReAct-агентом."""
        step_idx, plan = state["current_step"], state["plan"]
        if step_idx >= len(plan):
            return {"completed": True}
        if state["iterations"] >= state["max_iterations"]:
            trace("executor", f"⛔ ліміт {state['max_iterations']} ітерацій executor-а")
            return {
                "completed": True,
                "stop_reason": f"max_iterations: ліміт {state['max_iterations']} ітерацій executor-а",
            }

        step = plan[step_idx]
        trace("executor", f"крок {step_idx + 1}/{len(plan)}: {short(step, 160)}")
        nested = run_react(
            f"Ціль: {state['goal']}\n"
            f"Повний план: {plan}\n"
            f"Уже зроблено: {state['results'] or 'нічого'}\n\n"
            f"Виконай ЛИШЕ цей крок: {step}\n"
            "Викликай інструменти, потрібні саме для цього кроку. Не забігай наперед.",
            max_steps=STEP_MAX_STEPS,
            timeout=STEP_TIMEOUT_SECONDS,
            graph=react_graph,
        )
        # Траєкторія вкладеного агента лягає у спільний лог із позначкою кроку плану.
        log = [{**entry, "plan_step": step_idx + 1, "source": "react"} for entry in nested["trajectory"]]

        if nested["pending"]:
            call = nested["pending"][0]
            trace("executor", f"⚠ крок вимагає ризикової дії {signature(call)}")
            return {
                "pending": nested["pending"],
                "iterations": state["iterations"] + 1,
                "react_steps": state["react_steps"] + nested["steps"],
                "log": log,
            }

        text = (nested["answer"] or {}).get("summary") or "(без результату)"
        trace("executor", f"крок {step_idx + 1} закрито: {short(text, 160)}")
        return {
            "current_step": step_idx + 1,
            "iterations": state["iterations"] + 1,
            "react_steps": state["react_steps"] + nested["steps"],
            "results": [*state["results"], f"Крок {step_idx + 1} ({step}): {text}"],
            "messages": [AIMessage(content=f"Крок {step_idx + 1}: {text}")],
            "log": log,
        }

    def risky_act_node(state: PlanExecuteState) -> dict:
        """Виконує ризиковий виклик — але лише якщо людина його підтвердила.

        Граф зупиняється ПЕРЕД цим вузлом (interrupt_before), тож рішення вже лежить
        у стані. Відсутнє рішення трактуємо як відмову: незворотну дію без явного
        «так» виконувати не можна.
        """
        step_idx = state["current_step"]
        step = state["plan"][step_idx] if step_idx < len(state["plan"]) else "—"
        call = state["pending"][0]
        approval = state.get("approval") or {}
        approved = bool(approval.get("approved"))
        args = approval.get("args") or call["args"]
        reason = approval.get("reason", "")

        if approved:
            trace("risky_act", f"APPROVE — виконую {call['name']}({short(args, 120)})")
            output = call_tool(call["name"], args)
            trace("risky_act", f"{call['name']} → {short(output, 160)}")
            text = f"{call['name']}: {output}"
            log = [{"type": "approval", "tool": call["name"], "approved": True, "args": args},
                   {"type": "observation", "tool": call["name"], "content": output}]
        else:
            trace("risky_act", f"REJECT — {call['name']} не виконано{reason and ': ' + reason}")
            text = (
                f"{call['name']}: ВІДХИЛЕНО людиною"
                f"{' — ' + reason if reason else ''}. Дію не виконано."
            )
            log = [{"type": "rejected", "tool": call["name"], "args": call["args"], "reason": reason}]

        return {
            "current_step": step_idx + 1,
            "results": [*state["results"], f"Крок {step_idx + 1} ({step}): {text}"],
            "messages": [AIMessage(content=f"Крок {step_idx + 1}: {text}")],
            "pending": [],
            "approval": None,
            "log": log,
        }

    def replanner_node(state: PlanExecuteState) -> dict:
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
                    "finish — якщо ціль досягнута або решта нічого не додасть."
                ),
            ]
        )
        if decision is None:
            trace("replanner", "⚠ structured output не прийшов — продовжую план")
            return {"log": [{"type": "decision", "action": "continue",
                             "reasoning": "structured output не прийшов, продовжуємо план"}]}

        entry = {"type": "decision", "action": decision.action, "reasoning": decision.reasoning}
        trace("replanner", f"{decision.action} — {short(decision.reasoning, 160)}")

        if decision.action == "finish":
            return {"completed": True, "log": [entry],
                    "messages": [AIMessage(content=f"Завершено: {decision.reasoning}")]}
        if decision.action == "replan" and decision.updated_steps:
            # Виконані кроки лишаємо як є — переписуємо тільки хвіст плану.
            new_plan = plan[:step_idx] + decision.updated_steps
            for number, step in enumerate(decision.updated_steps, start=step_idx + 1):
                trace("replanner", f"  новий крок {number}. {short(step, 160)}")
            entry["updated_steps"] = decision.updated_steps
            return {"plan": new_plan, "log": [entry],
                    "messages": [AIMessage(content=f"Перепланування: {decision.updated_steps}")]}
        return {"log": [entry]}

    def respond_node(state: PlanExecuteState) -> dict:
        trace("respond", f"збираю відповідь із {len(state['results'])} результатів")
        prompt = (
            f"Ціль: {state['goal']}\n"
            f"План: {state['plan']}\n"
            "Результати кроків:\n" + "\n".join(state["results"]) + "\n\n"
            "Сформуй фінальну відповідь агроному лише з цих результатів."
        )
        if state.get("stop_reason"):
            prompt += (
                f"\n\nУВАГА: агента примусово зупинено ({state['stop_reason']}). "
                "Познач у warnings, чого не встиг зробити."
            )
        answer = answer_llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
        if answer is None:
            answer = AgroAnswer(
                summary="\n".join(state["results"]),
                warnings=["Провайдер не віддав структуровану відповідь — це сирі результати кроків"],
            )
        return {"answer": answer.model_dump(), "messages": [AIMessage(content=answer.summary)]}

    def after_executor(state: PlanExecuteState) -> str:
        return "risky_act" if state.get("pending") else "replanner"

    def after_replanner(state: PlanExecuteState) -> str:
        return "respond" if state.get("completed") else "executor"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("risky_act", risky_act_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", after_executor,
                               {"risky_act": "risky_act", "replanner": "replanner"})
    graph.add_edge("risky_act", "replanner")
    graph.add_conditional_edges("replanner", after_replanner,
                               {"executor": "executor", "respond": "respond"})
    graph.add_edge("respond", END)

    # Без checkpointer-а паузу відновити нічим, тому interrupt_before ставимо лише з ним.
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["risky_act"] if checkpointer else None,
    )
    return compiled.with_config({"recursion_limit": 6 * MAX_ITERATIONS})


def run_plan(app, query: str, config: dict, max_iterations: int = MAX_ITERATIONS) -> dict:
    """Один прогін Plan-and-Execute із заміром часу."""
    started = time.monotonic()
    state = app.invoke(initial_state(query, max_iterations), config)
    result = summarise(state)
    result |= {"query": query, "elapsed_seconds": round(time.monotonic() - started, 2)}
    return result


def tools_used(state: dict) -> list[str]:
    return [e["signature"] for e in state.get("log", []) if e.get("type") == "action"]


def summarise(state: dict) -> dict:
    """Стан графа → компактний словник для друку та артефактів."""
    return {
        "goal": state.get("goal"),
        "plan": state.get("plan", []),
        "current_step": state.get("current_step", 0),
        "results": state.get("results", []),
        "completed": state.get("completed", False),
        "iterations": state.get("iterations", 0),
        "react_steps": state.get("react_steps", 0),
        "pending": state.get("pending", []),
        "stop_reason": state.get("stop_reason"),
        "answer": state.get("answer"),
        "tools_used": tools_used(state),
        "log": state.get("log", []),
    }
