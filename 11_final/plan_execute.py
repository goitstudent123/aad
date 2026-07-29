"""Plan-and-Execute з ДЗ2 як підграф білінг-агента: planner → executor → replanner → …

Кожен крок плану виконує вкладений ReAct-цикл (react_core): план каже «що робити»,
ReAct вирішує «якими інструментами». Стан підграфа — підмножина полів MASState,
тому LangGraph зшиває його з батьківським графом без перекладу.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import MAX_ITERATIONS, MAX_STEPS
from logs import short, trace
from prompts import PLANNER, WORKERS
from react_core import react_loop
from trajectory_logger import log_step, steps_from_messages


class Plan(BaseModel):
    """План робіт: ціль і послідовність конкретних кроків."""

    goal: str = Field(description="Головна ціль запиту одним реченням")
    steps: list[str] = Field(description="2-4 кроки, кожен — одна дія, виконувана інструментом")


class ReplanDecision(BaseModel):
    """Рішення replanner-а після виконання кроку."""

    action: Literal["continue", "replan", "finish"] = Field(
        description="continue = наступний крок, replan = переписати решту, finish = досить"
    )
    updated_steps: Optional[list[str]] = Field(
        default=None, description="Нові кроки замість тих, що лишилися (лише для replan)"
    )
    reasoning: str = Field(description="Коротке пояснення українською")


class PlanState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step: int
    results: Annotated[list[str], operator.add]
    step_count: int
    plan_iterations: int
    trajectory: Annotated[list[dict], operator.add]


def build_plan_subgraph(llm, tools: list):
    """Підграф білінг-агента. Компілюється без checkpointer — його дає батьківський граф."""
    planner_llm = llm.with_structured_output(Plan, method="function_calling")
    replanner_llm = llm.with_structured_output(ReplanDecision, method="function_calling")
    system = WORKERS["billing"]

    def query_of(state: PlanState) -> str:
        return next((m.content for m in reversed(state["messages"])
                     if isinstance(m, HumanMessage)), "")

    async def planner(state: PlanState) -> dict:
        query = query_of(state)
        trace("billing", f"планую: {short(query, 120)}")
        plan = await planner_llm.ainvoke([
            SystemMessage(system),
            HumanMessage(f"Запит клієнта: {query}\n\n{PLANNER}"),
        ])
        # with_structured_output віддає None, якщо провайдер відповів текстом.
        steps = (plan.steps if plan else None) or [query]
        goal = plan.goal if plan else query
        for number, step in enumerate(steps, start=1):
            trace("billing", f"  {number}. {short(step, 140)}")
        return {
            "plan": steps,
            "current_step": 0,
            "plan_iterations": 0,
            "messages": [AIMessage(content=f"План ({goal}): " + "; ".join(steps), name="billing")],
            "trajectory": [log_step("billing", "planner", query, f"план: {steps}")],
        }

    async def executor(state: PlanState) -> dict:
        index, plan = state["current_step"], state["plan"]
        if index >= len(plan) or state["plan_iterations"] >= MAX_ITERATIONS:
            return {"current_step": len(plan)}

        step = plan[index]
        trace("billing", f"крок {index + 1}/{len(plan)}: {short(step, 140)}")
        run = await react_loop(
            llm, tools, system,
            f"Повний план: {plan}\nУже зроблено: {state['results'] or 'нічого'}\n\n"
            f"Виконай ЛИШЕ цей крок: {step}",
            agent="billing", max_steps=MAX_STEPS,
        )
        return {
            "current_step": index + 1,
            "plan_iterations": state["plan_iterations"] + 1,
            "step_count": state["step_count"] + run["steps"],
            "results": [f"Крок {index + 1} ({step}): {run['text']}"],
            "messages": [AIMessage(content=f"Крок {index + 1}: {run['text']}", name="billing")],
            "trajectory": steps_from_messages("billing", f"executor:{index + 1}", run["messages"]),
        }

    async def replanner(state: PlanState) -> dict:
        index, plan = state["current_step"], state["plan"]
        if index >= len(plan) or state["plan_iterations"] >= MAX_ITERATIONS:
            trace("billing", "план завершено")
            return {"trajectory": [log_step("billing", "replanner", "finish", "план виконано")]}

        decision = await replanner_llm.ainvoke([
            SystemMessage(system),
            HumanMessage(
                f"План: {plan}\nВиконано: {index}/{len(plan)}\nРезультати: {state['results']}\n"
                f"Залишилось: {plan[index:]}\n\nЩо робити далі?"
            ),
        ])
        if decision is None:
            return {"trajectory": [log_step("billing", "replanner", "continue",
                                            "structured output не прийшов")]}

        trace("billing", f"replanner: {decision.action} — {short(decision.reasoning, 120)}")
        entry = log_step("billing", "replanner", decision.action, decision.reasoning)
        if decision.action == "finish":
            return {"current_step": len(plan), "trajectory": [entry]}
        if decision.action == "replan" and decision.updated_steps:
            return {"plan": plan[:index] + decision.updated_steps, "trajectory": [entry]}
        return {"trajectory": [entry]}

    def after_replanner(state: PlanState) -> str:
        done = (state["current_step"] >= len(state["plan"])
                or state["plan_iterations"] >= MAX_ITERATIONS)
        return "done" if done else "executor"

    graph = StateGraph(PlanState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("replanner", replanner)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges("replanner", after_replanner,
                                {"executor": "executor", "done": END})
    return graph.compile()
