# -*- coding: utf-8 -*-
"""LangGraph backend for CellScientist orchestration.

This module provides a framework-backed orchestration path that reuses the
existing agent implementations but drives them through an explicit LangGraph
state machine.
"""

from __future__ import annotations

from typing import Any, Dict, TypedDict

from .agents import EvaluationAgent, ExecutionAgent, ModelingAgent, ResearchAgent, TaskContext
from .orchestrator import _now_iso


class LGState(TypedDict, total=False):
    config: Dict[str, Any]
    task_id: str
    smiles_list: list[str]
    h5_file_path: str
    iteration: int
    max_iterations: int
    research_input: Dict[str, Any]
    modeling_input: Dict[str, Any]
    execution_input: Dict[str, Any]
    evaluation_input: Dict[str, Any]
    orchestration_msg: Dict[str, Any]
    next_step: str
    final_result: Dict[str, Any]


def _build_initial_state(config: Dict[str, Any]) -> LGState:
    ctx = TaskContext.from_config(config)
    max_iter = int(ctx.max_iterations or ((config.get("review") or {}).get("max_iterations") or 10))
    return {
        "config": config,
        "task_id": ctx.task_id,
        "smiles_list": list(ctx.smiles_list or []),
        "h5_file_path": str(ctx.h5_file_path or ""),
        "iteration": 0,
        "max_iterations": max_iter,
        "research_input": {
            "smiles_list": list(ctx.smiles_list or []),
            "h5_file_path": str(ctx.h5_file_path or ""),
            "context_text": "",
            "stage": "design",
            "iteration": 0,
            "max_iterations": max_iter,
            "history_summary": [],
            "best_metric_score": None,
            "_fsm_state": "KNOWLEDGE_RETRIEVAL",
            "_task_id": ctx.task_id,
        },
    }


async def run_orchestrator_langgraph(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run agent-mode orchestration through LangGraph.

    Notes:
    - Reuses existing agent `process(...)` implementations.
    - Keeps deterministic routing on `decision/suggested_target/max_iterations`.
    """
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "LangGraph backend requested but dependency is unavailable."
        ) from exc

    # Instantiate agents once for graph run.
    from .message_bus import SimpleMessageBus

    bus = SimpleMessageBus()
    research_agent = ResearchAgent(bus, config)
    modeling_agent = ModelingAgent(bus, config)
    execution_agent = ExecutionAgent(bus, config)
    evaluation_agent = EvaluationAgent(bus, config)

    async def research_node(state: LGState) -> Dict[str, Any]:
        msg = dict(state.get("research_input") or {})
        msg.setdefault("iteration", state.get("iteration", 0))
        msg.setdefault("max_iterations", state.get("max_iterations", 10))
        resp = await research_agent.process(msg)
        return {"modeling_input": dict(resp.data or {})}

    async def modeling_node(state: LGState) -> Dict[str, Any]:
        msg = dict(state.get("modeling_input") or {})
        msg.setdefault("iteration", state.get("iteration", 0))
        msg.setdefault("max_iterations", state.get("max_iterations", 10))
        resp = await modeling_agent.process(msg)
        return {"execution_input": dict(resp.data or {})}

    async def execution_node(state: LGState) -> Dict[str, Any]:
        msg = dict(state.get("execution_input") or {})
        msg.setdefault("iteration", state.get("iteration", 0))
        msg.setdefault("max_iterations", state.get("max_iterations", 10))
        resp = await execution_agent.process(msg)
        return {"evaluation_input": dict(resp.data or {})}

    async def evaluation_node(state: LGState) -> Dict[str, Any]:
        msg = dict(state.get("evaluation_input") or {})
        msg.setdefault("iteration", state.get("iteration", 0))
        msg.setdefault("max_iterations", state.get("max_iterations", 10))
        resp = await evaluation_agent.process(msg)
        return {
            "orchestration_msg": {"status": resp.status, "data": dict(resp.data or {})},
            "iteration": int(state.get("iteration", 0)) + 1,
        }

    async def routing_node(state: LGState) -> Dict[str, Any]:
        msg = dict(state.get("orchestration_msg") or {})
        data = dict(msg.get("data") or {})
        status = str(msg.get("status") or "")
        decision = str(data.get("decision") or "")

        # Stop conditions.
        if (
            status == "success"
            or decision == "SUCCESS"
            or data.get("max_iterations_reached")
            or int(state.get("iteration", 0)) >= int(state.get("max_iterations", 10))
        ):
            return {
                "next_step": "end",
                "final_result": {
                    "status": "terminated",
                    "task_id": state.get("task_id"),
                    "ended_at": _now_iso(),
                    "data": data,
                }
            }

        feedback = dict(data.get("feedback_package") or {})
        suggested = str(feedback.get("suggested_target") or "modeling").lower()

        if suggested == "research":
            return {
                "next_step": "research",
                "research_input": {
                    "smiles_list": list(state.get("smiles_list") or []),
                    "h5_file_path": str(state.get("h5_file_path") or ""),
                    "context_text": str(feedback.get("technical_feedback") or ""),
                    "knowledge_gap": str(feedback.get("knowledge_gap") or ""),
                    "stage": "refinement",
                    "iteration": state.get("iteration", 0),
                    "max_iterations": state.get("max_iterations", 10),
                    "_fsm_state": "KNOWLEDGE_RETRIEVAL",
                    "_task_id": state.get("task_id"),
                }
            }

        return {
            "next_step": "modeling",
            "modeling_input": {
                "execution_report": data.get("insight_report") or {},
                "literature_context": "",
                "falsifiable_context": {
                    "technical_feedback": str(feedback.get("technical_feedback") or ""),
                    "suggested_target": suggested,
                },
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 10),
                "_fsm_state": "MODEL_GENERATION",
                "_task_id": state.get("task_id"),
            }
        }

    def _route_after_routing(state: LGState) -> str:
        step = str(state.get("next_step") or "").lower().strip()
        if step in {"research", "modeling", "end"}:
            return step
        return "end"

    graph = StateGraph(LGState)
    graph.add_node("research", research_node)
    graph.add_node("modeling", modeling_node)
    graph.add_node("execution", execution_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("routing", routing_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "modeling")
    graph.add_edge("modeling", "execution")
    graph.add_edge("execution", "evaluation")
    graph.add_edge("evaluation", "routing")
    graph.add_conditional_edges(
        "routing",
        _route_after_routing,
        {
            "research": "research",
            "modeling": "modeling",
            "end": END,
        },
    )

    app = graph.compile()
    final_state = await app.ainvoke(_build_initial_state(config))
    result = final_state.get("final_result") if isinstance(final_state, dict) else None
    if isinstance(result, dict):
        return result.get("data") or result
    return {"status": "terminated", "data": {}}
