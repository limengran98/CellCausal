# -*- coding: utf-8 -*-
"""Multi-Agent Pipeline Orchestrator.

Initialises the message bus and all agents, loads :class:`~.agents.TaskContext`
from ``pipeline_config.json``, and manages the iteration loop using a
Finite State Machine (FSM).

Entry-point integration
-----------------------
``run_pipeline.py`` can launch the orchestrator via::

    from cellscientist.core.orchestrator import run_orchestrator
    asyncio.run(run_orchestrator(config))

Or for a synchronous call::

    from cellscientist.core.orchestrator import run_orchestrator_sync
    run_orchestrator_sync(config)

FSM States
----------
``INITIALIZING`` → ``KNOWLEDGE_RETRIEVAL`` → ``MODEL_GENERATION`` →
``EXECUTING`` → ``EVALUATING`` → ``RETRY_LOGIC`` (on failure) or
``TERMINATED`` (on success / exhausted iterations).

Each transition emits its legacy ``[Phase: …]`` alias so that existing
log-parsers and monitoring systems continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .agents import (
    EvaluationAgent,
    ExecutionAgent,
    ModelingAgent,
    ResearchAgent,
    TaskContext,
)
from .message_bus import SimpleMessageBus

logger = logging.getLogger(__name__)


# =============================================================================
# Finite State Machine definitions
# =============================================================================


class PipelineState(enum.Enum):
    """FSM states for the multi-agent pipeline.

    Each state is paired with a legacy ``[Phase: …]`` alias in
    :data:`STATE_LEGACY_ALIASES` so that existing monitoring and log-parsers
    remain unaffected.
    """

    INITIALIZING = "INITIALIZING"
    KNOWLEDGE_RETRIEVAL = "KNOWLEDGE_RETRIEVAL"
    MODEL_GENERATION = "MODEL_GENERATION"
    EXECUTING = "EXECUTING"
    EVALUATING = "EVALUATING"
    RETRY_LOGIC = "RETRY_LOGIC"
    TERMINATED = "TERMINATED"


#: Mapping from :class:`PipelineState` to the legacy ``[Phase: …]`` marker
#: that MUST appear in log output for backward-compatible monitoring.
STATE_LEGACY_ALIASES: Dict[PipelineState, str] = {
    PipelineState.INITIALIZING: "[Phase: Setup]",
    PipelineState.KNOWLEDGE_RETRIEVAL: "[Phase: Review]",
    PipelineState.MODEL_GENERATION: "[Phase: Experiment]",
    PipelineState.EXECUTING: "[Phase: Execution]",
    PipelineState.EVALUATING: "[Phase: Evaluation]",
    PipelineState.RETRY_LOGIC: "[Phase: Retry]",
    PipelineState.TERMINATED: "[Phase: Complete]",
}


# =============================================================================
# Logging helper (consistent with other core modules)
# =============================================================================


def _log(msg: str, *, console: bool = False) -> None:
    """Print a structured log message.

    Args:
        msg: Message text.
        console: If ``True`` uses ``[CELL_CONSOLE]`` prefix; otherwise ``[DETAIL]``.
    """
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    else:
        print(f"[DETAIL] {msg}", flush=True)


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Orchestrator
# =============================================================================


class PipelineOrchestrator:
    """Async orchestrator that coordinates the multi-agent pipeline via an FSM.

    Initialises a :class:`~.message_bus.SimpleMessageBus`, instantiates one
    agent of each type, and drives the agent loop until the accuracy goal is
    met or ``max_iterations`` is exhausted.

    The orchestrator uses a :class:`PipelineState` FSM.  Each state transition
    prints the corresponding legacy ``[Phase: …]`` alias so that existing
    monitoring and log-parsers continue to work unchanged.

    Attributes:
        config: Full pipeline configuration dict.
        bus: Shared message bus.
        context: Shared task context (SMILES, paths, iteration state).
        research_agent: :class:`~.agents.ResearchAgent` instance.
        modeling_agent: :class:`~.agents.ModelingAgent` instance.
        execution_agent: :class:`~.agents.ExecutionAgent` instance.
        evaluation_agent: :class:`~.agents.EvaluationAgent` instance.
        current_state: The active :class:`PipelineState`.
        previous_state: The last :class:`PipelineState` before the current one.
        transition_history: Ordered list of ``(from_state, to_state, timestamp)``
            tuples recorded at every state transition.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Create the orchestrator.

        Args:
            config: Full pipeline configuration (``pipeline_config.json``).
        """
        self.config: Dict[str, Any] = config
        self.bus: SimpleMessageBus = SimpleMessageBus()

        # Instantiate agents.
        self.research_agent = ResearchAgent(self.bus, config)
        self.modeling_agent = ModelingAgent(self.bus, config)
        self.execution_agent = ExecutionAgent(self.bus, config)
        self.evaluation_agent = EvaluationAgent(self.bus, config)

        # Build shared task context.
        self.context: TaskContext = TaskContext.from_config(config)

        # FSM state tracking.  Start from a sentinel value so that the first
        # real _enter_state() call produces a clean `None → INITIALIZING` entry
        # in transition_history (stored as ``INITIALIZING → INITIALIZING`` for
        # the bootstrap case is avoided by starting with the sentinel below).
        self.current_state: Optional[PipelineState] = None
        self.previous_state: Optional[PipelineState] = None
        self.transition_history: List[Tuple[Optional[PipelineState], PipelineState, str]] = []

        # Bootstrap: enter the INITIALIZING state.
        self._enter_state(PipelineState.INITIALIZING)

        _log(
            f"[Orchestrator] Initialised task '{self.context.task_id}' "
            f"(max_iterations={self.context.max_iterations}, "
            f"smiles={len(self.context.smiles_list)})",
            console=True,
        )

    # ------------------------------------------------------------------
    # FSM helpers
    # ------------------------------------------------------------------

    def _enter_state(self, new_state: PipelineState) -> None:
        """Transition to *new_state* and emit the legacy phase alias.

        Records the transition in :attr:`transition_history`.

        Args:
            new_state: The :class:`PipelineState` to transition to.
        """
        from_state = self.current_state
        self.previous_state = from_state
        self.current_state = new_state
        ts = _now_iso()
        self.transition_history.append((from_state, new_state, ts))

        from_label = from_state.value if from_state is not None else "None"
        legacy = STATE_LEGACY_ALIASES.get(new_state, "")
        _log(
            f"🚀 [Orchestrator] State Change: {from_label} -> {new_state.value} "
            f"{legacy}",
            console=True,
        )
        # Emit the bare legacy alias on its own line so log-parsers can match it.
        if legacy:
            _log(legacy, console=True)

    async def run(self) -> Dict[str, Any]:
        """Execute the full multi-agent pipeline using the FSM.

        Kick-off sequence:
        1. Transition to ``KNOWLEDGE_RETRIEVAL`` and send initial message.
        2. Research → ``biology_insights`` → Modeling (``MODEL_GENERATION``).
        3. Modeling → ``code_execution`` → Execution (``EXECUTING``).
        4. Execution → ``evaluation`` → Evaluation (``EVALUATING``).
        5. Evaluation → ``orchestration`` (``TERMINATED``) OR
           failure → ``RETRY_LOGIC`` → back to ``MODEL_GENERATION``.

        Returns:
            A summary dict with ``status``, ``iterations``, and final
            ``accuracy`` (if detected).
        """
        # Subscribe to the orchestration topic so we can detect completion.
        orchestration_queue = self.bus.subscribe("orchestration")

        # Subscribe each agent to its input topic.
        research_inbox = self.bus.subscribe("biology_insights")
        modeling_inbox = self.bus.subscribe("modeling")
        code_exec_inbox = self.bus.subscribe("code_execution")
        eval_inbox = self.bus.subscribe("evaluation")

        # Launch agent run-loops as background tasks.
        tasks = [
            asyncio.create_task(
                self.research_agent.run(research_inbox), name="research_agent"
            ),
            asyncio.create_task(
                self.modeling_agent.run(modeling_inbox), name="modeling_agent"
            ),
            asyncio.create_task(
                self.execution_agent.run(code_exec_inbox), name="execution_agent"
            ),
            asyncio.create_task(
                self.evaluation_agent.run(eval_inbox), name="evaluation_agent"
            ),
        ]

        # ---- KNOWLEDGE_RETRIEVAL state ----
        self._enter_state(PipelineState.KNOWLEDGE_RETRIEVAL)

        # Seed the pipeline.
        initial_message = {
            "smiles_list": self.context.smiles_list,
            "h5_file_path": self.context.h5_file_path,
            "context_text": "",
            "stage": "design",
            "iteration": 0,
            "max_iterations": self.context.max_iterations,
            # Embed FSM state so agents can include it in their logs.
            "_fsm_state": self.current_state.value if self.current_state else "",
            "_task_id": self.context.task_id,
        }
        # The ResearchAgent subscribes to biology_insights; trigger it.
        await self.bus.publish("biology_insights", initial_message)

        # Wait for the pipeline to signal completion via orchestration topic.
        pipeline_timeout: float = float(
            (self.config.get("exec") or {}).get("timeout_seconds") or 7200
        )
        _log("[Orchestrator] Pipeline started; waiting for completion…", console=True)
        result: Optional[Dict[str, Any]] = None
        try:
            result = await asyncio.wait_for(
                orchestration_queue.get(),
                timeout=pipeline_timeout,
            )
        except asyncio.TimeoutError:
            _log(
                f"[Orchestrator] ⚠️ Pipeline timed out after {pipeline_timeout:.0f}s "
                "before reaching orchestration topic.",
                console=True,
            )
            result = {"status": "timeout", "data": {}}
        finally:
            # Cancel all agent background tasks.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        status = (result or {}).get("status", "unknown")
        data = (result or {}).get("data") or {}
        accuracy = data.get("accuracy")
        iterations_done = self.context.iteration

        # Handle error states: log full context and transition to RETRY_LOGIC.
        if status == "error":
            _log(
                f"[Orchestrator] ❌ Agent reported error. Full context:\n"
                f"  agent_id={data.get('agent_id', 'unknown')}\n"
                f"  error={data.get('error', 'unknown')}\n"
                f"  traceback={str(data.get('traceback', ''))[:500]}",
                console=True,
            )
            self._enter_state(PipelineState.RETRY_LOGIC)

        # Terminal state.
        self._enter_state(PipelineState.TERMINATED)

        summary = {
            "status": status,
            "task_id": self.context.task_id,
            "iterations": iterations_done,
            "accuracy": accuracy,
            "max_iterations_reached": data.get("max_iterations_reached", False),
            "fsm_transitions": [
                {"from": f.value, "to": t.value, "timestamp": ts}
                for f, t, ts in self.transition_history
            ],
        }

        _log(
            f"[Orchestrator] Pipeline finished: status={status}, "
            f"accuracy={accuracy}, iterations={iterations_done}",
            console=True,
        )
        return summary


# =============================================================================
# Public helpers
# =============================================================================


async def run_orchestrator(config: Dict[str, Any]) -> Dict[str, Any]:
    """Async entry-point: create the orchestrator and run the pipeline.

    Args:
        config: Full pipeline configuration dict.

    Returns:
        Summary dict from :meth:`PipelineOrchestrator.run`.
    """
    orchestrator = PipelineOrchestrator(config)
    return await orchestrator.run()


def run_orchestrator_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper around :func:`run_orchestrator`.

    Convenience function for callers that cannot use ``await``.

    Args:
        config: Full pipeline configuration dict.

    Returns:
        Summary dict from :meth:`PipelineOrchestrator.run`.
    """
    return asyncio.run(run_orchestrator(config))


def load_config(pipeline_config_path: str = "configs/pipeline_config.json") -> Dict[str, Any]:
    """Load and return the pipeline configuration from a JSON file.

    Args:
        pipeline_config_path: Path to ``pipeline_config.json``.

    Returns:
        Parsed configuration dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    abs_path = os.path.abspath(pipeline_config_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Pipeline config not found at: {abs_path}"
        )
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)
