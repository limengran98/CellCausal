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
from .llm_client import TokenMeter

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
    FEEDBACK_ROUTING = "FEEDBACK_ROUTING"
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
    PipelineState.FEEDBACK_ROUTING: "[Phase: Routing]",
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


def _interpolate_path(path_template: str, config: Dict[str, Any]) -> str:
    """Replace ``${key}`` placeholders in *path_template* with config values.

    Args:
        path_template: Path string that may contain ``${key}`` placeholders.
        config: Pipeline configuration dict supplying replacement values.

    Returns:
        The interpolated path string.  Unrecognised placeholders are left
        unchanged.
    """
    result = path_template
    for key, value in config.items():
        if isinstance(value, str):
            result = result.replace(f"${{{key}}}", value)
    return result


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
        best_metric_score: Globally best primary metric value seen across all
            iterations.  Used to return the optimal checkpoint when
            ``max_iterations`` is reached.
        best_code: Python source that produced :attr:`best_metric_score`.
        best_artifacts_path: File-system path where the best iteration's
            artifacts were persisted.
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

        # Global Best Checkpoint (Bug 1 fix): track the historically optimal
        # code and metric so the system returns the best state even when the
        # last iteration regressed.
        self.best_metric_score: float = -float("inf")
        self.best_code: str = ""
        self.best_artifacts_path: str = ""
        self.best_notebook_json: str = ""
        self.best_artifact_type: str = ""
        self.best_modeling_metadata: Dict[str, Any] = {}

        # Notebook-aware best/accepted artifacts for agent-mode.
        self.last_accepted_notebook_json: str = ""
        self.last_accepted_artifact_type: str = ""
        self.last_accepted_modeling_metadata: Dict[str, Any] = {}

        # ---- Falsifiable Iteration Protocol state ----
        # Per-iteration metric history for accept/reject decisions.
        self.iteration_history: List[Dict[str, Any]] = []
        # The last accepted (improving) code; used for revert on regression.
        self.last_accepted_code: str = ""
        self.last_accepted_metrics: Dict[str, Any] = {}
        # Counter for consecutive rejections; triggers forced new hypothesis.
        self.consecutive_rejections: int = 0

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

    # ------------------------------------------------------------------
    # Falsifiable Iteration Protocol
    # ------------------------------------------------------------------

    def _evaluate_iteration(
        self, data: Dict[str, Any], iteration: int
    ) -> Dict[str, Any]:
        """Apply the Falsifiable Iteration Protocol to an evaluation result.

        Compares current metrics with the previously accepted state.  Returns
        a verdict dict that the orchestrator uses to decide whether to
        **accept** (lock improvement) or **reject** (revert to best state and
        force a new hypothesis).

        Args:
            data: Evaluation payload containing ``accuracy``, ``metrics``, and
                ``code``.
            iteration: Current iteration number.

        Returns:
            Dict with keys ``verdict`` (``"ACCEPT"`` or ``"REJECT"``),
            ``metric_delta``, ``current_score``, ``previous_score``, and
            ``forced_new_hypothesis`` (bool).
        """
        current_score: Optional[float] = None
        try:
            current_score = float(data.get("accuracy") or 0)
        except (TypeError, ValueError):
            current_score = None

        previous_score = self.last_accepted_metrics.get("accuracy")
        code = data.get("code") or ""
        metrics = data.get("metrics") or {}
        notebook_json = data.get("notebook_json") or ""
        artifact_type = data.get("artifact_type") or ""
        modeling_metadata = data.get("modeling_metadata") or {}

        record = {
            "iteration": iteration,
            "score": current_score,
            "metrics": metrics,
            "code_len": len(code),
            "timestamp": _now_iso(),
        }

        # First iteration — always accept.
        if not self.iteration_history:
            self.iteration_history.append(record)
            if current_score is not None:
                self.last_accepted_code = code
                self.last_accepted_metrics = {"accuracy": current_score, **metrics}
                self.last_accepted_notebook_json = notebook_json or self.last_accepted_notebook_json
                self.last_accepted_artifact_type = artifact_type or self.last_accepted_artifact_type
                self.last_accepted_modeling_metadata = dict(modeling_metadata or {})
                self.consecutive_rejections = 0
            _log(
                f"[Falsifiable] ✅ ACCEPT (first iteration, "
                f"score={current_score})",
                console=True,
            )
            return {
                "verdict": "ACCEPT",
                "metric_delta": 0.0,
                "current_score": current_score,
                "previous_score": None,
                "forced_new_hypothesis": False,
            }

        self.iteration_history.append(record)

        # Compare with the previous accepted score.
        if current_score is not None and previous_score is not None:
            delta = current_score - previous_score
            if delta >= 0:
                # Improvement or no change — ACCEPT / LOCK.
                self.last_accepted_code = code
                self.last_accepted_metrics = {"accuracy": current_score, **metrics}
                self.last_accepted_notebook_json = notebook_json or self.last_accepted_notebook_json
                self.last_accepted_artifact_type = artifact_type or self.last_accepted_artifact_type
                self.last_accepted_modeling_metadata = dict(modeling_metadata or {})
                self.consecutive_rejections = 0
                _log(
                    f"[Falsifiable] ✅ ACCEPT — score improved: "
                    f"{previous_score:.4f} → {current_score:.4f} "
                    f"(Δ={delta:+.4f})",
                    console=True,
                )
                return {
                    "verdict": "ACCEPT",
                    "metric_delta": delta,
                    "current_score": current_score,
                    "previous_score": previous_score,
                    "forced_new_hypothesis": False,
                }
            else:
                # Degradation — REJECT / REVERT.
                self.consecutive_rejections += 1
                force_new = self.consecutive_rejections >= 2
                _log(
                    f"[Falsifiable] ❌ REJECT — score degraded: "
                    f"{previous_score:.4f} → {current_score:.4f} "
                    f"(Δ={delta:+.4f}, consecutive_rejections="
                    f"{self.consecutive_rejections})",
                    console=True,
                )
                if force_new:
                    _log(
                        "[Falsifiable] 🔬 Forcing NEW HYPOTHESIS "
                        "(consecutive rejections ≥ 2)",
                        console=True,
                    )
                return {
                    "verdict": "REJECT",
                    "metric_delta": delta,
                    "current_score": current_score,
                    "previous_score": previous_score,
                    "forced_new_hypothesis": force_new,
                }

        # Metrics not available — accept tentatively.
        if notebook_json:
            self.last_accepted_notebook_json = notebook_json
            self.last_accepted_artifact_type = artifact_type or self.last_accepted_artifact_type
            self.last_accepted_modeling_metadata = dict(modeling_metadata or {})
        if code:
            self.last_accepted_code = code
        _log(
            f"[Falsifiable] ⚠️ ACCEPT (no comparable metrics; "
            f"current={current_score}, previous={previous_score})",
            console=True,
        )
        return {
            "verdict": "ACCEPT",
            "metric_delta": 0.0,
            "current_score": current_score,
            "previous_score": previous_score,
            "forced_new_hypothesis": False,
        }

    def _build_history_entry(
        self,
        *,
        iteration: int,
        data: Dict[str, Any],
        verdict_info: Dict[str, Any],
        suggested_target: str,
        technical_feedback: str,
        knowledge_gap: str,
    ) -> Dict[str, Any]:
        """Build a compact history record compatible with legacy review flows."""
        modeling_metadata = data.get("modeling_metadata") or {}
        accuracy = data.get("accuracy")
        try:
            accuracy_val = float(accuracy) if accuracy is not None else None
        except (TypeError, ValueError):
            accuracy_val = None
        verdict = str(verdict_info.get("verdict") or "").upper()
        return {
            "iter": int(iteration),
            "strategy": modeling_metadata.get("selected_strategy") or "",
            "decision": modeling_metadata.get("decision_type") or verdict or "REFINE",
            "focus": modeling_metadata.get("focus_area") or suggested_target,
            "reflection": (technical_feedback or "")[:2000],
            "critique": technical_feedback or "",
            "semantic_gradient": modeling_metadata.get("semantic_gradient_analysis") or "",
            "status": "IMPROVED" if verdict == "ACCEPT" else "REJECTED",
            "score": accuracy_val,
            "metric_delta": verdict_info.get("metric_delta"),
            "target_metric": data.get("target_metric") or "",
            "knowledge_gap": knowledge_gap or "",
            "artifact_type": data.get("artifact_type") or "",
            "tasks": modeling_metadata.get("used_evidence_ids") or [],
            "task_names": modeling_metadata.get("used_evidence_ids") or [],
        }

    def _record_history_entry(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Append *entry* to shared history and return a shallow copy."""
        self.context.results.append(dict(entry))
        return list(self.context.results)

    async def run(self) -> Dict[str, Any]:
        """Execute the full multi-agent pipeline using the FSM.

        Kick-off sequence:
        1. Transition to ``KNOWLEDGE_RETRIEVAL`` and send initial message.
        2. Research → ``biology_insights`` → Modeling (``MODEL_GENERATION``).
        3. Modeling → ``code_execution`` → Execution (``EXECUTING``).
        4. Execution → ``evaluation`` → Evaluation (``EVALUATING``).
        5. Evaluation always publishes to ``orchestration``.
        6. Orchestrator loops on ``orchestration``:
           - ``decision == "SUCCESS"`` → ``TERMINATED``
           - ``decision == "REFINE"`` + ``suggested_target == "research"`` →
             ``FEEDBACK_ROUTING`` → ``KNOWLEDGE_RETRIEVAL`` (re-triggers RAG)
           - ``decision == "REFINE"`` + ``suggested_target == "modeling"`` →
             ``FEEDBACK_ROUTING`` → ``MODEL_GENERATION`` (re-triggers codegen)
           - ``max_iterations_reached`` or timeout → ``TERMINATED``

        Returns:
            A summary dict with ``status``, ``iterations``, ``accuracy``,
            ``best_accuracy``, ``experiment_success_count``, and
            ``total_iterations``.
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
            "history_summary": [],
            "best_metric_score": None,
            # Embed FSM state so agents can include it in their logs.
            "_fsm_state": self.current_state.value if self.current_state else "",
            "_task_id": self.context.task_id,
        }
        # The ResearchAgent subscribes to biology_insights; trigger it.
        await self.bus.publish("biology_insights", initial_message)

        # Telemetry counters.
        TokenMeter.get_and_reset()
        best_accuracy: float = 0.0
        experiment_success_count: int = 0
        total_iterations: int = 0
        execution_stats_log: List[Dict[str, Any]] = []

        pipeline_timeout: float = float(
            (self.config.get("exec") or {}).get("timeout_seconds") or 7200
        )
        _log("[Orchestrator] Pipeline started; entering closed-loop…", console=True)

        loop = asyncio.get_event_loop()
        deadline: float = loop.time() + pipeline_timeout
        final_result: Optional[Dict[str, Any]] = None

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    _log(
                        f"[Orchestrator] ⚠️ Pipeline timed out after "
                        f"{pipeline_timeout:.0f}s.",
                        console=True,
                    )
                    final_result = {"status": "timeout", "data": {}}
                    break

                try:
                    msg = await asyncio.wait_for(
                        orchestration_queue.get(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    _log(
                        f"[Orchestrator] ⚠️ Pipeline timed out waiting on "
                        f"orchestration queue.",
                        console=True,
                    )
                    final_result = {"status": "timeout", "data": {}}
                    break

                status = (msg or {}).get("status", "unknown")
                data = (msg or {}).get("data") or {}
                accuracy = data.get("accuracy")
                decision = data.get("decision")
                raw_metrics_obj = data.get("raw_metrics") if isinstance(data.get("raw_metrics"), dict) else {}
                exec_stats = raw_metrics_obj.get("execution_stats") if isinstance(raw_metrics_obj, dict) else None
                if isinstance(exec_stats, dict):
                    execution_stats_log.append(dict(exec_stats))
                total_iterations += 1

                # Bug 1 fix: always increment self.context.iteration so that
                # EvaluationAgent's `iteration >= max_iterations - 1` guard
                # eventually triggers and prevents an infinite loop.
                self.context.iteration += 1

                # Track execution successes (ExecutionAgent returning success).
                if status == "success":
                    experiment_success_count += 1

                # Update best accuracy and Global Best Checkpoint (Bug 1 fix).
                if accuracy is not None:
                    try:
                        acc_float = float(accuracy)
                        if acc_float > best_accuracy:
                            prev = best_accuracy
                            best_accuracy = acc_float
                            _log(
                                f"📈 [IMPROVEMENT] New best score: {acc_float:.4f} "
                                f"(Prev: {prev:.4f})",
                                console=True,
                            )
                        # Track Global Best Checkpoint.
                        if acc_float > self.best_metric_score:
                            self.best_metric_score = acc_float
                            self.best_code = data.get("code") or ""
                            self.best_notebook_json = data.get("notebook_json") or self.best_notebook_json
                            self.best_artifact_type = data.get("artifact_type") or self.best_artifact_type
                            self.best_modeling_metadata = dict(data.get("modeling_metadata") or self.best_modeling_metadata)
                            self.best_artifacts_path = _interpolate_path(
                                (
                                    (self.config.get("paths") or {}).get(
                                        "generate_execution_root"
                                    )
                                    or "./results/${dataset_name}/generate_execution"
                                ),
                                self.config,
                            )
                    except (TypeError, ValueError):
                        pass

                # ---- Handle ERROR ----
                if status == "error":
                    _log(
                        f"[Orchestrator] ❌ Agent reported error. Full context:\n"
                        f"  agent_id={data.get('agent_id', 'unknown')}\n"
                        f"  error={data.get('error', 'unknown')}\n"
                        f"  traceback={str(data.get('traceback', ''))[:500]}",
                        console=True,
                    )
                    self._enter_state(PipelineState.RETRY_LOGIC)
                    final_result = msg
                    break

                # ---- Handle SUCCESS ----
                if decision == "SUCCESS" or status == "success":
                    _log(
                        f"[Orchestrator] ✅ SUCCESS — accuracy={accuracy}, "
                        f"iterations={total_iterations}",
                        console=True,
                    )
                    self._enter_state(PipelineState.TERMINATED)
                    final_result = msg
                    break

                # ---- Handle max iterations exhausted ----
                if data.get("max_iterations_reached"):
                    _log(
                        f"[Orchestrator] ⚠️ Max iterations reached; "
                        f"returning Global Best (score={self.best_metric_score:.4f}).",
                        console=True,
                    )
                    self._enter_state(PipelineState.TERMINATED)
                    # Return the historically best result instead of the last
                    # (possibly regressed) one.
                    best_data = dict(data)
                    if self.best_code:
                        best_data["code"] = self.best_code
                    if self.best_notebook_json:
                        best_data["notebook_json"] = self.best_notebook_json
                    if self.best_artifact_type:
                        best_data["artifact_type"] = self.best_artifact_type
                    if self.best_modeling_metadata:
                        best_data["modeling_metadata"] = self.best_modeling_metadata
                    if self.best_metric_score > -float("inf"):
                        best_data["accuracy"] = self.best_metric_score
                        best_data["best_metric_score"] = self.best_metric_score
                    best_data["best_artifacts_path"] = self.best_artifacts_path
                    final_result = {"status": "terminated", "data": best_data}
                    break

                # ---- Handle REFINE ----
                if decision == "REFINE":
                    self._enter_state(PipelineState.FEEDBACK_ROUTING)
                    feedback_package = data.get("feedback_package") or {}
                    suggested_target = feedback_package.get("suggested_target", "modeling")
                    technical_feedback = feedback_package.get("technical_feedback", "")
                    knowledge_gap = feedback_package.get("knowledge_gap", "")
                    # next_iteration uses self.context.iteration which was already
                    # incremented above (Bug 1 fix).
                    next_iteration = self.context.iteration

                    # ---- Falsifiable Iteration Protocol ----
                    # Evaluate this iteration vs. the last accepted state.
                    verdict_info = self._evaluate_iteration(data, next_iteration)
                    verdict = verdict_info["verdict"]

                    # On REJECT: revert the working artifact to the last accepted
                    # notebook/code state and augment feedback to force a new idea.
                    code_for_next = data.get("code") or ""
                    notebook_for_next = data.get("notebook_json") or ""
                    artifact_type_for_next = data.get("artifact_type") or ""
                    modeling_metadata_for_next = dict(data.get("modeling_metadata") or {})
                    if verdict == "REJECT":
                        if self.last_accepted_code:
                            code_for_next = self.last_accepted_code
                        if self.last_accepted_notebook_json:
                            notebook_for_next = self.last_accepted_notebook_json
                        if self.last_accepted_artifact_type:
                            artifact_type_for_next = self.last_accepted_artifact_type
                        if self.last_accepted_modeling_metadata:
                            modeling_metadata_for_next = dict(self.last_accepted_modeling_metadata)
                        prev_score = verdict_info.get("previous_score")
                        curr_score = verdict_info.get("current_score")
                        if prev_score is not None and curr_score is not None:
                            revert_note = (
                                f"[FALSIFICATION] The previous change DEGRADED "
                                f"the metric ({prev_score:.4f} → {curr_score:.4f}). "
                                f"The working artifact has been REVERTED to the last accepted state. "
                                f"Please try a fundamentally different approach."
                            )
                        else:
                            revert_note = (
                                "[FALSIFICATION] The previous change was rejected. "
                                "The working artifact has been REVERTED to the last accepted state. "
                                "Please try a fundamentally different approach."
                            )
                        technical_feedback = f"{revert_note}\n\n{technical_feedback}".strip()

                    # On consecutive rejections: force route to research for a
                    # new biological hypothesis rather than more code tweaks.
                    if verdict_info.get("forced_new_hypothesis"):
                        suggested_target = "research"
                        knowledge_gap = (
                            knowledge_gap
                            or "novel biological mechanism for cell perturbation "
                            "response prediction beyond current approach"
                        )

                    history_entry = self._build_history_entry(
                        iteration=next_iteration,
                        data=data,
                        verdict_info=verdict_info,
                        suggested_target=suggested_target,
                        technical_feedback=technical_feedback,
                        knowledge_gap=knowledge_gap,
                    )
                    history_summary = self._record_history_entry(history_entry)
                    current_metrics_payload = data.get("raw_metrics") or data.get("metrics") or {}
                    previous_iteration_logs = {
                        "stdout": data.get("stdout") or "",
                        "stderr": data.get("stderr") or "",
                        "traceback": data.get("traceback") or "",
                        "metrics": data.get("metrics") or {},
                        "raw_metrics": data.get("raw_metrics") or {},
                        "code": code_for_next,
                        "notebook_json": notebook_for_next,
                        "artifact_type": artifact_type_for_next,
                        "modeling_metadata": modeling_metadata_for_next,
                    }

                    # Save iteration artifacts.
                    artifact_data = dict(data)
                    artifact_data["falsifiable_verdict"] = verdict_info
                    artifact_data["history_entry"] = history_entry
                    artifact_data["history_summary"] = history_summary
                    self._save_iteration_artifacts(
                        iteration=int(data.get("iteration") or 0),
                        agent_name="evaluation",
                        data=artifact_data,
                    )

                    if suggested_target == "research":
                        self._enter_state(PipelineState.KNOWLEDGE_RETRIEVAL)
                        _log(
                            f"[Orchestrator] 🔄 Routing to ResearchAgent (knowledge_gap='{knowledge_gap}')",
                            console=True,
                        )
                        await self.bus.publish(
                            "biology_insights",
                            {
                                "smiles_list": self.context.smiles_list,
                                "h5_file_path": self.context.h5_file_path,
                                "context_text": technical_feedback,
                                "knowledge_gap": knowledge_gap,
                                "stage": "refinement",
                                "iteration": next_iteration,
                                "max_iterations": self.context.max_iterations,
                                "previous_iteration_logs": previous_iteration_logs,
                                "history_summary": history_summary,
                                "best_metric_score": self.best_metric_score if self.best_metric_score > -float("inf") else None,
                                "falsifiable_verdict": verdict_info,
                                "_fsm_state": self.current_state.value if self.current_state else "",
                                "_task_id": self.context.task_id,
                            },
                        )
                    else:
                        self._enter_state(PipelineState.MODEL_GENERATION)
                        _log(
                            f"[Orchestrator] 🔄 Routing to ModelingAgent (feedback='{technical_feedback[:80]}…')",
                            console=True,
                        )
                        await self.bus.publish(
                            "modeling",
                            {
                                "biological_insight_report": data.get("insight_report") or {},
                                "error_logs": data.get("stderr") or data.get("traceback") or "",
                                "code": code_for_next,
                                "notebook_json": notebook_for_next,
                                "artifact_type": artifact_type_for_next or data.get("artifact_type") or "",
                                "modeling_metadata": modeling_metadata_for_next,
                                "current_metrics": current_metrics_payload,
                                "technical_feedback": technical_feedback,
                                "history_summary": history_summary,
                                "best_metric_score": self.best_metric_score if self.best_metric_score > -float("inf") else None,
                                "iteration": next_iteration,
                                "max_iterations": self.context.max_iterations,
                                "falsifiable_verdict": verdict_info,
                                "_fsm_state": self.current_state.value if self.current_state else "",
                                "_task_id": self.context.task_id,
                            },
                        )
                    continue

                # Unknown message — terminate to avoid infinite loop.
                _log(
                    f"[Orchestrator] ⚠️ Unknown orchestration message "
                    f"(status={status}, decision={decision}). Terminating.",
                    console=True,
                )
                final_result = msg
                break

        finally:
            # Cancel all agent background tasks.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Ensure we always reach TERMINATED.
        if self.current_state != PipelineState.TERMINATED:
            self._enter_state(PipelineState.TERMINATED)

        status_final = (final_result or {}).get("status", "unknown")
        data_final = (final_result or {}).get("data") or {}
        accuracy_final = data_final.get("accuracy") or (best_accuracy if best_accuracy > 0 else None)

        _log(
            f"🏁 LOOP FINISHED | Success: {experiment_success_count}/{total_iterations} "
            f"| Best PCC: {best_accuracy:.4f}",
            console=True,
        )

        # Extended telemetry / robustness / explainability metrics.
        token_usage = TokenMeter.get_and_reset()
        prompt_toks = int(token_usage.get("prompt_tokens", 0) or 0)
        completion_toks = int(token_usage.get("completion_tokens", 0) or 0)
        total_toks = int(token_usage.get("total_tokens", prompt_toks + completion_toks) or 0)
        total_latency = float(token_usage.get("total_latency_sec", 0.0) or 0.0)
        api_calls = int(token_usage.get("api_calls", 0) or 0)

        nb_success_n = sum(1 for e in execution_stats_log if e.get("notebook_success") is True)
        nb_total_n = len(execution_stats_log)
        autofix_attempted = sum(int(e.get("autofix_attempted_cells", 0) or 0) for e in execution_stats_log)
        autofix_success = sum(int(e.get("autofix_success_cells", 0) or 0) for e in execution_stats_log)
        avg_fix_rounds_vals = [float(e.get("avg_fix_rounds", 0.0) or 0.0) for e in execution_stats_log if e.get("avg_fix_rounds") is not None]
        crash_recovered_flags = [1 if e.get("framework_recovered") else 0 for e in execution_stats_log]
        validity_rate = (sum(1 for h in self.iteration_history if h.get("score") is not None) / float(len(self.iteration_history))) if self.iteration_history else 0.0

        all_tasks = []
        for r in self.context.results:
            ts = r.get("tasks") or []
            if isinstance(ts, list):
                all_tasks.extend([str(t) for t in ts if str(t).strip()])
        unique_tasks = sorted(set(all_tasks))
        evidence_chain_completeness = (sum(1 for r in self.context.results if isinstance(r.get("tasks"), list) and len(r.get("tasks") or []) > 0) / float(len(self.context.results))) if self.context.results else 0.0
        external_knowledge_coverage = (len(unique_tasks) / float(len(all_tasks))) if all_tasks else 0.0

        # Task-graph evolution stability proxy: mean Jaccard between consecutive used-evidence sets.
        j_scores = []
        prev_set = None
        for r in self.context.results:
            cur_set = set([str(t) for t in (r.get("tasks") or []) if str(t).strip()])
            if prev_set is not None:
                union = prev_set | cur_set
                inter = prev_set & cur_set
                j_scores.append((len(inter) / float(len(union))) if union else 1.0)
            prev_set = cur_set
        task_graph_evolution_stability = (sum(j_scores) / float(len(j_scores))) if j_scores else 0.0

        summary = {
            "status": status_final,
            "task_id": self.context.task_id,
            "iterations": self.context.iteration,
            "total_iterations": total_iterations,
            "accuracy": accuracy_final,
            "best_accuracy": best_accuracy,
            "best_metric_score": self.best_metric_score if self.best_metric_score > -float("inf") else None,
            "best_code": self.best_code,
            "best_notebook_json": self.best_notebook_json,
            "best_artifact_type": self.best_artifact_type,
            "best_modeling_metadata": self.best_modeling_metadata,
            "best_artifacts_path": self.best_artifacts_path,
            "experiment_success_count": experiment_success_count,
            "max_iterations_reached": data_final.get("max_iterations_reached", False),
            "robustness_metrics": {
                "notebook_execution_success_rate": (nb_success_n / float(nb_total_n)) if nb_total_n > 0 else 0.0,
                "cell_fix_success_rate": (autofix_success / float(autofix_attempted)) if autofix_attempted > 0 else 0.0,
                "avg_fix_rounds": (sum(avg_fix_rounds_vals) / float(len(avg_fix_rounds_vals))) if avg_fix_rounds_vals else 0.0,
                "crash_recovery_ratio": (sum(crash_recovered_flags) / float(len(crash_recovered_flags))) if crash_recovered_flags else 0.0,
                "validity_rate": validity_rate,
            },
            "resource_cost_metrics": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": total_toks,
                "total_llm_latency_sec": total_latency,
                "avg_prompt_tokens": (prompt_toks / float(total_iterations)) if total_iterations > 0 else 0.0,
                "avg_completion_tokens": (completion_toks / float(total_iterations)) if total_iterations > 0 else 0.0,
                "avg_llm_latency_sec": (total_latency / float(api_calls)) if api_calls > 0 else 0.0,
                "wall_clock_to_threshold_sec": None,
                "cost_to_success_ratio": (total_toks / float(experiment_success_count)) if experiment_success_count > 0 else None,
            },
            "scientific_interpretability_metrics": {
                "evidence_chain_completeness": evidence_chain_completeness,
                "external_knowledge_coverage": external_knowledge_coverage,
                "bioprocess_mapping_consistency": task_graph_evolution_stability,
                "expert_mechanism_score": None,
                "task_graph_evolution_stability": task_graph_evolution_stability,
                "unique_evidence_ids": unique_tasks,
            },
            "iteration_history": self.iteration_history,
            "consecutive_rejections": self.consecutive_rejections,
            "fsm_transitions": [
                {"from": f.value if f else "None", "to": t.value if t else "None", "timestamp": ts}
                for f, t, ts in self.transition_history
            ],
        }

        _log(
            f"[Orchestrator] Pipeline finished: status={status_final}, "
            f"best_accuracy={best_accuracy:.4f}, "
            f"success_count={experiment_success_count}/{total_iterations}",
            console=True,
        )

        robustness_metrics = summary.get("robustness_metrics") or {}
        resource_cost_metrics = summary.get("resource_cost_metrics") or {}
        scientific_interpretability_metrics = summary.get("scientific_interpretability_metrics") or {}

        _log("[Metrics][7.2] System Execution Robustness", console=True)
        _log(
            f"├─ notebook_execution_success_rate={float(robustness_metrics.get('notebook_execution_success_rate', 0.0)):.4f}",
            console=True,
        )
        _log(
            f"├─ cell_fix_success_rate={float(robustness_metrics.get('cell_fix_success_rate', 0.0)):.4f}",
            console=True,
        )
        _log(
            f"├─ avg_fix_rounds={float(robustness_metrics.get('avg_fix_rounds', 0.0)):.4f}",
            console=True,
        )
        _log(
            f"├─ crash_recovery_ratio={float(robustness_metrics.get('crash_recovery_ratio', 0.0)):.4f}",
            console=True,
        )
        _log(
            f"└─ validity_rate={float(robustness_metrics.get('validity_rate', 0.0)):.4f}",
            console=True,
        )

        _log("[Metrics][7.3] Resource & Cost", console=True)
        _log(f"├─ prompt_tokens={int(resource_cost_metrics.get('prompt_tokens', 0) or 0)}", console=True)
        _log(f"├─ completion_tokens={int(resource_cost_metrics.get('completion_tokens', 0) or 0)}", console=True)
        _log(f"├─ total_tokens={int(resource_cost_metrics.get('total_tokens', 0) or 0)}", console=True)
        _log(
            f"├─ total_llm_latency_sec={float(resource_cost_metrics.get('total_llm_latency_sec', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"├─ avg_prompt_tokens={float(resource_cost_metrics.get('avg_prompt_tokens', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"├─ avg_completion_tokens={float(resource_cost_metrics.get('avg_completion_tokens', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"├─ avg_llm_latency_sec={float(resource_cost_metrics.get('avg_llm_latency_sec', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"├─ wall_clock_to_threshold_sec={resource_cost_metrics.get('wall_clock_to_threshold_sec')}",
            console=True,
        )
        _log(
            f"└─ cost_to_success_ratio={resource_cost_metrics.get('cost_to_success_ratio')}",
            console=True,
        )

        _log("[Metrics][7.4] Scientific Interpretability", console=True)
        _log(
            f"├─ evidence_chain_completeness={float(scientific_interpretability_metrics.get('evidence_chain_completeness', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"├─ external_knowledge_coverage={float(scientific_interpretability_metrics.get('external_knowledge_coverage', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"├─ bioprocess_mapping_consistency={float(scientific_interpretability_metrics.get('bioprocess_mapping_consistency', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"├─ expert_mechanism_score={scientific_interpretability_metrics.get('expert_mechanism_score')}",
            console=True,
        )
        _log(
            f"├─ task_graph_evolution_stability={float(scientific_interpretability_metrics.get('task_graph_evolution_stability', 0.0) or 0.0):.4f}",
            console=True,
        )
        _log(
            f"└─ unique_evidence_ids={len(scientific_interpretability_metrics.get('unique_evidence_ids') or [])}",
            console=True,
        )

        return summary

    # ------------------------------------------------------------------
    # Artifact persistence helpers
    # ------------------------------------------------------------------

    def _save_iteration_artifacts(
        self,
        iteration: int,
        agent_name: str,
        data: Dict[str, Any],
    ) -> None:
        """Persist agent output for a given iteration.

        Saves to ``runs/{task_id}/{iteration}/{agent_name}/`` using
        :class:`~.workspace_manager.WorkspaceManager`.

        Args:
            iteration: Current iteration number.
            agent_name: Name of the agent producing the data.
            data: Arbitrary payload to persist as JSON.
        """
        try:
            from .workspace_manager import WorkspaceManager  # type: ignore

            wm = WorkspaceManager(self.context.task_id)
            filename = f"artifacts_{agent_name}_iter{iteration}.json"
            wm.ensure_dirs()
            wm.save_artifact("execution", filename, data)
        except Exception as exc:
            _log(f"[Orchestrator] Could not save iteration artifacts: {exc}")


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
