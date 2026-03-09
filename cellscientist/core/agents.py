# -*- coding: utf-8 -*-
"""Multi-Agent framework for CellCausal virtual cell pipeline.

This module defines the agent hierarchy used by the Multi-Agent Orchestration
system:

- :class:`TaskContext` — shared workflow state
- :class:`AgentResponse` — validated JSON response envelope
- :class:`BaseAgent` — abstract base with retry logic and bus integration
- :class:`ResearchAgent` — biological knowledge retrieval
- :class:`ModelingAgent` — PyTorch/GNN code generation with self-correction
- :class:`ExecutionAgent` — subprocess-based code execution
- :class:`EvaluationAgent` — result evaluation against accuracy goals

All inter-agent communication goes through a :class:`~.message_bus.SimpleMessageBus`.
"""

from __future__ import annotations

import abc
import asyncio
import functools
import json
import logging
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .message_bus import SimpleMessageBus
from .agent_monitor import monitor_agent, Heartbeat

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Logging helper (consistent with other core modules)
# =============================================================================


def _log(msg: str, *, console: bool = False) -> None:
    """Print a log message with the standard CellCausal prefix.

    Args:
        msg: Message text.
        console: If True uses ``[CELL_CONSOLE]`` prefix; otherwise ``[DETAIL]``.
    """
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    else:
        print(f"[DETAIL] {msg}", flush=True)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class AgentResponse:
    """Validated JSON envelope returned by every agent.

    Attributes:
        status: One of ``"success"``, ``"error"``, or ``"needs_iteration"``.
        data: Arbitrary result payload.
        next_recipient: Topic name where the response should be published.
    """

    status: str  # "success" | "error" | "needs_iteration"
    data: Dict[str, Any]
    next_recipient: str

    # Allowed status values.
    _VALID_STATUSES = frozenset({"success", "error", "needs_iteration"})

    def __post_init__(self) -> None:
        if self.status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid AgentResponse status '{self.status}'. "
                f"Must be one of {sorted(self._VALID_STATUSES)}."
            )
        if not isinstance(self.data, dict):
            raise TypeError("AgentResponse.data must be a dict.")
        if not isinstance(self.next_recipient, str) or not self.next_recipient:
            raise ValueError("AgentResponse.next_recipient must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "status": self.status,
            "data": self.data,
            "next_recipient": self.next_recipient,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentResponse":
        """Deserialise from a plain dict.

        Args:
            d: Dict with ``status``, ``data``, and ``next_recipient`` keys.

        Returns:
            A validated :class:`AgentResponse` instance.
        """
        return cls(
            status=d["status"],
            data=d.get("data") or {},
            next_recipient=d["next_recipient"],
        )


@dataclass
class TaskContext:
    """Shared workflow state passed through the agent pipeline.

    Attributes:
        task_id: Unique identifier for this pipeline run.
        smiles_list: SMILES strings loaded from the H5 file.
        h5_file_path: Resolved path to the H5 data file.
        iteration: Current iteration counter (starts at 0).
        current_stage: Name of the stage currently executing.
        results: Accumulated results from previous iterations.
        max_iterations: Maximum number of refinement iterations allowed.
        config: Full pipeline configuration dict.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    smiles_list: List[str] = field(default_factory=list)
    h5_file_path: str = ""
    iteration: int = 0
    current_stage: str = "init"
    results: List[Dict[str, Any]] = field(default_factory=list)
    max_iterations: int = 5
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "TaskContext":
        """Construct a :class:`TaskContext` from a pipeline config dict.

        Args:
            config: Full pipeline configuration (``pipeline_config.json``).

        Returns:
            A populated :class:`TaskContext`.
        """
        # Lazily import to avoid hard dependency when not using smiles_resolver.
        try:
            from .smiles_resolver import resolve_smiles, validate_h5_smiles_path
            h5_path = validate_h5_smiles_path(config)
            smiles = resolve_smiles(config)
        except Exception as exc:
            _log(
                f"[TaskContext] ⚠️ Could not load SMILES from H5 file: {exc}. "
                "Continuing with empty SMILES list.",
                console=False,
            )
            h5_path = (config.get("paths") or {}).get("data_h5_path") or ""
            smiles = []

        max_iter = (
            (config.get("review") or {}).get("max_iterations")
            or (config.get("experiment") or {}).get("max_iterations")
            or 5
        )

        return cls(
            smiles_list=smiles,
            h5_file_path=h5_path,
            max_iterations=int(max_iter),
            config=config,
        )


# =============================================================================
# Retry decorator for transient LLM/network errors
# =============================================================================


def retry_llm_call(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
):
    """Decorator that retries an async coroutine on transient network errors.

    Uses exponential back-off between attempts.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before the first retry.
        backoff: Multiplier applied to delay after each retry.

    Returns:
        Decorator function.
    """

    def decorator(func):  # type: ignore[no-untyped-def]
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError, OSError) as exc:
                    last_exc = exc
                    _log(
                        f"[retry_llm_call] Transient error on attempt "
                        f"{attempt + 1}/{max_retries}: {exc}. "
                        f"Retrying in {delay:.1f}s…"
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff
            raise RuntimeError(
                f"LLM call failed after {max_retries} attempts. "
                f"Last error: {last_exc}"
            )

        return wrapper

    return decorator


# =============================================================================
# BaseAgent
# =============================================================================


class BaseAgent(abc.ABC):
    """Abstract base class for all pipeline agents.

    Subclasses must implement :meth:`process`.

    Attributes:
        role: Human-readable role name (e.g., ``"biological_researcher"``).
        tools: List of tool names this agent uses.
        agent_id: Unique identifier for this agent instance.
        bus: Reference to the shared :class:`~.message_bus.SimpleMessageBus`.
    """

    role: str = "base_agent"
    tools: List[str] = []

    def __init__(self, bus: SimpleMessageBus, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the agent.

        Args:
            bus: The shared message bus instance.
            config: Optional full pipeline configuration dict.
        """
        self.agent_id: str = f"{self.role}_{uuid.uuid4().hex[:8]}"
        self.bus: SimpleMessageBus = bus
        self.config: Dict[str, Any] = config or {}

    @abc.abstractmethod
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Process an incoming message and return a validated response.

        Args:
            message: Arbitrary payload delivered from the message bus.

        Returns:
            A :class:`AgentResponse` with ``status``, ``data``, and
            ``next_recipient`` populated.
        """

    async def run(self, inbox: asyncio.Queue) -> None:
        """Consume messages from *inbox* and process each one.

        Published results are forwarded to the :attr:`bus`.

        Args:
            inbox: Queue supplying incoming messages for this agent.
        """
        _log(f"[{self.agent_id}] Agent started, waiting for messages…")
        while True:
            message: Dict[str, Any] = await inbox.get()
            _log(f"[{self.agent_id}] Received message (stage={message.get('stage', '?')})")
            try:
                response = await self.process(message)
                await self.bus.publish(response.next_recipient, response.to_dict())
                _log(
                    f"[{self.agent_id}] Published to '{response.next_recipient}' "
                    f"(status={response.status})"
                )
            except Exception as exc:
                _log(
                    f"[{self.agent_id}] ❌ Unhandled error: {exc}",
                    console=True,
                )
                logger.exception("[%s] Unhandled error in run loop.", self.agent_id)
                try:
                    await self.bus.publish(
                        "orchestration",
                        {
                            "status": "error",
                            "data": {
                                "error": str(exc),
                                "agent_id": self.agent_id,
                                "traceback": traceback.format_exc(),
                            },
                            "next_recipient": "orchestration",
                        },
                    )
                except Exception as pub_exc:
                    _log(
                        f"[{self.agent_id}] ❌ Could not publish error to orchestration: {pub_exc}",
                        console=True,
                    )
            finally:
                inbox.task_done()


# =============================================================================
# ResearchAgent
# =============================================================================


class ResearchAgent(BaseAgent):
    """Performs biological knowledge retrieval for a set of SMILES strings.

    Tools: BioKB semantic enrichment + Serper web search (with Jina fallback).

    Publishes a "Biological Insight Report" to the ``biology_insights`` topic.
    """

    role = "biological_researcher"
    tools = ["BioKB", "Serper_Search"]

    @monitor_agent
    @retry_llm_call()
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Analyse SMILES and contextual data to produce a biology report.

        Args:
            message: Payload that should contain ``smiles_list``, ``h5_file_path``,
                ``context_text``, and optionally ``knowledge_gap`` for targeted
                second-pass RAG retrieval.

        Returns:
            :class:`AgentResponse` directed to ``biology_insights``.
        """
        smiles_list: List[str] = message.get("smiles_list") or []
        h5_file_path: str = message.get("h5_file_path") or ""
        context_text: str = message.get("context_text") or ""
        knowledge_gap: str = message.get("knowledge_gap") or ""

        # Use "refinement" stage for second-pass RAG, "design" for initial pass.
        stage: str = "refinement" if knowledge_gap else (message.get("stage") or "design")

        if knowledge_gap:
            _log(
                f"[RAG] 📚 Second-pass RAG targeting: '{knowledge_gap}'",
                console=True,
            )
        else:
            _log(
                f"[RAG] 📚 ResearchAgent is searching literature for "
                f"{len(smiles_list)} SMILES compounds…",
                console=True,
            )

        _log(
            f"[{self.agent_id}] Retrieving biology insights for "
            f"{len(smiles_list)} SMILES (stage={stage})…",
            console=True,
        )

        # Attempt BioKB + literature retrieval; degrade gracefully on errors.
        bio_kb_data: Dict[str, Any] = {}
        literature_data: Dict[str, Any] = {}

        # BioKB enrichment
        try:
            from .bio_kb import generate_biokb_semantic_table  # type: ignore

            bio_kb_cfg = (self.config.get("literature") or {}).get("bio_kb") or {}
            if bio_kb_cfg.get("enabled", True) and smiles_list:
                bio_kb_data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: generate_biokb_semantic_table(smiles_list, bio_kb_cfg),
                )
        except Exception as exc:
            _log(f"[{self.agent_id}] BioKB enrichment skipped: {exc}")

        # Web / literature retrieval via external_knowledge_mirothink.
        # When knowledge_gap is provided, use it as both the query text and
        # the query_hint so that the retrieval is targeted.
        try:
            from .external_knowledge_mirothink import (  # type: ignore
                retrieve_external_knowledge,
                knowledge_pack_to_markdown,
            )

            query_text = knowledge_gap if knowledge_gap else (context_text or " ".join(smiles_list[:5]))
            pack = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retrieve_external_knowledge(
                    self.config,
                    query_text,
                    stage,
                    query_hint=knowledge_gap if knowledge_gap else None,
                ),
            )
            literature_data = {"markdown_summary": knowledge_pack_to_markdown(pack)}
        except Exception as exc:
            _log(f"[{self.agent_id}] Literature retrieval skipped: {exc}")

        report = {
            "smiles_list": smiles_list,
            "h5_file_path": h5_file_path,
            "bio_kb_summary": bio_kb_data,
            "literature_summary": literature_data,
            "context_text": context_text,
            "knowledge_gap": knowledge_gap,
        }

        return AgentResponse(
            status="success",
            data={"biological_insight_report": report},
            next_recipient="modeling",
        )


# =============================================================================
# ModelingAgent
# =============================================================================


class ModelingAgent(BaseAgent):
    """Generates and revises PyTorch/GNN model code.

    Subscribes to ``biology_insights`` and ``evaluation`` topics.
    Publishes generated code to ``code_execution``.
    """

    role = "model_architect"
    tools = ["LLM_codegen"]

    @monitor_agent
    @retry_llm_call()
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Generate or revise model code based on biology insights or error logs.

        Args:
            message: May contain ``biological_insight_report`` (from
                :class:`ResearchAgent`) or ``error_logs`` (from
                :class:`ExecutionAgent`) for self-correction.

        Returns:
            :class:`AgentResponse` directed to ``code_execution``.
        """
        insight_report: Dict[str, Any] = message.get("biological_insight_report") or {}
        raw_error_logs: Optional[str] = message.get("error_logs")
        existing_code: Optional[str] = message.get("code")
        technical_feedback: str = message.get("technical_feedback") or ""

        # Merge error_logs with technical_feedback from EvaluationAgent when present.
        if technical_feedback and raw_error_logs:
            error_logs: Optional[str] = f"{raw_error_logs}\n\n[EvaluationAgent Feedback]\n{technical_feedback}"
        elif technical_feedback:
            error_logs = f"[EvaluationAgent Feedback]\n{technical_feedback}"
        else:
            error_logs = raw_error_logs

        iteration: int = int(message.get("iteration") or 0)
        if error_logs:
            action_desc = f"refining GNN architecture based on iteration {iteration} feedback"
        else:
            action_desc = "generating initial model code"
        _log(f"[CODE] 💻 ModelingAgent is {action_desc}…", console=True)

        # Build a prompt for the LLM using PromptManager (falls back to hardcoded defaults).
        smiles_list: List[str] = insight_report.get("smiles_list") or []
        literature = (insight_report.get("literature_summary") or {}).get("markdown_summary") or ""

        try:
            from .prompt_manager import get_default_prompt_manager  # type: ignore

            pm = get_default_prompt_manager()
            if error_logs and existing_code:
                system_prompt = pm.get_prompt("modeling_agent", "self_correction_system_prompt")
                user_content = pm.get_prompt(
                    "modeling_agent",
                    "self_correction_template",
                    error_logs=error_logs,
                    existing_code=existing_code,
                )
            else:
                system_prompt = pm.get_prompt("modeling_agent", "system_prompt")
                user_content = pm.get_prompt(
                    "modeling_agent",
                    "code_generation_template",
                    literature=literature,
                    smiles_list=str(smiles_list[:10]),
                )
        except Exception:
            # Hard fallback if PromptManager is unavailable.
            if error_logs and existing_code:
                system_prompt = (
                    "You are an expert PyTorch/GNN engineer. "
                    "The code below raised errors. Fix ALL errors and return ONLY the corrected Python code."
                )
                user_content = (
                    f"## Error Logs\n{error_logs}\n\n"
                    f"## Existing Code\n```python\n{existing_code}\n```"
                )
            else:
                system_prompt = (
                    "You are an expert PyTorch/GNN engineer specialising in cell perturbation modeling. "
                    "Write complete, runnable Python code for a GNN model that predicts cell painting "
                    "perturbation responses. Return ONLY the Python code."
                )
                user_content = (
                    f"## Biological Context\n{literature}\n\n"
                    f"## SMILES Compounds (sample)\n{smiles_list[:10]}"
                )

        code: str = ""
        try:
            from .llm_client import chat_text, resolve_llm_config  # type: ignore

            llm_cfg = resolve_llm_config(self.config)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            code = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: chat_text(messages, llm_config=llm_cfg),
            )
        except Exception as exc:
            _log(f"[{self.agent_id}] LLM code generation failed: {exc}")
            return AgentResponse(
                status="error",
                data={"error": str(exc)},
                next_recipient="code_execution",
            )

        return AgentResponse(
            status="success",
            data={
                "code": code,
                "insight_report": insight_report,
                "iteration": iteration,
                "max_iterations": int(message.get("max_iterations") or 5),
                "_task_id": message.get("_task_id") or "",
            },
            next_recipient="code_execution",
        )


# =============================================================================
# ExecutionAgent
# =============================================================================


class ExecutionAgent(BaseAgent):
    """Executes generated Python code in a subprocess.

    Subscribes to ``code_execution`` topic.
    Publishes execution results (stdout, stderr, traceback) to ``evaluation``.
    """

    role = "code_executor"
    tools = ["subprocess"]

    @monitor_agent
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Run the code contained in *message* and capture all output.

        Args:
            message: Must contain a ``code`` key with Python source to execute.

        Returns:
            :class:`AgentResponse` directed to ``evaluation`` with
            ``stdout``, ``stderr``, and ``traceback`` fields.
        """
        code: str = message.get("code") or ""
        insight_report: Dict[str, Any] = message.get("insight_report") or {}

        if not code.strip():
            return AgentResponse(
                status="error",
                data={
                    "error": "No code provided for execution.",
                    "stdout": "",
                    "stderr": "",
                    "traceback": "",
                    "insight_report": insight_report,
                },
                next_recipient="evaluation",
            )

        _log(f"[EXEC] ⚙️ ExecutionAgent is running code in Sandbox ({len(code)} chars)…", console=True)

        timeout: int = int(
            (self.config.get("exec") or {}).get("timeout_seconds") or 300
        )
        iteration: int = int(message.get("iteration") or 0)
        task_id: str = str(message.get("_task_id") or "")

        stdout_val = ""
        stderr_val = ""
        tb_val = ""
        success = False

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise TimeoutError(
                    f"Code execution timed out after {timeout}s."
                )

            stdout_val = raw_stdout.decode("utf-8", errors="replace")
            stderr_val = raw_stderr.decode("utf-8", errors="replace")
            success = proc.returncode == 0

            if not success:
                tb_val = stderr_val

        except Exception as exc:
            tb_val = traceback.format_exc()
            stderr_val = str(exc)
            _log(f"[{self.agent_id}] ❌ Execution error: {exc}", console=True)

        status_val = "success" if success else "error"
        _log(
            f"[{self.agent_id}] Execution {'✅ succeeded' if success else '❌ failed'}.",
            console=True,
        )

        # Build and persist a structured error report when execution fails.
        if not success:
            # Extract actual error type from the traceback string if available.
            error_type = "ExecutionError"
            if tb_val:
                # Traceback lines end with: ExceptionClass: message
                for line in reversed(tb_val.strip().splitlines()):
                    stripped = line.strip()
                    if stripped and ":" in stripped and not stripped.startswith(" "):
                        error_type = stripped.split(":")[0].strip()
                        break
            self._save_error_report(
                task_id=task_id,
                iteration=iteration,
                error_type=error_type,
                error_message=stderr_val,
                tb=tb_val,
                code=code,
            )

        return AgentResponse(
            status=status_val,
            data={
                "stdout": stdout_val,
                "stderr": stderr_val,
                "traceback": tb_val,
                "code": code,
                "insight_report": insight_report,
                "iteration": iteration,
            },
            next_recipient="evaluation",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_error_report(
        self,
        task_id: str,
        iteration: int,
        error_type: str,
        error_message: str,
        tb: str,
        code: str,
    ) -> None:
        """Persist a structured JSON error report via :class:`WorkspaceManager`.

        The report can be read by :class:`ModelingAgent` for self-correction.

        Args:
            task_id: Pipeline task identifier.
            iteration: Current iteration number.
            error_type: Exception class name.
            error_message: Short error description.
            tb: Full traceback string.
            code: The Python source that failed.
        """
        if not task_id:
            return
        try:
            from .workspace_manager import WorkspaceManager  # type: ignore

            wm = WorkspaceManager(task_id)
            wm.ensure_dirs()
            report = {
                "error_type": error_type,
                "error_message": error_message,
                "traceback": tb,
                "agent": self.__class__.__name__,
                "state": "EXECUTING",
                "timestamp": _now_iso(),
                "context": {
                    "task_id": task_id,
                    "iteration": iteration,
                    "code_snippet": code[:500],
                },
            }
            wm.save_artifact(
                "execution",
                f"error_report_iter_{iteration}.json",
                report,
            )
        except Exception as exc:
            _log(f"[{self.agent_id}] Could not save error report: {exc}")


# =============================================================================
# EvaluationAgent
# =============================================================================


class EvaluationAgent(BaseAgent):
    """Evaluates execution results and decides whether to iterate.

    Subscribes to ``evaluation`` topic.
    Always publishes to ``orchestration`` — including REFINE decisions — so the
    orchestrator can update the FSM and route feedback to the correct agent.

    When the goal is not yet met, uses the LLM to generate a structured
    ``feedback_package`` containing:

    * ``technical_feedback`` — actionable guidance for :class:`ModelingAgent`
    * ``knowledge_gap`` — topic for :class:`ResearchAgent` to search
    * ``suggested_target`` — ``"modeling"`` or ``"research"``
    """

    role = "evaluator"
    tools = ["metrics_parser"]

    # Target accuracy threshold (90 %).
    ACCURACY_GOAL: float = 0.90

    @monitor_agent
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Assess execution output and determine next action.

        Args:
            message: Payload from :class:`ExecutionAgent` containing
                ``stdout``, ``stderr``, ``traceback``, ``code``, and
                ``insight_report``.

        Returns:
            :class:`AgentResponse` directed to ``orchestration`` with
            ``decision`` set to ``"SUCCESS"`` or ``"REFINE"``.  When
            ``"REFINE"``, a ``feedback_package`` dict is included.
        """
        stdout: str = message.get("stdout") or ""
        stderr: str = message.get("stderr") or ""
        tb: str = message.get("traceback") or ""
        code: str = message.get("code") or ""
        insight_report: Dict[str, Any] = message.get("insight_report") or {}
        iteration: int = int(message.get("iteration") or 0)
        max_iterations: int = int(message.get("max_iterations") or 5)

        _log(
            f"[EVAL] 📊 EvaluationAgent is comparing results against "
            f"{self.ACCURACY_GOAL:.0%} accuracy goal "
            f"(iteration {iteration}/{max_iterations})…",
            console=True,
        )

        # Attempt to parse an accuracy/PCC metric from stdout.
        accuracy = self._parse_accuracy(stdout)

        _log(
            f"[{self.agent_id}] Detected accuracy: "
            f"{'N/A' if accuracy is None else f'{accuracy:.3f}'} "
            f"(goal={self.ACCURACY_GOAL:.0%})",
            console=True,
        )

        # Goal achieved → SUCCESS.
        if accuracy is not None and accuracy >= self.ACCURACY_GOAL:
            _log(
                f"[{self.agent_id}] ✅ Accuracy goal met ({accuracy:.3f} ≥ {self.ACCURACY_GOAL:.0%}).",
                console=True,
            )
            return AgentResponse(
                status="success",
                data={
                    "decision": "SUCCESS",
                    "accuracy": accuracy,
                    "stdout": stdout,
                    "insight_report": insight_report,
                    "iteration": iteration,
                },
                next_recipient="orchestration",
            )

        # Max iterations reached → REFINE with exhausted flag.
        if iteration >= max_iterations - 1:
            _log(
                f"[{self.agent_id}] ⚠️ Max iterations reached without meeting goal.",
                console=True,
            )
            feedback_package = await self._generate_feedback_package(
                stdout, stderr, tb, code, accuracy
            )
            return AgentResponse(
                status="needs_iteration",
                data={
                    "decision": "REFINE",
                    "feedback_package": feedback_package,
                    "accuracy": accuracy,
                    "max_iterations_reached": True,
                    "stdout": stdout,
                    "insight_report": insight_report,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                },
                next_recipient="orchestration",
            )

        # Need another iteration — generate structured feedback via LLM.
        feedback_package = await self._generate_feedback_package(
            stdout, stderr, tb, code, accuracy
        )
        suggested_target = feedback_package.get("suggested_target", "modeling")
        _log(
            f"[{self.agent_id}] 🔄 REFINE decision (target={suggested_target}). "
            f"Sending feedback_package to orchestration.",
            console=True,
        )
        return AgentResponse(
            status="needs_iteration",
            data={
                "decision": "REFINE",
                "feedback_package": feedback_package,
                "accuracy": accuracy,
                "stdout": stdout,
                "stderr": stderr,
                "traceback": tb,
                "code": code,
                "insight_report": insight_report,
                "iteration": iteration,
                "max_iterations": max_iterations,
            },
            next_recipient="orchestration",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_feedback_package(
        self,
        stdout: str,
        stderr: str,
        tb: str,
        code: str,
        accuracy: Optional[float],
    ) -> Dict[str, Any]:
        """Use the LLM to produce a structured feedback package.

        Analyses the execution output and accuracy to generate:

        * ``technical_feedback`` — actionable guidance for :class:`ModelingAgent`
        * ``knowledge_gap`` — search topic for :class:`ResearchAgent`
        * ``suggested_target`` — ``"modeling"`` or ``"research"``

        Falls back to a heuristic package if the LLM call fails.

        Args:
            stdout: Captured standard output from execution.
            stderr: Captured standard error from execution.
            tb: Full traceback (if any).
            code: The Python source that was executed.
            accuracy: Parsed accuracy value (``None`` if not detected).

        Returns:
            Dict with ``technical_feedback``, ``knowledge_gap``, and
            ``suggested_target`` keys.
        """
        accuracy_str = f"{accuracy:.3f}" if accuracy is not None else "N/A"
        try:
            from .llm_client import chat_json, resolve_llm_config  # type: ignore

            llm_cfg = resolve_llm_config(self.config)
            system_prompt = (
                "You are an AI evaluation assistant for a cell perturbation GNN pipeline. "
                "Analyse the execution output and accuracy, then return a JSON object with exactly "
                "these keys:\n"
                "- technical_feedback (str): Specific, actionable guidance to improve the model code "
                "(e.g., 'Add residual connections', 'Fix gradient explosion in layer 3').\n"
                "- knowledge_gap (str): A targeted literature search query if more biological context "
                "would help (e.g., 'MAPK pathway stability for GNN models'). Empty string if not needed.\n"
                "- suggested_target (str): Either 'modeling' if code changes are the priority, or "
                "'research' if more biological knowledge is needed first."
            )
            user_content = (
                f"## Execution Results\n"
                f"Accuracy: {accuracy_str} (goal: {self.ACCURACY_GOAL:.0%})\n\n"
                f"### stdout (last 500 chars)\n{stdout[-500:]}\n\n"
                f"### stderr / traceback (last 500 chars)\n{(tb or stderr)[-500:]}\n\n"
                f"### Code snippet (first 300 chars)\n{code[:300]}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: chat_json(messages, llm_config=llm_cfg, temperature=0.3),
            )
            return {
                "technical_feedback": str(result.get("technical_feedback") or ""),
                "knowledge_gap": str(result.get("knowledge_gap") or ""),
                "suggested_target": str(result.get("suggested_target") or "modeling"),
            }
        except Exception as exc:
            _log(
                f"[{self.agent_id}] LLM feedback generation failed ({exc}); "
                "using heuristic fallback.",
            )
            # Heuristic fallback: prefer code fix if there's an error, else research.
            if tb or stderr:
                return {
                    "technical_feedback": (
                        f"Fix the execution error: {(tb or stderr)[:300]}"
                    ),
                    "knowledge_gap": "",
                    "suggested_target": "modeling",
                }
            return {
                "technical_feedback": (
                    f"Current accuracy ({accuracy_str}) is below the "
                    f"{self.ACCURACY_GOAL:.0%} target. "
                    "Improve model architecture or training procedure."
                ),
                "knowledge_gap": (
                    "Retrieve more papers on GNN models for cell perturbation response prediction"
                ),
                "suggested_target": "research",
            }

    @staticmethod
    def _parse_accuracy(stdout: str) -> Optional[float]:
        """Extract a numeric accuracy / PCC metric from stdout.

        Looks for lines like ``"accuracy: 0.92"`` or ``"PCC: 0.85"``.

        Args:
            stdout: Captured standard output from the execution.

        Returns:
            Float value in ``[0, 1]`` if found, else ``None``.
        """
        import re

        patterns = [
            r"(?:accuracy|acc|pcc|pearson|r2|score)[:\s=]+([0-9]+\.?[0-9]*)",
        ]
        for pat in patterns:
            for line in stdout.splitlines():
                m = re.search(pat, line, re.IGNORECASE)
                if m:
                    try:
                        val = float(m.group(1))
                        if 0.0 <= val <= 1.0:
                            return val
                    except ValueError:
                        pass
        return None
