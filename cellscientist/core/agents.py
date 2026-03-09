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
from typing import Any, Dict, List, Optional

from .message_bus import SimpleMessageBus

logger = logging.getLogger(__name__)


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

    @retry_llm_call()
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Analyse SMILES and contextual data to produce a biology report.

        Args:
            message: Payload that should contain ``smiles_list``, ``h5_file_path``,
                and optionally ``context_text``.

        Returns:
            :class:`AgentResponse` directed to ``biology_insights``.
        """
        smiles_list: List[str] = message.get("smiles_list") or []
        h5_file_path: str = message.get("h5_file_path") or ""
        context_text: str = message.get("context_text") or ""
        stage: str = message.get("stage") or "design"

        _log(
            f"[{self.agent_id}] Retrieving biology insights for "
            f"{len(smiles_list)} SMILES…",
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

        # Web / literature retrieval via external_knowledge_mirothink
        try:
            from .external_knowledge_mirothink import (  # type: ignore
                retrieve_external_knowledge,
                knowledge_pack_to_markdown,
            )

            pack = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retrieve_external_knowledge(
                    self.config,
                    context_text or " ".join(smiles_list[:5]),
                    stage,
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
        }

        return AgentResponse(
            status="success",
            data={"biological_insight_report": report},
            next_recipient="biology_insights",
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
        error_logs: Optional[str] = message.get("error_logs")
        existing_code: Optional[str] = message.get("code")

        _log(
            f"[{self.agent_id}] {'Revising' if error_logs else 'Generating'} model code…",
            console=True,
        )

        # Build a prompt for the LLM.
        smiles_list: List[str] = insight_report.get("smiles_list") or []
        literature = (insight_report.get("literature_summary") or {}).get("markdown_summary") or ""

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

        _log(f"[{self.agent_id}] Executing code ({len(code)} chars)…", console=True)

        timeout: int = int(
            (self.config.get("exec") or {}).get("timeout_seconds") or 300
        )

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

        return AgentResponse(
            status=status_val,
            data={
                "stdout": stdout_val,
                "stderr": stderr_val,
                "traceback": tb_val,
                "code": code,
                "insight_report": insight_report,
            },
            next_recipient="evaluation",
        )


# =============================================================================
# EvaluationAgent
# =============================================================================


class EvaluationAgent(BaseAgent):
    """Evaluates execution results and decides whether to iterate.

    Subscribes to ``evaluation`` topic.
    Publishes final status to ``orchestration``; on failure, sends feedback
    to ``biology_insights`` or ``modeling`` for further iteration.
    """

    role = "evaluator"
    tools = ["metrics_parser"]

    # Target accuracy threshold (90 %).
    ACCURACY_GOAL: float = 0.90

    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Assess execution output and determine next action.

        Args:
            message: Payload from :class:`ExecutionAgent` containing
                ``stdout``, ``stderr``, ``traceback``, ``code``, and
                ``insight_report``.

        Returns:
            :class:`AgentResponse` directed to ``orchestration`` (if goal met
            or max iterations reached), or to ``modeling`` / ``biology_insights``
            for another iteration.
        """
        stdout: str = message.get("stdout") or ""
        stderr: str = message.get("stderr") or ""
        tb: str = message.get("traceback") or ""
        code: str = message.get("code") or ""
        insight_report: Dict[str, Any] = message.get("insight_report") or {}
        iteration: int = int(message.get("iteration") or 0)
        max_iterations: int = int(message.get("max_iterations") or 5)

        _log(
            f"[{self.agent_id}] Evaluating results (iteration {iteration}/{max_iterations})…",
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

        # Goal achieved?
        if accuracy is not None and accuracy >= self.ACCURACY_GOAL:
            _log(
                f"[{self.agent_id}] ✅ Accuracy goal met ({accuracy:.3f} ≥ {self.ACCURACY_GOAL:.0%}).",
                console=True,
            )
            return AgentResponse(
                status="success",
                data={
                    "accuracy": accuracy,
                    "stdout": stdout,
                    "insight_report": insight_report,
                },
                next_recipient="orchestration",
            )

        # Max iterations reached?
        if iteration >= max_iterations - 1:
            _log(
                f"[{self.agent_id}] ⚠️ Max iterations reached without meeting goal.",
                console=True,
            )
            return AgentResponse(
                status="needs_iteration",
                data={
                    "accuracy": accuracy,
                    "max_iterations_reached": True,
                    "stdout": stdout,
                    "insight_report": insight_report,
                },
                next_recipient="orchestration",
            )

        # There was an execution error → send to ModelingAgent for code fix.
        if stderr or tb:
            error_logs = tb or stderr
            _log(
                f"[{self.agent_id}] 🔄 Sending error logs to ModelingAgent for revision.",
                console=True,
            )
            return AgentResponse(
                status="needs_iteration",
                data={
                    "error_logs": error_logs,
                    "code": code,
                    "biological_insight_report": insight_report,
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations,
                },
                next_recipient="modeling",
            )

        # No error, but accuracy not met → send back to ResearchAgent.
        _log(
            f"[{self.agent_id}] 🔄 Accuracy not met; requesting more biology insights.",
            console=True,
        )
        accuracy_str = f"{accuracy:.3f}" if accuracy is not None else "N/A"
        return AgentResponse(
            status="needs_iteration",
            data={
                "accuracy": accuracy,
                "feedback": (
                    f"Current accuracy ({accuracy_str}) is below the "
                    f"{self.ACCURACY_GOAL:.0%} target. "
                    "Please provide deeper biological insights to guide model improvement."
                ),
                "insight_report": insight_report,
                "iteration": iteration + 1,
                "max_iterations": max_iterations,
            },
            next_recipient="biology_insights",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
