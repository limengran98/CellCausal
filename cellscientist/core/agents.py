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
import contextlib
import functools
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
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
# Code extraction helpers (Bug 2)
# =============================================================================

#: Compiled regex to strip <think>...</think> blocks produced by reasoning LLMs.
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)

#: Compiled regex to extract Python source from markdown code fences.
#: Matches ```python ... ``` or plain ``` ... ```.
_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```", flags=re.DOTALL
)


def _extract_python_code(raw_text: str) -> str:
    """Strip ``<think>`` tags and extract Python code from markdown fences.

    Applied by :class:`ExecutionAgent` before passing LLM output to the
    Python interpreter so that markdown-formatted responses do not cause
    :class:`SyntaxError`.

    Resolution order:

    1. Remove all ``<think>…</think>`` blocks.
    2. If a ````` ```python … ``` ````` or ````` ``` … ``` ````` fence is
       found, return the content of the **first** (longest) fence.
    3. Otherwise return the stripped text as-is (the LLM may have output
       plain Python without any fences).

    Args:
        raw_text: Raw text returned by the LLM, potentially containing
            markdown fences and/or ``<think>`` reasoning blocks.

    Returns:
        Clean Python source code ready for execution.
    """
    # 1. Strip <think>...</think> blocks.
    text = _THINK_RE.sub("", raw_text).strip()

    # 2. Extract the first fenced code block (LLM responses typically
    #    contain exactly one primary code block as the first fence).
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        return matches[0].strip()

    # 3. No fences found — return stripped text directly.
    return text


# =============================================================================
# Logging helper (consistent with other core modules)
# =============================================================================


def _log(msg: str, *, console: bool = False) -> None:
    """Print a log message with the standard CellCausal prefix.

    Args:
        msg: Message text.
        console: If True uses ``[CELL_CONSOLE]`` prefix; otherwise ``[DETAIL]``.
    """
    summary_only = str(os.environ.get("CELL_SUMMARY_ONLY", "0")).lower() in {"1", "true", "yes"}
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    elif not summary_only:
        print(f"[DETAIL] {msg}", flush=True)


class _TeeTextIO(io.TextIOBase):
    """Mirror redirected text writes to both an in-memory buffer and console stream."""

    def __init__(self, buffer: io.StringIO, stream: io.TextIOBase) -> None:
        super().__init__()
        self._buffer = buffer
        self._stream = stream

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer.write(s)
        self._stream.write(s)
        self._stream.flush()
        return len(s)

    def flush(self) -> None:
        self._buffer.flush()
        self._stream.flush()


# =============================================================================
# Path interpolation helper (Bug 4)
# =============================================================================


def _short_md5(text: str) -> str:
    """Return a short deterministic digest for log-friendly content identity checks."""
    return hashlib.md5((text or "").encode("utf-8", errors="replace")).hexdigest()[:10]


def _interpolate_path(path_template: str, config: Dict[str, Any]) -> str:
    """Replace ``${key}`` placeholders in *path_template* with config values.

    Recognises the following template variables (checked in order):

    * Top-level config keys (e.g. ``dataset_name``, ``split_name``).
    * The special key ``task_id`` if present under ``context``.

    Args:
        path_template: Path string that may contain ``${key}`` placeholders.
        config: Pipeline configuration dict supplying replacement values.

    Returns:
        The interpolated path string.  Unrecognised placeholders are left
        unchanged so that downstream code can detect them.
    """
    result = path_template
    # Substitute top-level scalar config values first.
    for key, value in config.items():
        if isinstance(value, str):
            result = result.replace(f"${{{key}}}", value)
    return result


def _project_root() -> str:
    """Return the absolute repository root for this package tree."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_project_path(path_value: str) -> str:
    """Resolve *path_value* relative to the repository root when needed."""
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(_project_root(), path_value))


def _notebook_to_script_text(nb: Any) -> str:
    """Flatten notebook code cells into a single Python script string."""
    parts: List[str] = []
    for cell in getattr(nb, "cells", []) or []:
        if getattr(cell, "cell_type", "") == "code":
            src = getattr(cell, "source", "") or ""
            if src.strip():
                parts.append(src.rstrip())
    return "\n\n".join(parts).strip()


def _ensure_task_workspace(task_id: str, category: str, iteration: int) -> str:
    """Create and return a per-task, per-iteration workspace directory."""
    safe_task_id = task_id or f"agent_task_{uuid.uuid4().hex[:8]}"
    try:
        from .workspace_manager import WorkspaceManager  # type: ignore

        wm = WorkspaceManager(safe_task_id)
        wm.ensure_dirs()
        base_dir = wm.get_task_dir() / category
        base_dir.mkdir(parents=True, exist_ok=True)
        out_dir = base_dir / f"iter_{int(iteration):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)
    except Exception:
        out_dir = os.path.join(_project_root(), "runs", safe_task_id, category, f"iter_{int(iteration):03d}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir


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

        Path templates containing ``${dataset_name}`` / ``${split_name}`` etc.
        are interpolated using :func:`_interpolate_path` so that no literal
        ``${…}`` strings appear in the resolved paths.

        Args:
            config: Full pipeline configuration (``pipeline_config.json``).

        Returns:
            A populated :class:`TaskContext`.
        """
        # Interpolate the H5 path template before resolution (Bug 4).
        paths_cfg: Dict[str, Any] = config.get("paths") or {}
        h5_raw = paths_cfg.get("data_h5_path") or paths_cfg.get("data_h5_filename") or ""
        h5_interpolated = _interpolate_path(h5_raw, config) if h5_raw else h5_raw

        # Build a shallow copy of config with the interpolated path so that
        # validate_h5_smiles_path() sees the resolved value.
        resolved_config = dict(config)
        resolved_paths = dict(paths_cfg)
        if h5_interpolated and h5_interpolated != h5_raw:
            if paths_cfg.get("data_h5_path"):
                resolved_paths["data_h5_path"] = h5_interpolated
            else:
                resolved_paths["data_h5_filename"] = h5_interpolated
            resolved_config["paths"] = resolved_paths

        # Lazily import to avoid hard dependency when not using smiles_resolver.
        try:
            from .smiles_resolver import resolve_smiles, validate_h5_smiles_path
            h5_path = validate_h5_smiles_path(resolved_config)
            smiles = resolve_smiles(resolved_config)
        except Exception as exc:
            _log(
                f"[TaskContext] ⚠️ Could not load SMILES from H5 file: {exc}. "
                "Continuing with empty SMILES list.",
                console=False,
            )
            h5_path = _interpolate_path(
                (config.get("paths") or {}).get("data_h5_path") or "", config
            )
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

    @staticmethod
    def _unwrap_message(message: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the inner payload when *message* is a bus envelope."""
        if not isinstance(message, dict):
            return {}
        data = message.get("data")
        if (
            isinstance(data, dict)
            and "status" in message
            and "next_recipient" in message
        ):
            payload = dict(data)
            for key, value in message.items():
                if key in {"status", "data", "next_recipient"}:
                    continue
                payload.setdefault(key, value)
            return payload
        return message

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
            payload = self._unwrap_message(message)
            _log(f"[{self.agent_id}] Received message (stage={payload.get('stage', '?')})")
            try:
                response = await self.process(payload)
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
        """Analyse SMILES and contextual data to produce a Causal Context Payload.

        The output is a structured payload containing:
        - ``smiles_mechanism_prior``: SMILES → Target → Pathway causal chains
          (Innovation 2: SMILES-Driven Mechanism Prior)
        - ``biokb_evidence``: BioKB semantic enrichment data
        - ``literature_context``: Web search / literature results
        - ``previous_iteration_logs``: Execution logs and metrics from the
          previous iteration (if available)

        Args:
            message: Payload that should contain ``smiles_list``, ``h5_file_path``,
                ``context_text``, and optionally ``knowledge_gap`` for targeted
                second-pass RAG retrieval.

        Returns:
            :class:`AgentResponse` directed to ``modeling`` with a structured
            Causal Context Payload.
        """
        message = self._unwrap_message(message)

        smiles_list: List[str] = message.get("smiles_list") or []
        h5_file_path: str = message.get("h5_file_path") or ""
        context_text: str = message.get("context_text") or ""
        knowledge_gap: str = message.get("knowledge_gap") or ""
        previous_iteration_logs: Dict[str, Any] = message.get("previous_iteration_logs") or {}
        falsifiable_verdict: Dict[str, Any] = message.get("falsifiable_verdict") or {}

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
                    lambda: generate_biokb_semantic_table(
                        self.config,
                        stage,
                        lambda msg: _log(f"[{self.agent_id}] {msg}"),
                    ),
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

        # ---- Innovation 2: SMILES-Driven Mechanism Prior ----
        # Build weak-supervision mechanism priors from SMILES chemical
        # structures.  This serves as a semantic anchor constraining the
        # ModelingAgent to biologically plausible modifications.
        smiles_mechanism_prior = self._build_smiles_mechanism_prior(
            smiles_list, bio_kb_data
        )
        domain_model_causal_chains = self._derive_domain_model_causal_chains(
            smiles_mechanism_prior,
            (literature_data.get("markdown_summary") or "") if isinstance(literature_data, dict) else "",
            knowledge_gap,
        )

        # Assemble the structured Causal Context Payload.
        report = {
            "smiles_list": smiles_list,
            "h5_file_path": h5_file_path,
            "smiles_mechanism_prior": smiles_mechanism_prior,
            "domain_model_causal_chains": domain_model_causal_chains,
            "biokb_evidence": bio_kb_data,
            "literature_context": literature_data,
            "context_text": context_text,
            "knowledge_gap": knowledge_gap,
            "previous_iteration_logs": previous_iteration_logs,
            "falsifiable_verdict": falsifiable_verdict,
            # Legacy keys preserved for backward compatibility.
            "bio_kb_summary": bio_kb_data,
            "literature_summary": literature_data,
        }

        _log(
            f"[{self.agent_id}] Causal Context Payload assembled: "
            f"smiles_priors={len(smiles_mechanism_prior)}, "
            f"domain_model_chains={len(domain_model_causal_chains)}, "
            f"biokb={'yes' if bio_kb_data else 'no'}, "
            f"literature={'yes' if literature_data else 'no'}, "
            f"prev_logs={'yes' if previous_iteration_logs else 'no'}",
            console=True,
        )

        return AgentResponse(
            status="success",
            data={
                "biological_insight_report": report,
                "iteration": int(message.get("iteration") or 0),
                "max_iterations": int(message.get("max_iterations") or 5),
                "history_summary": message.get("history_summary") or [],
                "best_metric_score": message.get("best_metric_score"),
                "_task_id": message.get("_task_id") or "",
            },
            next_recipient="modeling",
        )

    # ------------------------------------------------------------------
    # SMILES-Driven Mechanism Prior builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_smiles_mechanism_prior(
        smiles_list: List[str],
        bio_kb_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Construct SMILES → Target → Pathway causal chains from BioKB data.

        Supports both the legacy flat ``records``/``rows`` layouts and the
        current BioKB ``molecules`` layout. When no enrichment is available, a
        minimal SMILES-only prior is returned so downstream agents still keep a
        semantic anchor.
        """
        priors: List[Dict[str, Any]] = []
        if not smiles_list:
            return priors

        smiles_to_record: Dict[str, Dict[str, Any]] = {}

        records = bio_kb_data.get("records") or bio_kb_data.get("rows") or []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            smi = rec.get("smiles") or rec.get("SMILES") or ""
            if smi:
                smiles_to_record[str(smi)] = dict(rec)

        molecules = bio_kb_data.get("molecules") or []
        for mol in molecules:
            if not isinstance(mol, dict):
                continue
            smi = mol.get("smiles") or mol.get("canonical_smiles") or ""
            if not smi:
                continue

            targets_raw = mol.get("targets") or []
            target_names: List[str] = []
            mechanism_bits: List[str] = []
            for target in targets_raw:
                if isinstance(target, dict):
                    target_name = (
                        target.get("gene_symbol")
                        or target.get("target_name")
                        or target.get("accession")
                        or target.get("target_chembl_id")
                        or ""
                    )
                    if target_name:
                        target_names.append(str(target_name))
                    mechanism = (
                        target.get("mechanism")
                        or target.get("mechanism_of_action")
                        or target.get("action_type")
                        or ""
                    )
                    if mechanism:
                        mechanism_bits.append(str(mechanism))
                elif target:
                    target_names.append(str(target))

            pathways_raw = mol.get("pathways") or []
            pathway_names: List[str] = []
            for pathway in pathways_raw:
                if isinstance(pathway, dict):
                    pathway_name = (
                        pathway.get("display_name")
                        or pathway.get("pathway_name")
                        or pathway.get("name")
                        or pathway.get("pathway_id")
                        or ""
                    )
                    if pathway_name:
                        pathway_names.append(str(pathway_name))
                elif pathway:
                    pathway_names.append(str(pathway))

            if mechanism_bits:
                mechanism_summary = "; ".join(dict.fromkeys(mechanism_bits))
            else:
                process_names = []
                for proc in mol.get("inferred_processes") or []:
                    if isinstance(proc, dict) and proc.get("process"):
                        process_names.append(str(proc.get("process")))
                mechanism_summary = "; ".join(dict.fromkeys(process_names))

            smiles_to_record[str(smi)] = {
                "smiles": smi,
                "targets": target_names,
                "pathways": pathway_names,
                "mechanism_of_action": mechanism_summary,
            }

        for smi in smiles_list[:20]:
            rec = smiles_to_record.get(smi, {})
            targets = rec.get("targets") or rec.get("target_names") or []
            pathways = rec.get("pathways") or rec.get("pathway_names") or []
            mechanism = rec.get("mechanism_of_action") or rec.get("mechanism") or ""
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(",") if t.strip()]
            if isinstance(pathways, str):
                pathways = [p.strip() for p in pathways.split(",") if p.strip()]
            priors.append({
                "smiles": smi,
                "targets": list(dict.fromkeys([str(t) for t in targets if str(t).strip()])),
                "pathways": list(dict.fromkeys([str(p) for p in pathways if str(p).strip()])),
                "mechanism_summary": str(mechanism or ""),
            })

        return priors

    @staticmethod
    def _derive_domain_model_causal_chains(
        smiles_priors: List[Dict[str, Any]],
        literature_md: str,
        knowledge_gap: str = "",
    ) -> List[Dict[str, Any]]:
        """Derive domain->model causal chains for modeling guidance.

        Each chain links:
        domain signal/hypothesis -> mechanism rationale -> concrete modeling choice.
        """
        text = (literature_md or "").lower()
        kg = (knowledge_gap or "").strip()
        chain_templates = [
            (
                ("high-variance", "deg", "differentially expressed"),
                "High-variance DEG signals are underfit",
                "Model tends toward mean predictions and misses expression tails",
                "Use variance-aware objective (e.g., weighted MSE/PCC hybrid) and tail-focused sampling.",
            ),
            (
                ("correlation", "pcc", "pearson"),
                "Primary objective depends on correlation structure",
                "Scale bias can hurt PCC despite reasonable MSE",
                "Add differentiable PCC term and post-hoc calibration head.",
            ),
            (
                ("cross-attention", "multimodal", "fusion"),
                "Cross-modal signal alignment is critical",
                "Weak alignment between chemistry and morphology reduces transfer",
                "Use gated/residual fusion with explicit modality balance regularization.",
            ),
            (
                ("dose", "non-linear", "magnitude"),
                "Dose-response magnitude is non-linear",
                "Single linear head underfits magnitude scaling",
                "Add dose-conditioned branch or monotonic calibration subnetwork.",
            ),
        ]

        chains: List[Dict[str, Any]] = []
        for kws, signal, hypothesis, modeling in chain_templates:
            if any(kw in text for kw in kws) or any(kw in kg.lower() for kw in kws):
                chains.append({
                    "domain_signal": signal,
                    "causal_hypothesis": hypothesis,
                    "modeling_implication": modeling,
                })

        # Mechanism priors can also produce chain entries when targets/pathways exist.
        for p in (smiles_priors or [])[:10]:
            targets = p.get("targets") or []
            pathways = p.get("pathways") or []
            if targets or pathways:
                t = ", ".join([str(x) for x in targets[:3]]) if targets else "unknown-target"
                pw = ", ".join([str(x) for x in pathways[:3]]) if pathways else "unknown-pathway"
                chains.append({
                    "domain_signal": f"SMILES prior links to targets/pathways ({t} | {pw})",
                    "causal_hypothesis": "Perturbation effects should preserve pathway-consistent directionality",
                    "modeling_implication": "Add pathway-aware weighting or target-conditional gating to avoid biologically implausible fits.",
                })

        # Always return at least one chain so downstream prompt has explicit causal framing.
        if not chains:
            chains.append({
                "domain_signal": "Limited external mechanism evidence in current iteration",
                "causal_hypothesis": "Observed performance likely dominated by representation/fusion mismatch",
                "modeling_implication": "Prioritize robust baseline with conservative fusion and stable loss before adding complexity.",
            })

        # Deduplicate by triple fields and cap size.
        uniq = []
        seen = set()
        for c in chains:
            key = (c.get("domain_signal"), c.get("causal_hypothesis"), c.get("modeling_implication"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
            if len(uniq) >= 8:
                break
        return uniq


# =============================================================================
# ModelingAgent
# =============================================================================


class ModelingAgent(BaseAgent):
    """Generates and revises PyTorch/GNN model code.

    Subscribes to ``biology_insights`` and ``evaluation`` topics.
    Publishes generated code to ``code_execution``.

    Scientific rules (Hypergraph Nodes, Cell Granularity, protected/target
    sections) are loaded from ``configs/review_config.json`` and
    ``prompts/review_optimize.yaml`` at process time so that no scientific
    protocol details are hardcoded.
    """

    role = "model_architect"
    tools = ["LLM_codegen"]

    # ------------------------------------------------------------------
    # Scientific protocol helpers
    # ------------------------------------------------------------------

    def _load_scientific_rules(self) -> Dict[str, Any]:
        """Load Hypergraph Node and Cell Granularity rules from config/prompts.

        Sources (in priority order):
        1. ``self.config["review"]`` — protected_sections, target_sections,
           optimization_hierarchy from ``review_config.json``.
        2. ``prompts/review_optimize.yaml`` — Hypergraph Node decomposition.
        3. ``prompts/pipeline_prompt.yaml`` — Cell Granularity (T0-T6) and
           HDF5 mandatory datasets.

        Returns:
            Dict with keys ``protected_sections``, ``target_sections``,
            ``optimization_hierarchy``, ``cell_granularity_hint``,
            ``hypergraph_hint``, and ``mandatory_datasets``.
        """
        review_cfg: Dict[str, Any] = self.config.get("review") or {}
        protected_sections: List[str] = review_cfg.get("protected_sections") or [
            "SECTION 1", "SECTION 2", "Data Loading", "Evaluation"
        ]
        target_sections: List[str] = review_cfg.get("target_sections") or [
            "SECTION 3", "Innovation", "Modeling", "Training"
        ]
        optimization_hierarchy: List[str] = review_cfg.get("optimization_hierarchy") or [
            "1. Model Architecture (Backbone/Encoder)",
            "2. Data Fusion Mechanism (Interaction/Attention)",
            "3. Loss Function & Optimization Strategy",
        ]

        # Load YAML prompts for richer context; degrade gracefully on failure.
        hypergraph_hint = (
            "Decompose the model as a Hypergraph with three sub-nodes:\n"
            "  Node A (Architecture): The backbone/encoder.\n"
            "  Node B (Fusion): How morphology images + molecular fingerprints interact.\n"
            "  Node C (Loss): The training objective."
        )
        cell_granularity_hint = (
            "Structure the solution as Jupyter Notebook cells (T0-T6):\n"
            "  T0: Imports & setup.\n"
            "  T1: HDF5 data loading (LOCKED — do NOT modify).\n"
            "  T2: Preprocessing.\n"
            "  T3: Model definition (MUTABLE — primary target).\n"
            "  T4: Training loop (MUTABLE).\n"
            "  T5: Evaluation & metric reporting (LOCKED).\n"
            "  T6: Results serialisation.\n"
            "Mandatory HDF5 datasets: morphology_pre, morphology_post, smiles, dose, split_id."
        )

        try:
            import yaml  # type: ignore

            prompts_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "prompts"
            )
            ro_path = os.path.join(prompts_dir, "review_optimize.yaml")
            if os.path.isfile(ro_path):
                with open(ro_path, "r", encoding="utf-8") as fh:
                    ro_data = yaml.safe_load(fh) or {}
                sys_text: str = ro_data.get("system") or ""
                if sys_text:
                    # Extract the Hypergraph section using precise word-boundary matching
                    # to avoid false positives (e.g. "Note A" matching "Node A").
                    _node_re = re.compile(
                        r"\bHypergraph\b|\bNode\s+[ABC]\b", re.IGNORECASE
                    )
                    hypergraph_section = ""
                    in_block = False
                    for line in sys_text.splitlines():
                        if _node_re.search(line):
                            in_block = True
                        if in_block:
                            hypergraph_section += line + "\n"
                            if line.strip() == "" and hypergraph_section.count("\n") > 5:
                                break
                    if hypergraph_section.strip():
                        hypergraph_hint = hypergraph_section.strip()

            pp_path = os.path.join(prompts_dir, "pipeline_prompt.yaml")
            if os.path.isfile(pp_path):
                with open(pp_path, "r", encoding="utf-8") as fh:
                    pp_data = yaml.safe_load(fh) or {}
                pp_sys: str = pp_data.get("system") or ""
                if pp_sys and "T0" in pp_sys and "T1" in pp_sys:
                    # Use the raw cell-tag and mandatory dataset lines as the
                    # granularity hint.  Use anchored patterns to avoid false
                    # positives (e.g. "AT0" matching T0).
                    _cell_tag_re = re.compile(r"\bT[0-6]\b")
                    _dataset_re = re.compile(
                        r"\b(?:morphology_pre|morphology_post|split_id|dose|smiles)\b"
                    )
                    lines = pp_sys.splitlines()
                    t_lines = [
                        ln for ln in lines
                        if _cell_tag_re.search(ln) or _dataset_re.search(ln)
                    ]
                    if t_lines:
                        cell_granularity_hint = "\n".join(t_lines[:20])
        except Exception:
            pass  # Fall back to hardcoded hints above.

        return {
            "protected_sections": protected_sections,
            "target_sections": target_sections,
            "optimization_hierarchy": optimization_hierarchy,
            "cell_granularity_hint": cell_granularity_hint,
            "hypergraph_hint": hypergraph_hint,
        }

    def _use_legacy_notebook_mode(
        self,
        message: Dict[str, Any],
        insight_report: Dict[str, Any],
    ) -> bool:
        """Return True when agent-mode should preserve the legacy notebook contract."""
        prompt_branch = self.config.get("prompt_branch") or {}
        output_type = str(prompt_branch.get("output_type") or "").lower()
        previous_logs = insight_report.get("previous_iteration_logs") or {}
        return (
            output_type == "notebook"
            or bool(message.get("notebook_json"))
            or bool(previous_logs.get("notebook_json"))
            or str(message.get("artifact_type") or previous_logs.get("artifact_type") or "").startswith("notebook")
        )

    def _resolve_legacy_prompt_file(self) -> str:
        """Resolve the legacy generation prompt file used by Experiment stage."""
        prompt_branch = self.config.get("prompt_branch") or {}
        prompt_file = str(prompt_branch.get("prompt_file") or "prompts/pipeline_prompt.yaml")
        resolved = _resolve_project_path(prompt_file)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Legacy prompt file not found: {resolved}")
        return resolved

    def _build_legacy_agent_context(
        self,
        *,
        literature: str,
        mechanism_context: str,
        falsifiable_context: str,
        technical_feedback: str,
        current_metrics: Optional[Dict[str, Any]],
        sci_rules: Dict[str, Any],
        previous_logs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Build additive context while preserving the legacy prompt contract."""
        protected_str = ", ".join(sci_rules.get("protected_sections") or [])
        target_str = ", ".join(sci_rules.get("target_sections") or [])
        hierarchy_str = "\n".join(sci_rules.get("optimization_hierarchy") or [])
        hypergraph_hint = str(sci_rules.get("hypergraph_hint") or "")
        cell_granularity_hint = str(sci_rules.get("cell_granularity_hint") or "")

        system_appendix = (
            "## MULTI-AGENT VIRTUAL CELL CONTEXT (ADDITIVE, DO NOT OVERRIDE LEGACY CONTRACT)\n"
            "Follow the ORIGINAL prompt schema and response format exactly.\n"
            "Treat the information below as extra scientific constraints layered on top of the existing legacy prompt.\n"
            "Mechanism consistency is mandatory: architectural changes must be biologically justified, not random hyperparameter drift.\n"
        )

        sections: List[str] = [
            "## LEGACY CONTRACT GUARDRAILS\n"
            f"LOCKED sections: {protected_str or 'N/A'}\n"
            f"MUTABLE sections: {target_str or 'N/A'}\n"
            f"Optimization hierarchy:\n{hierarchy_str or 'N/A'}\n"
            f"Hypergraph hint:\n{hypergraph_hint or 'N/A'}\n"
            f"Cell granularity hint:\n{cell_granularity_hint or 'N/A'}",
        ]

        if mechanism_context:
            sections.append(
                "## MECHANISM-CONSTRAINED HYPOTHESIS GENERATION\n"
                "Every meaningful modeling change must map to a plausible Target → Pathway → Phenotype rationale.\n"
                "Use the SMILES-derived weak supervision below as a semantic anchor, not as a perfect oracle.\n"
                f"{mechanism_context}"
            )

        if literature:
            sections.append(
                "## RESEARCH AGENT LITERATURE CONTEXT\n"
                f"{literature[:30000]}"
            )

        if previous_logs:
            prev_metrics = previous_logs.get("raw_metrics") or previous_logs.get("metrics") or {}
            prev_stdout = str(previous_logs.get("stdout") or "")
            prev_stderr = str(previous_logs.get("stderr") or "")
            sections.append(
                "## PREVIOUS ITERATION TRACE\n"
                f"Metrics JSON: {json.dumps(prev_metrics, ensure_ascii=False)[:5000]}\n"
                f"stdout tail: {prev_stdout[-1000:]}\n"
                f"stderr tail: {prev_stderr[-1000:]}"
            )

        metric_profile = self._build_metric_failure_profile(current_metrics or {})
        if metric_profile:
            sections.append(
                "## METRIC-DRIVEN CAUSAL DESIGN BRIEF\n"
                "Convert observed metric failures to explicit architectural interventions and "
                "explain why each intervention should improve the targeted metric.\n"
                f"{metric_profile}"
            )

        if technical_feedback:
            sections.append(
                "## EVALUATION AGENT FEEDBACK\n"
                f"{technical_feedback}"
            )

        if falsifiable_context:
            sections.append(falsifiable_context.strip())

        return {
            "system_appendix": system_appendix,
            "context_dump": "\n\n".join(section for section in sections if section),
        }

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert metric values to float when possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_metric_failure_profile(self, metrics: Dict[str, Any]) -> str:
        """Build a compact metric-to-intervention brief for causal model redesign.

        This converts observed metric failures into concrete, testable modeling
        interventions so model iterations are mechanism-driven rather than
        generic prompt rewrites.
        """
        if not metrics:
            return ""

        pcc = self._safe_float(metrics.get("PCC"))
        mse = self._safe_float(metrics.get("MSE"))
        r2 = self._safe_float(metrics.get("R2"))
        deg_pcc_20 = self._safe_float(metrics.get("DEG_PCC_20"))
        deg_rmse_20 = self._safe_float(metrics.get("DEG_RMSE_20"))

        bullets: List[str] = []

        if pcc is not None and pcc < 0.35:
            bullets.append(
                "- Low PCC indicates ranking inconsistency across perturbations; "
                "prioritize representation alignment and rank-aware objectives (e.g., PCC-aware auxiliary loss)."
            )
        if r2 is not None and r2 < 0.15:
            bullets.append(
                "- Low R2 suggests weak variance capture; add residual calibration head and stronger regularization on feature fusion blocks."
            )
        if deg_pcc_20 is not None and deg_pcc_20 < 0.40:
            bullets.append(
                "- Low DEG_PCC_20 means top differential genes are not preserved; apply DEG-focused reweighting and pathway-aware supervision."
            )
        if deg_rmse_20 is not None and deg_rmse_20 > 4.0:
            bullets.append(
                "- High DEG_RMSE_20 indicates unstable tail errors; use robust loss shaping and tail-aware minibatch sampling."
            )
        if mse is not None and pcc is not None and mse < 3.5 and pcc < 0.35:
            bullets.append(
                "- MSE is acceptable while PCC remains weak: likely scale-fit without ordering-fit; introduce correlation term and post-hoc monotonic calibration."
            )

        if not bullets:
            return "- Metrics are near-consistent; keep the winning mechanism and only perform minimal, falsifiable edits."

        return "\n".join(bullets)

    async def _generate_legacy_notebook_artifact(
        self,
        *,
        insight_report: Dict[str, Any],
        literature: str,
        mechanism_context: str,
        falsifiable_context: str,
        technical_feedback: str,
        current_metrics: Dict[str, Any],
        sci_rules: Dict[str, Any],
        iteration: int,
        task_id: str,
    ) -> Dict[str, Any]:
        """Generate a notebook artifact via the legacy ``pipeline_prompt.yaml`` contract."""
        import nbformat  # type: ignore
        import yaml  # type: ignore

        from .execution_workflow import _inject_api_key, _setup_stage1_resources  # type: ignore
        from .prompt_generator import generate_notebook_content  # type: ignore

        workspace = _ensure_task_workspace(task_id, "modeling", iteration)
        debug_dir = os.path.join(workspace, "debug_prompt")
        os.makedirs(debug_dir, exist_ok=True)

        prompt_path = self._resolve_legacy_prompt_file()
        with open(prompt_path, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}

        previous_logs = insight_report.get("previous_iteration_logs") or {}
        context_payload = self._build_legacy_agent_context(
            literature=literature,
            mechanism_context=mechanism_context,
            falsifiable_context=falsifiable_context,
            technical_feedback=technical_feedback,
            current_metrics=current_metrics or insight_report.get("current_metrics") or previous_logs.get("raw_metrics") or previous_logs.get("metrics") or {},
            sci_rules=sci_rules,
            previous_logs=previous_logs,
        )

        original_system = str(spec.get("system") or "You are an expert.")
        spec["system"] = f"{original_system.rstrip()}\n\n{context_payload['system_appendix'].strip()}\n"
        spec["agent_mode_virtual_cell_context"] = context_payload["context_dump"]

        augmented_prompt_path = os.path.join(
            workspace,
            f"agent_mode_pipeline_prompt_iter_{int(iteration):03d}.yaml",
        )
        with open(augmented_prompt_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)

        _inject_api_key(self.config)
        _setup_stage1_resources(self.config, True, spec_path=augmented_prompt_path)

        nb, _user_prompt, strategy_md = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_notebook_content(self.config, augmented_prompt_path, debug_dir),
        )

        notebook_json = nbformat.writes(nb)
        script_text = _notebook_to_script_text(nb)
        metadata = {
            "generation_mode": "legacy_initial_generation",
            "selected_strategy": (strategy_md or "").splitlines()[0].strip() if strategy_md else "",
            "decision_type": "EXPLORE",
            "focus_area": "All",
            "legacy_prompt_file": prompt_path,
            "augmented_prompt_file": augmented_prompt_path,
        }
        return {
            "artifact_type": "notebook_ipynb",
            "notebook_json": notebook_json,
            "code": script_text,
            "modeling_metadata": metadata,
        }

    async def _revise_legacy_notebook_artifact(
        self,
        *,
        notebook_json: str,
        current_metrics: Dict[str, Any],
        history_summary: List[Dict[str, Any]],
        best_metric_score: Optional[float],
        technical_feedback: str,
        mechanism_context: str,
        falsifiable_context: str,
        iteration: int,
        task_id: str,
    ) -> Dict[str, Any]:
        """Revise an existing notebook artifact via the legacy review protocol."""
        import nbformat  # type: ignore

        from .notebook_autofix import apply_llm_edits  # type: ignore
        from .review_workflow import (  # type: ignore
            _inject_llm_env,
            generate_optimization_suggestion,
            identify_mutable_cells,
        )

        if not notebook_json:
            raise ValueError("Legacy notebook refinement requires notebook_json.")

        workspace = _ensure_task_workspace(task_id, "modeling", iteration)
        nb = nbformat.reads(notebook_json, as_version=4)
        mutable_indices = identify_mutable_cells(nb, self.config)
        _inject_llm_env(self.config)

        metric_failure_profile = self._build_metric_failure_profile(current_metrics or {})

        task_graph_state = (
            "# AGENT MODE CONTEXT\n"
            f"best_metric_score: {best_metric_score}\n\n"
            "# MECHANISM CONTEXT\n"
            f"{mechanism_context or 'N/A'}\n\n"
            "# METRIC-DRIVEN CAUSAL DESIGN BRIEF\n"
            f"{metric_failure_profile or 'N/A'}\n\n"
            "# FALSIFIABLE CONTEXT\n"
            f"{falsifiable_context or 'N/A'}\n\n"
            "# EVALUATION FEEDBACK\n"
            f"{technical_feedback or 'N/A'}"
        )

        suggestion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_optimization_suggestion(
                self.config,
                nb,
                mutable_indices,
                current_metrics or {},
                iteration,
                float(best_metric_score or 0.0),
                workspace,
                history_summary or [],
                task_graph_state_text=task_graph_state,
            ),
        )
        if not isinstance(suggestion, dict):
            _log(
                "[REVIEW] ⚠️ Optimization output was not JSON; keeping prior notebook for this iteration.",
                console=True,
            )
            suggestion = {
                "edits": [],
                "selected_strategy": "fallback_keep_previous",
                "decision_type": "RETRY",
                "focus_area": "All",
                "critique": "Review optimization did not return JSON; no code edits applied.",
                "semantic_gradient_analysis": "LLM output parse failed; preserved prior artifact.",
                "used_evidence_ids": [],
            }

        applied_changes = apply_llm_edits(nb, suggestion.get("edits") or [])
        updated_notebook_json = nbformat.writes(nb)
        script_text = _notebook_to_script_text(nb)
        metadata = {
            "generation_mode": "legacy_review_refinement",
            "selected_strategy": suggestion.get("selected_strategy") or "",
            "decision_type": suggestion.get("decision_type") or "",
            "focus_area": suggestion.get("focus_area") or "",
            "critique": suggestion.get("critique") or "",
            "semantic_gradient_analysis": suggestion.get("semantic_gradient_analysis") or "",
            "used_evidence_ids": suggestion.get("used_evidence_ids") or [],
            "mutable_indices": mutable_indices,
            "applied_changes": int(applied_changes),
            "suggestion": suggestion,
        }
        return {
            "artifact_type": "notebook_ipynb",
            "notebook_json": updated_notebook_json,
            "code": script_text,
            "modeling_metadata": metadata,
        }

    @monitor_agent
    @retry_llm_call()
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Generate or revise model code based on the Causal Context Payload.

        Innovation 1 (Mechanism-Constrained Virtual Cell Modeling):
        Code modifications are constrained by biological semantic context
        from the ResearchAgent's Causal Context Payload.  Blind
        hyperparameter tuning is explicitly forbidden.

        Args:
            message: May contain ``biological_insight_report`` (Causal Context
                Payload from :class:`ResearchAgent`), ``error_logs``,
                ``technical_feedback``, and ``falsifiable_verdict``.

        Returns:
            :class:`AgentResponse` directed to ``code_execution``.
        """
        message = self._unwrap_message(message)

        insight_report: Dict[str, Any] = message.get("biological_insight_report") or {}
        previous_iteration_logs: Dict[str, Any] = insight_report.get("previous_iteration_logs") or {}
        raw_error_logs: Optional[str] = message.get("error_logs")
        existing_code: Optional[str] = message.get("code") or previous_iteration_logs.get("code")
        existing_notebook_json: str = (
            message.get("notebook_json")
            or previous_iteration_logs.get("notebook_json")
            or ""
        )
        technical_feedback: str = message.get("technical_feedback") or ""
        falsifiable_verdict: Dict[str, Any] = (
            message.get("falsifiable_verdict")
            or insight_report.get("falsifiable_verdict")
            or {}
        )

        # ---- Innovation 2: Extract SMILES mechanism prior ----
        smiles_mechanism_prior: List[Dict[str, Any]] = (
            insight_report.get("smiles_mechanism_prior") or []
        )
        mechanism_context = self._format_mechanism_prior(smiles_mechanism_prior)
        domain_model_causal_chains: List[Dict[str, Any]] = (
            insight_report.get("domain_model_causal_chains") or []
        )
        domain_model_chain_context = self._format_domain_model_causal_chains(domain_model_causal_chains)
        if domain_model_chain_context:
            _log(f"[MODEL] ├─ Domain→Model causal chains: {len(domain_model_causal_chains)}", console=True)
            mechanism_context = f"{mechanism_context}\n\n### Domain→Model Causal Chains\n{domain_model_chain_context}".strip()

        # ---- Falsifiable context ----
        falsifiable_context = ""
        if falsifiable_verdict:
            verdict = falsifiable_verdict.get("verdict", "")
            delta = falsifiable_verdict.get("metric_delta", 0)
            if verdict == "REJECT":
                falsifiable_context = (
                    f"\n## ⚠️ FALSIFICATION ALERT\n"
                    f"The previous iteration DEGRADED the metric (Δ={delta:+.4f}).\n"
                    f"The code has been reverted to the last accepted state.\n"
                    f"You MUST propose a FUNDAMENTALLY DIFFERENT approach — "
                    f"do NOT repeat or incrementally tweak the rejected change.\n"
                )

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

        current_metrics: Dict[str, Any] = (
            message.get("current_metrics")
            or message.get("raw_metrics")
            or previous_iteration_logs.get("raw_metrics")
            or previous_iteration_logs.get("metrics")
            or {}
        )
        history_summary: List[Dict[str, Any]] = message.get("history_summary") or []
        best_metric_score = message.get("best_metric_score")
        task_id: str = str(message.get("_task_id") or "")

        # Load scientific protocol rules from config/YAML files.
        sci_rules = self._load_scientific_rules()
        protected_str = ", ".join(sci_rules["protected_sections"])
        target_str = ", ".join(sci_rules["target_sections"])
        hierarchy_str = "\n".join(sci_rules["optimization_hierarchy"])
        hypergraph_hint: str = sci_rules["hypergraph_hint"]
        cell_granularity_hint: str = sci_rules["cell_granularity_hint"]

        # Build a prompt for the LLM using PromptManager (falls back to hardcoded defaults).
        smiles_list: List[str] = insight_report.get("smiles_list") or []
        lit_source = (
            insight_report.get("literature_summary")
            or insight_report.get("literature_context")
            or {}
        )
        literature = lit_source.get("markdown_summary") or ""

        if self._use_legacy_notebook_mode(message, insight_report):
            try:
                if existing_notebook_json:
                    artifact_payload = await self._revise_legacy_notebook_artifact(
                        notebook_json=existing_notebook_json,
                        current_metrics=current_metrics,
                        history_summary=history_summary,
                        best_metric_score=best_metric_score,
                        technical_feedback=technical_feedback,
                        mechanism_context=mechanism_context,
                        falsifiable_context=falsifiable_context,
                        iteration=iteration,
                        task_id=task_id,
                    )
                else:
                    artifact_payload = await self._generate_legacy_notebook_artifact(
                        insight_report=insight_report,
                        literature=literature,
                        mechanism_context=mechanism_context,
                        falsifiable_context=falsifiable_context,
                        technical_feedback=technical_feedback,
                        current_metrics=current_metrics,
                        sci_rules=sci_rules,
                        iteration=iteration,
                        task_id=task_id,
                    )
                artifact_payload.update({
                    "insight_report": insight_report,
                    "iteration": iteration,
                    "max_iterations": int(message.get("max_iterations") or 5),
                    "_task_id": task_id,
                })

                metadata = artifact_payload.get("modeling_metadata") or {}
                generated_nb = str(artifact_payload.get("notebook_json") or "")
                prev_hash = _short_md5(existing_notebook_json) if existing_notebook_json else "new"
                new_hash = _short_md5(generated_nb) if generated_nb else "empty"
                changed = (prev_hash != new_hash) if existing_notebook_json else True
                _log(
                    f"[MODEL] Iteration {iteration} update | mode={metadata.get('generation_mode', 'unknown')} | strategy={metadata.get('selected_strategy', '')}",
                    console=True,
                )
                _log(
                    f"├─ Notebook hash: {prev_hash} -> {new_hash} ({'changed' if changed else 'unchanged'})",
                    console=True,
                )
                if "applied_changes" in metadata:
                    _log(f"├─ Applied edits: {int(metadata.get('applied_changes') or 0)}", console=True)
                if existing_notebook_json and not changed:
                    _log("├─ ⚠️ No notebook delta this iteration (likely fallback/no-op review output).", console=True)

                return AgentResponse(
                    status="success",
                    data=artifact_payload,
                    next_recipient="code_execution",
                )
            except Exception as exc:
                logger.exception("[%s] Legacy notebook-mode modeling failed.", self.agent_id)
                return AgentResponse(
                    status="error",
                    data={
                        "error": f"Legacy notebook agent-mode modeling failed: {exc}",
                        "traceback": traceback.format_exc(),
                        "iteration": iteration,
                        "_task_id": task_id,
                    },
                    next_recipient="orchestration",
                )

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
            # Inject scientific protocol rules so the LLM respects the
            # Hypergraph Node decomposition and Cell Granularity structure.
            _mechanism_constraint = (
                "\n\n## MECHANISM-CONSTRAINED MODELING (MANDATORY)\n"
                "Every code change MUST map to a 'Target → Pathway' causal chain.\n"
                "You MUST NOT perform random hyperparameter tuning.\n"
                "Every architectural decision MUST have a biological justification.\n"
                "Include a comment block '# MECHANISM JUSTIFICATION:' explaining the "
                "biological rationale for each major modeling decision.\n"
            )
            if mechanism_context:
                _mechanism_constraint += (
                    f"\n### SMILES Mechanism Prior (use as semantic anchor)\n"
                    f"{mechanism_context}\n"
                )
            _sci_preamble = (
                f"\n\n## SCIENTIFIC PROTOCOL CONSTRAINTS\n\n"
                f"### Hypergraph Node Architecture\n{hypergraph_hint}\n\n"
                f"### Cell Granularity (Jupyter T0-T6)\n{cell_granularity_hint}\n\n"
                f"### Optimization Hierarchy\n{hierarchy_str}\n\n"
                f"### Section Constraints\n"
                f"LOCKED (do NOT modify): {protected_str}\n"
                f"MUTABLE (optimise these): {target_str}\n"
                + _mechanism_constraint
            )
            if error_logs and existing_code:
                system_prompt = (
                    "You are a Principal AI Scientist and expert PyTorch/GNN engineer. "
                    "The code below raised errors or did not meet the metric goals. "
                    "Fix ONLY the mutable cells/sections. Return ONLY the corrected Python code."
                    + _sci_preamble
                    + falsifiable_context
                )
                user_content = (
                    f"## Error Logs / Evaluation Feedback\n{error_logs}\n\n"
                    f"## Existing Code\n```python\n{existing_code}\n```"
                )
            else:
                system_prompt = (
                    "You are a Principal AI Scientist and expert PyTorch/GNN engineer "
                    "specialising in cell perturbation modeling. "
                    "Write complete, runnable Python code for a novel model that predicts "
                    "cell painting perturbation responses. Return ONLY the Python code."
                    + _sci_preamble
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
                "artifact_type": "raw_python",
                "modeling_metadata": {
                    "generation_mode": "raw_code_fallback",
                    "decision_type": "REFINE" if error_logs else "EXPLORE",
                },
                "_task_id": message.get("_task_id") or "",
            },
            next_recipient="code_execution",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_domain_model_causal_chains(chains: List[Dict[str, Any]]) -> str:
        """Format domain background -> modeling causal chains for prompt injection."""
        if not chains:
            return ""
        lines: List[str] = []
        for idx, c in enumerate(chains[:8], start=1):
            ds = str(c.get("domain_signal") or "unknown signal")
            hyp = str(c.get("causal_hypothesis") or "")
            imp = str(c.get("modeling_implication") or "")
            lines.append(f"{idx}. Signal: {ds}")
            if hyp:
                lines.append(f"   Hypothesis: {hyp}")
            if imp:
                lines.append(f"   Modeling: {imp}")
        return "\n".join(lines)


    @staticmethod
    def _format_mechanism_prior(
        priors: List[Dict[str, Any]],
    ) -> str:
        """Format SMILES mechanism priors into a human-readable string.

        Args:
            priors: List of mechanism prior dicts from
                :meth:`ResearchAgent._build_smiles_mechanism_prior`.

        Returns:
            Formatted string for inclusion in the LLM prompt.
        """
        if not priors:
            return ""
        lines: List[str] = []
        for p in priors[:10]:
            smi = p.get("smiles") or "?"
            targets = p.get("targets") or []
            pathways = p.get("pathways") or []
            mech = p.get("mechanism_summary") or ""
            target_str = ", ".join(targets[:5]) if targets else "unknown"
            pathway_str = ", ".join(pathways[:5]) if pathways else "unknown"
            line = f"- {smi[:60]}… → Targets: [{target_str}] → Pathways: [{pathway_str}]"
            if mech:
                line += f" | Mechanism: {mech[:100]}"
            lines.append(line)
        return "\n".join(lines)


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

    def _is_notebook_artifact(self, message: Dict[str, Any]) -> bool:
        """Return True when the payload carries a legacy notebook artifact."""
        artifact_type = str(message.get("artifact_type") or "")
        return artifact_type.startswith("notebook") or bool(message.get("notebook_json"))

    async def _execute_legacy_notebook_artifact(
        self,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a notebook artifact using the legacy execution stack."""
        import nbformat  # type: ignore

        from .config_loader import load_yaml_prompts  # type: ignore
        from .execution_workflow import _inject_api_key, _setup_stage1_resources  # type: ignore
        from .prompt_orchestrator import phase_analyze, phase_execute  # type: ignore

        notebook_json = str(message.get("notebook_json") or "")
        if not notebook_json.strip():
            raise ValueError("Notebook artifact missing notebook_json payload.")

        iteration = int(message.get("iteration") or 0)
        task_id = str(message.get("_task_id") or "")
        workspace = _ensure_task_workspace(task_id, "execution", iteration)
        trial_dir = os.path.join(workspace, "trial")
        os.makedirs(trial_dir, exist_ok=True)

        if not isinstance(self.config.get("prompts"), dict):
            self.config["prompts"] = load_yaml_prompts(_resolve_project_path("prompts"))

        nb = nbformat.reads(notebook_json, as_version=4)
        nb_path = os.path.join(trial_dir, "notebook_prompt.ipynb")
        with open(nb_path, "w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)

        prompt_file = str((self.config.get("prompt_branch") or {}).get("prompt_file") or "prompts/pipeline_prompt.yaml")
        _inject_api_key(self.config)
        _setup_stage1_resources(self.config, True, spec_path=_resolve_project_path(prompt_file))

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        tee_stdout = _TeeTextIO(stdout_buffer, sys.stdout)
        tee_stderr = _TeeTextIO(stderr_buffer, sys.stderr)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            exec_res = phase_execute(self.config, trial_dir)
            try:
                phase_analyze(self.config, trial_dir)
            except Exception as exc:  # pragma: no cover - best effort only
                print(f"[DETAIL] [ExecutionAgent] phase_analyze skipped: {exc}", flush=True)

        stdout_val = stdout_buffer.getvalue()
        stderr_val = stderr_buffer.getvalue()
        raw_metrics = exec_res.get("metrics") or {}

        executed_nb_path = exec_res.get("exec_notebook") or os.path.join(trial_dir, "notebook_prompt_exec.ipynb")
        if os.path.exists(executed_nb_path):
            executed_nb = nbformat.read(executed_nb_path, as_version=4)
            final_notebook_json = nbformat.writes(executed_nb)
            code_text = _notebook_to_script_text(executed_nb)
        else:
            final_notebook_json = notebook_json
            code_text = _notebook_to_script_text(nb)

        global_errors = []
        if isinstance(raw_metrics, dict):
            global_errors = list(raw_metrics.get("global_errors") or [])
        success = bool(raw_metrics) and not global_errors and raw_metrics.get("status") != "MISSING_METRICS_JSON"

        traceback_text = ""
        if not success:
            traceback_text = stderr_val or stdout_val[-4000:]

        return {
            "status": "success" if success else "error",
            "stdout": stdout_val,
            "stderr": stderr_val,
            "traceback": traceback_text,
            "code": code_text,
            "raw_metrics": raw_metrics,
            "notebook_json": final_notebook_json,
            "artifact_type": "notebook_ipynb",
            "trial_dir": trial_dir,
        }

    @monitor_agent
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Run the code contained in *message* and capture all output.

        Args:
            message: Must contain a ``code`` key with Python source to execute.

        Returns:
            :class:`AgentResponse` directed to ``evaluation`` with
            ``stdout``, ``stderr``, and ``traceback`` fields.
        """
        message = self._unwrap_message(message)

        raw_code: str = message.get("code") or ""
        insight_report: Dict[str, Any] = message.get("insight_report") or {}
        modeling_metadata: Dict[str, Any] = message.get("modeling_metadata") or {}
        iteration: int = int(message.get("iteration") or 0)
        task_id: str = str(message.get("_task_id") or "")
        max_iterations: int = int(message.get("max_iterations") or 5)

        if self._is_notebook_artifact(message):
            try:
                exec_payload = await self._execute_legacy_notebook_artifact(message)
            except Exception as exc:
                exec_payload = {
                    "status": "error",
                    "stdout": "",
                    "stderr": str(exc),
                    "traceback": traceback.format_exc(),
                    "code": message.get("code") or "",
                    "raw_metrics": {},
                    "notebook_json": message.get("notebook_json") or "",
                    "artifact_type": str(message.get("artifact_type") or "notebook_ipynb"),
                }

            success = exec_payload.get("status") == "success"
            if not success:
                self._save_error_report(
                    task_id=task_id,
                    iteration=iteration,
                    error_type="NotebookExecutionError",
                    error_message=str(exec_payload.get("stderr") or exec_payload.get("traceback") or "Notebook execution failed."),
                    tb=str(exec_payload.get("traceback") or ""),
                    code=str(exec_payload.get("code") or ""),
                )

            return AgentResponse(
                status="success" if success else "error",
                data={
                    "stdout": exec_payload.get("stdout") or "",
                    "stderr": exec_payload.get("stderr") or "",
                    "traceback": exec_payload.get("traceback") or "",
                    "code": exec_payload.get("code") or "",
                    "raw_metrics": exec_payload.get("raw_metrics") or {},
                    "notebook_json": exec_payload.get("notebook_json") or message.get("notebook_json") or "",
                    "artifact_type": exec_payload.get("artifact_type") or "notebook_ipynb",
                    "trial_dir": exec_payload.get("trial_dir") or "",
                    "insight_report": insight_report,
                    "modeling_metadata": modeling_metadata,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "_task_id": task_id,
                },
                next_recipient="evaluation",
            )

        if not raw_code.strip():
            return AgentResponse(
                status="error",
                data={
                    "error": "No code provided for execution.",
                    "stdout": "",
                    "stderr": "",
                    "traceback": "",
                    "insight_report": insight_report,
                    "artifact_type": str(message.get("artifact_type") or "raw_python"),
                    "modeling_metadata": modeling_metadata,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "_task_id": task_id,
                },
                next_recipient="evaluation",
            )

        # Bug 2 fix: strip <think> tags and extract Python from markdown fences
        # before passing the source to the Python interpreter.
        code: str = _extract_python_code(raw_code)
        _log(
            f"[EXEC] ⚙️ ExecutionAgent is running code in Sandbox "
            f"({len(raw_code)} raw → {len(code)} extracted chars)…",
            console=True,
        )

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
                "raw_metrics": {},
                "artifact_type": str(message.get("artifact_type") or "raw_python"),
                "modeling_metadata": modeling_metadata,
                "insight_report": insight_report,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "_task_id": task_id,
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


#: Ordered list of (metric_name, regex_pattern) pairs for the full DEG suite.
#: Patterns match lines emitted by the generated execution scripts.
#: The value group supports negative numbers and scientific notation.
_METRIC_VALUE_RE = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
_DEG_METRIC_PATTERNS: List[tuple] = [
    ("PCC",        rf"(?:global_?pcc|(?<!\w)pcc(?!\w)|pearson)[:\s=]+{_METRIC_VALUE_RE}"),
    ("MSE",        rf"(?:global_?mse|(?<!\w)mse(?!\w))[:\s=]+{_METRIC_VALUE_RE}"),
    ("R2",         rf"(?:global_?r2|(?<!\w)r2(?!\w)|r_?squared)[:\s=]+{_METRIC_VALUE_RE}"),
    ("DEG_RMSE_20", rf"deg_?rmse_?20[:\s=]+{_METRIC_VALUE_RE}"),
    ("DEG_PCC_20",  rf"deg_?pcc_?20[:\s=]+{_METRIC_VALUE_RE}"),
    ("DEG_RMSE_50", rf"deg_?rmse_?50[:\s=]+{_METRIC_VALUE_RE}"),
    ("DEG_PCC_50",  rf"deg_?pcc_?50[:\s=]+{_METRIC_VALUE_RE}"),
    ("MSE_DM",     rf"mse_?dm[:\s=]+{_METRIC_VALUE_RE}"),
    ("PCC_DM",     rf"pcc_?dm[:\s=]+{_METRIC_VALUE_RE}"),
    ("R2_DM",      rf"r2_?dm[:\s=]+{_METRIC_VALUE_RE}"),
]


class EvaluationAgent(BaseAgent):
    """Evaluates execution results and decides whether to iterate.

    Subscribes to ``evaluation`` topic.
    Always publishes to ``orchestration`` — including REFINE decisions — so the
    orchestrator can update the FSM and route feedback to the correct agent.

    The target metric, pass threshold, and direction are loaded from config
    (``review.target_metric``, ``review.pass_threshold``, ``review.direction``).
    The full DEG metric suite (PCC, MSE, R2, DEG_RMSE_20/50, DEG_PCC_20/50) is
    parsed from stdout and included in every response payload.

    When the goal is not yet met, uses the LLM to generate a structured
    ``feedback_package`` containing:

    * ``technical_feedback`` — actionable guidance for :class:`ModelingAgent`
    * ``knowledge_gap`` — topic for :class:`ResearchAgent` to search
    * ``suggested_target`` — ``"modeling"`` or ``"research"``
    """

    role = "evaluator"
    tools = ["metrics_parser"]

    def __init__(self, bus: SimpleMessageBus, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(bus, config)
        self._previous_metrics: Dict[str, Optional[float]] = {}
        self._previous_primary: Optional[float] = None

    # ------------------------------------------------------------------
    # Config-driven metric helpers
    # ------------------------------------------------------------------

    def _get_metric_config(self) -> Dict[str, Any]:
        """Load target metric, threshold, and direction from config.

        Falls back to sensible defaults (PCC / 0.7 / maximize) when the
        ``review`` section is absent.

        Returns:
            Dict with keys ``target_metric`` (str), ``pass_threshold`` (float),
            ``direction`` (``"maximize"`` or ``"minimize"``).
        """
        review_cfg: Dict[str, Any] = self.config.get("review") or {}
        target_metric: str = (
            review_cfg.get("target_metric") or "PCC"
        ).upper()
        direction: str = (
            review_cfg.get("direction") or "maximize"
        ).lower()
        pass_threshold: float = float(review_cfg.get("pass_threshold") or 0.7)
        return {
            "target_metric": target_metric,
            "direction": direction,
            "pass_threshold": pass_threshold,
        }

    def _goal_met(
        self,
        metrics: Dict[str, Optional[float]],
        target_metric: str,
        pass_threshold: float,
        direction: str,
    ) -> bool:
        """Return True if the target metric satisfies the pass threshold.

        Args:
            metrics: Dict of parsed metric values (key → float or None).
            target_metric: Name of the primary metric to check (e.g. ``"PCC"``).
            pass_threshold: Numeric threshold for success.
            direction: ``"maximize"`` (higher is better) or ``"minimize"``.

        Returns:
            ``True`` if the goal condition is satisfied.
        """
        val = metrics.get(target_metric)
        if val is None:
            return False
        if direction == "minimize":
            return val <= pass_threshold
        return val >= pass_threshold

    @staticmethod
    def _extract_metric_from_model_payload(
        model_payload: Dict[str, Any],
        metric_name: str,
    ) -> Optional[float]:
        """Extract a metric from ``aggregate`` or averaged ``per_fold`` fields."""
        try:
            if isinstance(model_payload.get("aggregate"), dict):
                val = model_payload["aggregate"].get(metric_name)
                if val is not None:
                    return float(val)
            if isinstance(model_payload.get("per_fold"), dict):
                vals: List[float] = []
                for fold_payload in model_payload["per_fold"].values():
                    if not isinstance(fold_payload, dict):
                        continue
                    val = fold_payload.get(metric_name)
                    if val is None and isinstance(fold_payload.get("metrics"), dict):
                        val = fold_payload["metrics"].get(metric_name)
                    if val is None:
                        continue
                    try:
                        vals.append(float(val))
                    except Exception:
                        continue
                if vals:
                    return float(sum(vals) / len(vals))
            if metric_name in model_payload:
                return float(model_payload.get(metric_name))
        except Exception:
            return None
        return None

    @classmethod
    def _parse_metrics_payload(
        cls,
        raw_metrics: Dict[str, Any],
    ) -> Dict[str, Optional[float]]:
        """Parse the DEG metric suite from ``metrics.json``-style payloads."""
        metrics: Dict[str, Optional[float]] = {name: None for name, _ in _DEG_METRIC_PATTERNS}
        if not isinstance(raw_metrics, dict) or not raw_metrics:
            return metrics

        models = raw_metrics.get("models") if isinstance(raw_metrics.get("models"), dict) else None
        if not models:
            models = {
                key: value
                for key, value in raw_metrics.items()
                if isinstance(value, dict) and key not in {"winner", "config", "methods", "trial_dir", "status", "note", "global_errors"}
            }
        if not models:
            return metrics

        winner = raw_metrics.get("winner")
        if not winner or winner not in models:
            non_baseline = [
                key for key in models.keys()
                if "baseline" not in key.lower() and "reference" not in key.lower() and key != "config"
            ]
            winner = non_baseline[-1] if non_baseline else next(iter(models.keys()))

        model_payload = models.get(winner) or {}
        for metric_name in metrics.keys():
            metrics[metric_name] = cls._extract_metric_from_model_payload(model_payload, metric_name)
        return metrics

    # ------------------------------------------------------------------
    # Core process method
    # ------------------------------------------------------------------

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
        message = self._unwrap_message(message)

        stdout: str = message.get("stdout") or ""
        stderr: str = message.get("stderr") or ""
        tb: str = message.get("traceback") or ""
        code: str = message.get("code") or ""
        raw_metrics: Dict[str, Any] = message.get("raw_metrics") or {}
        artifact_type: str = str(message.get("artifact_type") or "raw_python")
        notebook_json: str = str(message.get("notebook_json") or "")
        modeling_metadata: Dict[str, Any] = message.get("modeling_metadata") or {}
        insight_report: Dict[str, Any] = message.get("insight_report") or {}
        iteration: int = int(message.get("iteration") or 0)
        max_iterations: int = int(message.get("max_iterations") or 5)

        # Load dynamic metric config from pipeline/review configuration.
        metric_cfg = self._get_metric_config()
        target_metric: str = metric_cfg["target_metric"]
        direction: str = metric_cfg["direction"]
        pass_threshold: float = metric_cfg["pass_threshold"]

        _log(
            f"├─ Validation target: {target_metric} | threshold={pass_threshold} | direction={direction} | iteration={iteration}/{max_iterations}",
            console=True,
        )

        # Parse the full DEG metric suite from raw metrics first, then fill gaps from stdout.
        all_metrics: Dict[str, Optional[float]] = self._parse_metrics_payload(raw_metrics)
        stdout_metrics = self._parse_all_metrics(stdout)
        for metric_name, metric_value in stdout_metrics.items():
            if all_metrics.get(metric_name) is None and metric_value is not None:
                all_metrics[metric_name] = metric_value
        primary_value: Optional[float] = all_metrics.get(target_metric)

        metric_line = ", ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=N/A"
            for k, v in all_metrics.items()
            if v is not None
        )
        _log(f"├─ Parsed metrics: {metric_line}", console=True)

        # ---- Innovation 3: Falsifiable metric comparison ----
        metric_delta: Optional[float] = None
        metric_trend: str = "unknown"
        if primary_value is not None and self._previous_primary is not None:
            metric_delta = primary_value - self._previous_primary
            metric_trend = "improved" if metric_delta >= 0 else "degraded"
            _log(
                f"[{self.agent_id}] 📈 Metric trend: {self._previous_primary:.4f} → "
                f"{primary_value:.4f} (Δ={metric_delta:+.4f}, {metric_trend})",
                console=True,
            )

        # Update previous metrics for next iteration comparison.
        self._previous_metrics = dict(all_metrics)
        self._previous_primary = primary_value

        # Convenience alias kept for orchestrator's best-accuracy tracking.
        accuracy: Optional[float] = primary_value

        # Goal achieved → SUCCESS.
        if self._goal_met(all_metrics, target_metric, pass_threshold, direction):
            _log(
                f"├─ Verdict: SUCCESS ({target_metric}={primary_value:.4f} {'≥' if direction == 'maximize' else '≤'} {pass_threshold})",
                console=True,
            )
            return AgentResponse(
                status="success",
                data={
                    "decision": "SUCCESS",
                    "accuracy": accuracy,
                    "metrics": all_metrics,
                    "raw_metrics": raw_metrics,
                    "metric_delta": metric_delta,
                    "metric_trend": metric_trend,
                    "target_metric": target_metric,
                    "stdout": stdout,
                    "stderr": stderr,
                    "traceback": tb,
                    "code": code,
                    "artifact_type": artifact_type,
                    "notebook_json": notebook_json,
                    "modeling_metadata": modeling_metadata,
                    "insight_report": insight_report,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                },
                next_recipient="orchestration",
            )

        # Max iterations reached → REFINE with exhausted flag.
        if iteration >= max_iterations - 1:
            _log(
                f"[{self.agent_id}] ⚠️ Max iterations reached without meeting "
                f"'{target_metric}' goal.",
                console=True,
            )
            feedback_package = await self._generate_feedback_package(
                stdout, stderr, tb, code, all_metrics, target_metric,
                pass_threshold, direction,
            )
            feedback_package["metric_delta"] = metric_delta
            feedback_package["metric_trend"] = metric_trend
            return AgentResponse(
                status="needs_iteration",
                data={
                    "decision": "REFINE",
                    "feedback_package": feedback_package,
                    "accuracy": accuracy,
                    "metrics": all_metrics,
                    "raw_metrics": raw_metrics,
                    "metric_delta": metric_delta,
                    "metric_trend": metric_trend,
                    "target_metric": target_metric,
                    "max_iterations_reached": True,
                    "stdout": stdout,
                    "stderr": stderr,
                    "traceback": tb,
                    "code": code,
                    "artifact_type": artifact_type,
                    "notebook_json": notebook_json,
                    "modeling_metadata": modeling_metadata,
                    "insight_report": insight_report,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                },
                next_recipient="orchestration",
            )

        # Need another iteration — generate structured feedback via LLM.
        feedback_package = await self._generate_feedback_package(
            stdout, stderr, tb, code, all_metrics, target_metric,
            pass_threshold, direction,
        )
        feedback_package["metric_delta"] = metric_delta
        feedback_package["metric_trend"] = metric_trend
        suggested_target = feedback_package.get("suggested_target", "modeling")
        _log(f"├─ Verdict: REFINE (target={suggested_target})", console=True)
        return AgentResponse(
            status="needs_iteration",
            data={
                "decision": "REFINE",
                "feedback_package": feedback_package,
                "accuracy": accuracy,
                "metrics": all_metrics,
                "raw_metrics": raw_metrics,
                "metric_delta": metric_delta,
                "metric_trend": metric_trend,
                "target_metric": target_metric,
                "stdout": stdout,
                "stderr": stderr,
                "traceback": tb,
                "code": code,
                "artifact_type": artifact_type,
                "notebook_json": notebook_json,
                "modeling_metadata": modeling_metadata,
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
        all_metrics: Dict[str, Optional[float]],
        target_metric: str,
        pass_threshold: float,
        direction: str,
    ) -> Dict[str, Any]:
        """Use the LLM to produce a structured feedback package.

        Analyses execution output and the full DEG metric suite to generate:

        * ``technical_feedback`` — actionable guidance for :class:`ModelingAgent`
        * ``knowledge_gap`` — search topic for :class:`ResearchAgent`
        * ``suggested_target`` — ``"modeling"`` or ``"research"``

        Falls back to a heuristic package if the LLM call fails.

        Args:
            stdout: Captured standard output from execution.
            stderr: Captured standard error from execution.
            tb: Full traceback (if any).
            code: The Python source that was executed.
            all_metrics: Dict of all parsed DEG metrics (key → float or None).
            target_metric: Primary metric name (e.g. ``"PCC"``).
            pass_threshold: Numeric success threshold.
            direction: ``"maximize"`` or ``"minimize"``.

        Returns:
            Dict with ``technical_feedback``, ``knowledge_gap``, and
            ``suggested_target`` keys.
        """
        primary_value = all_metrics.get(target_metric)
        primary_str = f"{primary_value:.4f}" if primary_value is not None else "N/A"
        metrics_summary = "\n".join(
            f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A"
            for k, v in all_metrics.items()
        )
        try:
            from .llm_client import chat_json, resolve_llm_config  # type: ignore

            llm_cfg = resolve_llm_config(self.config)
            system_prompt = (
                "You are a Principal AI Scientist evaluating a virtual cell perturbation model. "
                "Analyse the execution output and the full biological metric suite, then return a "
                "JSON object with exactly these keys:\n"
                "- technical_feedback (str): Specific, actionable guidance grounded in the metric "
                "analysis. Reference Hypergraph Node concepts (Architecture/Backbone, "
                "Data Fusion/Attention, Loss Function & Optimization) and cell-granularity "
                "(T0-T6 Jupyter cells). E.g., 'DEG_PCC_20 is low → the model struggles with "
                "high-variance genes; add a rank-based loss term in Node C (Loss).'.\n"
                "- knowledge_gap (str): A targeted literature search query if more biological "
                "context is needed (e.g., 'MAPK pathway for DEG high-variance gene modeling'). "
                "Empty string if not needed.\n"
                "- suggested_target (str): Either 'modeling' (code changes priority) or "
                "'research' (more biological knowledge needed first)."
            )
            user_content = (
                f"## Metric Suite Results\n"
                f"Primary metric: {target_metric} = {primary_str} "
                f"(goal: {direction} {pass_threshold})\n\n"
                f"Full DEG metric suite:\n{metrics_summary}\n\n"
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
                    f"Current {target_metric} ({primary_str}) is below the "
                    f"{direction} threshold of {pass_threshold}. "
                    "Review the Hypergraph Node architecture (Node A: Backbone, "
                    "Node B: Fusion, Node C: Loss) and ensure DEG metrics improve."
                ),
                "knowledge_gap": (
                    "Retrieve SOTA papers on GNN models for cell perturbation "
                    "response prediction, focusing on DEG high-variance gene modeling"
                ),
                "suggested_target": "research",
            }

    @staticmethod
    def _parse_all_metrics(stdout: str) -> Dict[str, Optional[float]]:
        """Parse the full DEG metric suite from stdout.

        Recognises the following metrics (case-insensitive):
        PCC, MSE, R2, DEG_RMSE_20, DEG_PCC_20, DEG_RMSE_50, DEG_PCC_50,
        MSE_DM, PCC_DM, R2_DM.

        Args:
            stdout: Captured standard output from the execution script.

        Returns:
            Dict mapping metric name → float value (``None`` if not found).
        """
        metrics: Dict[str, Optional[float]] = {
            name: None for name, _ in _DEG_METRIC_PATTERNS
        }
        lines = stdout.splitlines()
        for name, pattern in _DEG_METRIC_PATTERNS:
            # Scan lines in reverse to capture the LAST (final) reported value
            # rather than an intermediate value from an earlier epoch.
            for line in reversed(lines):
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    try:
                        metrics[name] = float(m.group(1))
                        break
                    except ValueError:
                        pass
        return metrics

    @staticmethod
    def _parse_accuracy(stdout: str) -> Optional[float]:
        """Extract the primary PCC / accuracy metric from stdout.

        Convenience wrapper around :meth:`_parse_all_metrics` that returns
        only the PCC value (used by the orchestrator's best-accuracy tracker).

        Args:
            stdout: Captured standard output from the execution.

        Returns:
            Float PCC value if found, else ``None``.
        """
        metrics = EvaluationAgent._parse_all_metrics(stdout)
        return metrics.get("PCC")
