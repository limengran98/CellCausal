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
import os
import re
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
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    else:
        print(f"[DETAIL] {msg}", flush=True)


# =============================================================================
# Path interpolation helper (Bug 4)
# =============================================================================


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

        # ---- Innovation 2: SMILES-Driven Mechanism Prior ----
        # Build weak-supervision mechanism priors from SMILES chemical
        # structures.  This serves as a semantic anchor constraining the
        # ModelingAgent to biologically plausible modifications.
        smiles_mechanism_prior = self._build_smiles_mechanism_prior(
            smiles_list, bio_kb_data
        )

        # Assemble the structured Causal Context Payload.
        report = {
            "smiles_list": smiles_list,
            "h5_file_path": h5_file_path,
            "smiles_mechanism_prior": smiles_mechanism_prior,
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
            f"biokb={'yes' if bio_kb_data else 'no'}, "
            f"literature={'yes' if literature_data else 'no'}, "
            f"prev_logs={'yes' if previous_iteration_logs else 'no'}",
            console=True,
        )

        return AgentResponse(
            status="success",
            data={"biological_insight_report": report},
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

        When BioKB enrichment is available, this extracts the compound–target–
        pathway relationships.  When it is absent, returns a minimal list of
        SMILES-only entries that the ModelingAgent can use as a structural
        anchor (Innovation 2).

        Args:
            smiles_list: List of SMILES strings.
            bio_kb_data: Dict returned by :func:`generate_biokb_semantic_table`
                (may be empty).

        Returns:
            List of dicts with keys ``smiles``, ``targets``, ``pathways``,
            and ``mechanism_summary``.
        """
        priors: List[Dict[str, Any]] = []

        # Extract compound-level records from BioKB table if available.
        records = bio_kb_data.get("records") or bio_kb_data.get("rows") or []

        smiles_to_record: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            smi = rec.get("smiles") or rec.get("SMILES") or ""
            if smi:
                smiles_to_record[smi] = rec

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
                "targets": targets,
                "pathways": pathways,
                "mechanism_summary": mechanism,
            })

        return priors


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
        insight_report: Dict[str, Any] = message.get("biological_insight_report") or {}
        raw_error_logs: Optional[str] = message.get("error_logs")
        existing_code: Optional[str] = message.get("code")
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
                "_task_id": message.get("_task_id") or "",
            },
            next_recipient="code_execution",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    @monitor_agent
    async def process(self, message: Dict[str, Any]) -> AgentResponse:
        """Run the code contained in *message* and capture all output.

        Args:
            message: Must contain a ``code`` key with Python source to execute.

        Returns:
            :class:`AgentResponse` directed to ``evaluation`` with
            ``stdout``, ``stderr``, and ``traceback`` fields.
        """
        raw_code: str = message.get("code") or ""
        insight_report: Dict[str, Any] = message.get("insight_report") or {}

        if not raw_code.strip():
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
        stdout: str = message.get("stdout") or ""
        stderr: str = message.get("stderr") or ""
        tb: str = message.get("traceback") or ""
        code: str = message.get("code") or ""
        insight_report: Dict[str, Any] = message.get("insight_report") or {}
        iteration: int = int(message.get("iteration") or 0)
        max_iterations: int = int(message.get("max_iterations") or 5)

        # Load dynamic metric config from pipeline/review configuration.
        metric_cfg = self._get_metric_config()
        target_metric: str = metric_cfg["target_metric"]
        direction: str = metric_cfg["direction"]
        pass_threshold: float = metric_cfg["pass_threshold"]

        _log(
            f"[EVAL] 📊 EvaluationAgent evaluating '{target_metric}' "
            f"(threshold={pass_threshold}, direction={direction}) "
            f"— iteration {iteration}/{max_iterations}…",
            console=True,
        )

        # Parse the full DEG metric suite from stdout.
        all_metrics: Dict[str, Optional[float]] = self._parse_all_metrics(stdout)
        primary_value: Optional[float] = all_metrics.get(target_metric)

        _log(
            f"[{self.agent_id}] Parsed metrics: "
            + ", ".join(
                f"{k}={v:.4f}" if v is not None else f"{k}=N/A"
                for k, v in all_metrics.items()
                if v is not None
            ),
            console=True,
        )

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
                f"[{self.agent_id}] ✅ {target_metric} goal met "
                f"({primary_value:.4f} {'≥' if direction == 'maximize' else '≤'} "
                f"{pass_threshold}).",
                console=True,
            )
            return AgentResponse(
                status="success",
                data={
                    "decision": "SUCCESS",
                    "accuracy": accuracy,
                    "metrics": all_metrics,
                    "metric_delta": metric_delta,
                    "metric_trend": metric_trend,
                    "target_metric": target_metric,
                    "stdout": stdout,
                    "insight_report": insight_report,
                    "iteration": iteration,
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
                    "metric_delta": metric_delta,
                    "metric_trend": metric_trend,
                    "target_metric": target_metric,
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
            stdout, stderr, tb, code, all_metrics, target_metric,
            pass_threshold, direction,
        )
        feedback_package["metric_delta"] = metric_delta
        feedback_package["metric_trend"] = metric_trend
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
                "metrics": all_metrics,
                "metric_delta": metric_delta,
                "metric_trend": metric_trend,
                "target_metric": target_metric,
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
