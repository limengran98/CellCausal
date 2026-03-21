from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, Optional

from ..core.external_knowledge_mirothink import (
    knowledge_pack_to_markdown,
    retrieve_external_knowledge,
)
from ..runtime.notebook_models import NotebookRunResult
from .notebook_bridge import (
    _build_bridge_run_name,
    _derive_notebook_path_for_trial,
    _error_log_path_for_trial,
    _load_legacy_experiment_config,
    _resolve_query_notebook_path,
    _resolve_save_root,
    _trace_path_for_trial,
)
from .notebook_review_bridge import _find_latest_reviewable_trial

_LEGACY_RETRIEVAL_ENTRY = "cellscientist.core.external_knowledge_mirothink.retrieve_external_knowledge"


def _read_text_excerpt(path: Optional[str], *, max_chars: int = 1600) -> str:
    if not path or not os.path.exists(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read(max_chars).strip()
    except Exception:
        return ""


def _read_metrics_excerpt(path: Optional[str], *, max_keys: int = 10) -> str:
    if not path or not os.path.exists(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return ""

        excerpt = {key: payload[key] for key in list(payload)[:max_keys]}
        return json.dumps(excerpt, ensure_ascii=False, indent=2)
    except Exception:
        return ""


def _build_context_text(
    *,
    query: str,
    target_notebook_path: Optional[str],
    target_trial_dir: Optional[str],
    preferred_run_result: Optional[NotebookRunResult],
    source_artifact_metadata: Optional[Dict[str, Any]],
    metrics_excerpt: str,
    error_excerpt: str,
) -> str:
    lines = [
        "Notebook workflow retrieval refresh context",
        f"User query: {query}",
        f"Target notebook: {target_notebook_path or 'N/A'}",
        f"Target trial dir: {target_trial_dir or 'N/A'}",
    ]

    if preferred_run_result is not None:
        lines.extend(
            [
                f"Recent run status: {preferred_run_result.status}",
                f"Recent run log path: {preferred_run_result.run_log_path or 'N/A'}",
                f"Recent error log path: {preferred_run_result.error_log_path or 'N/A'}",
            ]
        )

    if source_artifact_metadata:
        lines.extend(
            [
                "Source artifact metadata:",
                json.dumps(source_artifact_metadata, ensure_ascii=False, indent=2),
            ]
        )

    if metrics_excerpt:
        lines.extend(
            [
                "Metrics excerpt:",
                metrics_excerpt,
            ]
        )

    if error_excerpt:
        lines.extend(
            [
                "Error excerpt:",
                error_excerpt,
            ]
        )

    return "\n".join(lines)


def bridge_refresh_notebook_retrieval(
    query: str,
    *,
    preferred_notebook_path: Optional[str] = None,
    preferred_trial_dir: Optional[str] = None,
    preferred_run_result: Optional[NotebookRunResult] = None,
    source_artifact_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "query": query,
        "source_artifact_metadata": dict(source_artifact_metadata or {}),
        "used_recent_run_result": bool(preferred_run_result is not None),
    }

    try:
        from ..core.prompt_orchestrator import _get_latest_trial

        cfg, config_paths = _load_legacy_experiment_config()
        details.update(config_paths)

        explicit_notebook_path = _resolve_query_notebook_path(query)
        latest_trial_dir = _get_latest_trial(cfg)
        if latest_trial_dir and not (
            os.path.exists(os.path.join(latest_trial_dir, "metrics.json"))
            or _derive_notebook_path_for_trial(latest_trial_dir, executed=True)
            or _derive_notebook_path_for_trial(latest_trial_dir)
        ):
            latest_trial_dir = _find_latest_reviewable_trial(cfg)

        target_notebook_path = explicit_notebook_path
        target_trial_dir = preferred_trial_dir
        if preferred_run_result is not None:
            if preferred_run_result.notebook_path and not target_notebook_path:
                target_notebook_path = preferred_run_result.notebook_path
            if preferred_run_result.trial_dir and not target_trial_dir:
                target_trial_dir = preferred_run_result.trial_dir
            details["recent_run_result"] = {
                "notebook_path": preferred_run_result.notebook_path,
                "trial_dir": preferred_run_result.trial_dir,
                "status": preferred_run_result.status,
                "error_log_path": preferred_run_result.error_log_path,
                "run_log_path": preferred_run_result.run_log_path,
                "metadata": dict(preferred_run_result.metadata or {}),
            }

        if not target_notebook_path and preferred_notebook_path and os.path.exists(preferred_notebook_path):
            target_notebook_path = preferred_notebook_path
        if not target_trial_dir and latest_trial_dir:
            target_trial_dir = latest_trial_dir
        if not target_notebook_path and target_trial_dir:
            target_notebook_path = _derive_notebook_path_for_trial(
                target_trial_dir,
                executed=True,
            ) or _derive_notebook_path_for_trial(target_trial_dir)
        if target_notebook_path and not target_trial_dir:
            target_trial_dir = os.path.dirname(target_notebook_path)

        used_query_only_context = False
        if not target_trial_dir:
            target_trial_dir = os.path.join(_resolve_save_root(cfg), _build_bridge_run_name("bridge_refresh"))
            used_query_only_context = True
        os.makedirs(target_trial_dir, exist_ok=True)

        metrics_path = os.path.join(target_trial_dir, "metrics.json") if target_trial_dir else None
        error_log_path = (
            preferred_run_result.error_log_path
            if preferred_run_result and preferred_run_result.error_log_path
            else _error_log_path_for_trial(target_trial_dir)
        )
        run_log_path = (
            preferred_run_result.run_log_path
            if preferred_run_result and preferred_run_result.run_log_path
            else _trace_path_for_trial(target_trial_dir)
        )

        metrics_excerpt = _read_metrics_excerpt(metrics_path if metrics_path and os.path.exists(metrics_path) else None)
        error_excerpt = _read_text_excerpt(error_log_path)
        context_text = _build_context_text(
            query=query,
            target_notebook_path=target_notebook_path,
            target_trial_dir=target_trial_dir,
            preferred_run_result=preferred_run_result,
            source_artifact_metadata=source_artifact_metadata,
            metrics_excerpt=metrics_excerpt,
            error_excerpt=error_excerpt,
        )

        pack = retrieve_external_knowledge(
            cfg=cfg,
            context_text=context_text,
            stage="review",
            query_hint=query,
            workspace_dir=target_trial_dir,
            tag="notebook_refresh",
        )
        evidence_summary = knowledge_pack_to_markdown(pack, max_chars=2400)
        evidence_count = len(pack.items)
        evidence_ids = [item.eid for item in pack.items if item.eid]
        evidence_sources = sorted({item.source for item in pack.items if item.source})

        workspace_knowledge_dir = os.path.join(target_trial_dir, "external_knowledge")
        latest_json_path = os.path.join(workspace_knowledge_dir, "external_knowledge_review.json")
        latest_md_path = os.path.join(workspace_knowledge_dir, "external_knowledge_review.md")

        details.update(
            {
                "target_notebook_path": target_notebook_path,
                "target_trial_dir": target_trial_dir,
                "used_query_only_context": used_query_only_context,
                "metrics_path": metrics_path,
                "error_log_path": error_log_path,
                "run_log_path": run_log_path,
                "evidence_ids": evidence_ids,
                "evidence_sources": evidence_sources,
                "external_knowledge_json_path": latest_json_path if os.path.exists(latest_json_path) else None,
                "external_knowledge_md_path": latest_md_path if os.path.exists(latest_md_path) else None,
            }
        )

        return {
            "action": "retrieval_refresh",
            "status": "retrieval_refreshed" if evidence_count else "retrieval_refresh_empty",
            "message": (
                "Notebook workflow refreshed biological evidence before downstream review/execution."
                if evidence_count
                else "Notebook workflow attempted a retrieval refresh, but no evidence items were returned."
            ),
            "query": query,
            "target_notebook_path": target_notebook_path,
            "evidence_summary": evidence_summary,
            "evidence_count": evidence_count,
            "legacy_entry": _LEGACY_RETRIEVAL_ENTRY,
            "details": details,
        }
    except Exception as exc:
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc(limit=8)
        return {
            "action": "retrieval_refresh",
            "status": "retrieval_refresh_failed",
            "message": "Notebook retrieval refresh could not complete, but downstream review/execution can still continue.",
            "query": query,
            "target_notebook_path": preferred_notebook_path,
            "evidence_summary": "",
            "evidence_count": 0,
            "legacy_entry": _LEGACY_RETRIEVAL_ENTRY,
            "details": details,
        }
