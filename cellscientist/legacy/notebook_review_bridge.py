from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, Optional

import nbformat

from ..runtime.notebook_models import NotebookRunResult
from .notebook_bridge import (
    _derive_notebook_path_for_trial,
    _error_log_path_for_trial,
    _load_legacy_experiment_config,
    _resolve_query_notebook_path,
    _resolve_save_root,
    _trace_path_for_trial,
)

_LEGACY_REVIEW_ENTRY = "cellscientist.core.prompt_orchestrator.phase_analyze"
_LEGACY_REVIEW_HELPER = "cellscientist.core.review_workflow.identify_mutable_cells"


def _find_latest_reviewable_trial(cfg: Dict[str, Any]) -> Optional[str]:
    save_root = _resolve_save_root(cfg)
    if not os.path.isdir(save_root):
        return None

    metric_candidates = []
    notebook_candidates = []
    for entry in os.listdir(save_root):
        trial_dir = os.path.join(save_root, entry)
        if not os.path.isdir(trial_dir):
            continue
        has_metrics = os.path.exists(os.path.join(trial_dir, "metrics.json"))
        has_notebook = bool(
            _derive_notebook_path_for_trial(trial_dir, executed=True)
            or _derive_notebook_path_for_trial(trial_dir)
        )
        if has_metrics:
            metric_candidates.append(trial_dir)
        elif has_notebook:
            notebook_candidates.append(trial_dir)

    candidates = metric_candidates or notebook_candidates
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _analyze_notebook_structure(notebook_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from ..core.executor_engine import collect_cell_errors
    from ..core.review_workflow import identify_mutable_cells

    nb = nbformat.read(notebook_path, as_version=4)
    total_cells = len(nb.cells)
    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in nb.cells if cell.get("cell_type") == "markdown"]
    executed_code_cells = sum(
        1 for cell in code_cells if cell.get("execution_count") is not None or cell.get("outputs")
    )
    mutable_indices = identify_mutable_cells(nb, cfg)
    errors = collect_cell_errors(nb)

    return {
        "total_cells": total_cells,
        "code_cells": len(code_cells),
        "markdown_cells": len(markdown_cells),
        "executed_code_cells": executed_code_cells,
        "mutable_cell_count": len(mutable_indices),
        "mutable_indices": mutable_indices,
        "error_cell_count": len(errors),
        "error_cell_indices": [int(error.get("cell_index", -1)) for error in errors if "cell_index" in error],
    }


def _write_review_summary(
    *,
    summary_dir: str,
    target_notebook_path: Optional[str],
    trial_dir: Optional[str],
    status: str,
    structure: Dict[str, Any],
    run_result: Optional[NotebookRunResult],
    source_artifact_metadata: Optional[Dict[str, Any]],
    metrics_path: Optional[str],
    error_log_path: Optional[str],
    run_log_path: Optional[str],
    legacy_report_path: Optional[str],
) -> str:
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "notebook_review_summary.md")

    lines = [
        "# Notebook Review Summary",
        "",
        f"- Status: `{status}`",
        f"- Target notebook: `{target_notebook_path or 'N/A'}`",
        f"- Trial dir: `{trial_dir or 'N/A'}`",
        f"- Metrics path: `{metrics_path or 'N/A'}`",
        f"- Error log path: `{error_log_path or 'N/A'}`",
        f"- Run log path: `{run_log_path or 'N/A'}`",
        f"- Legacy report path: `{legacy_report_path or 'N/A'}`",
        "",
        "## Notebook Structure",
        "",
        f"- Total cells: `{structure.get('total_cells', 0)}`",
        f"- Code cells: `{structure.get('code_cells', 0)}`",
        f"- Markdown cells: `{structure.get('markdown_cells', 0)}`",
        f"- Executed code cells: `{structure.get('executed_code_cells', 0)}`",
        f"- Mutable cell count: `{structure.get('mutable_cell_count', 0)}`",
        f"- Error cell count: `{structure.get('error_cell_count', 0)}`",
        "",
        "## Context",
        "",
        f"- Used recent run result: `{bool(run_result is not None)}`",
        f"- Recent run status: `{run_result.status if run_result is not None else 'N/A'}`",
        "",
        "## Source Artifact Metadata",
        "",
        "```json",
        json.dumps(source_artifact_metadata or {}, ensure_ascii=False, indent=2),
        "```",
    ]

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return summary_path


def bridge_review_notebook(
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
        "legacy_helpers": [_LEGACY_REVIEW_HELPER],
    }

    try:
        from ..core.prompt_orchestrator import _get_latest_trial, phase_analyze

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
        details["query_notebook_path"] = explicit_notebook_path
        details["preferred_notebook_path"] = preferred_notebook_path
        details["preferred_trial_dir"] = preferred_trial_dir
        details["latest_trial_dir"] = latest_trial_dir

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

        metrics_path = os.path.join(target_trial_dir, "metrics.json") if target_trial_dir else None
        run_log_path = (
            preferred_run_result.run_log_path
            if preferred_run_result and preferred_run_result.run_log_path
            else _trace_path_for_trial(target_trial_dir)
        )
        error_log_path = (
            preferred_run_result.error_log_path
            if preferred_run_result and preferred_run_result.error_log_path
            else _error_log_path_for_trial(target_trial_dir)
        )

        details["target_trial_dir"] = target_trial_dir
        details["target_metrics_path"] = metrics_path
        details["target_error_log_path"] = error_log_path
        details["target_run_log_path"] = run_log_path

        if not target_notebook_path or not os.path.exists(target_notebook_path):
            return {
                "action": "review",
                "status": "review_missing_context",
                "message": (
                    "Notebook review could not find a recent notebook artifact or legacy trial. "
                    "In future multi-turn sessions, this skill will prefer the latest notebook artifact "
                    "and run result from session state."
                ),
                "query": query,
                "target_notebook_path": target_notebook_path,
                "review_report_path": None,
                "legacy_entry": _LEGACY_REVIEW_ENTRY,
                "details": details,
            }

        structure = _analyze_notebook_structure(target_notebook_path, cfg)
        details["structure"] = structure

        summary_dir = target_trial_dir or os.path.dirname(target_notebook_path)
        summary_path = _write_review_summary(
            summary_dir=summary_dir,
            target_notebook_path=target_notebook_path,
            trial_dir=target_trial_dir,
            status="review_summary_only",
            structure=structure,
            run_result=preferred_run_result,
            source_artifact_metadata=source_artifact_metadata,
            metrics_path=metrics_path if metrics_path and os.path.exists(metrics_path) else None,
            error_log_path=error_log_path,
            run_log_path=run_log_path,
            legacy_report_path=None,
        )
        details["bridge_summary_path"] = summary_path

        legacy_report_path = None
        if target_trial_dir and metrics_path and os.path.exists(metrics_path):
            phase_report = phase_analyze(cfg, target_trial_dir) or {}
            legacy_report_path = phase_report.get("report_path")
            details["legacy_report_path"] = legacy_report_path
            summary_path = _write_review_summary(
                summary_dir=summary_dir,
                target_notebook_path=target_notebook_path,
                trial_dir=target_trial_dir,
                status="reviewed_via_legacy" if legacy_report_path else "review_summary_only",
                structure=structure,
                run_result=preferred_run_result,
                source_artifact_metadata=source_artifact_metadata,
                metrics_path=metrics_path,
                error_log_path=error_log_path,
                run_log_path=run_log_path,
                legacy_report_path=legacy_report_path,
            )
            details["bridge_summary_path"] = summary_path

        return {
            "action": "review",
            "status": "reviewed_via_legacy" if legacy_report_path else "review_summary_only",
            "message": (
                "Notebook review completed using legacy analysis/report hooks."
                if legacy_report_path
                else "Notebook review produced a structured summary, but no full legacy metrics report was available."
            ),
            "query": query,
            "target_notebook_path": target_notebook_path,
            "review_report_path": legacy_report_path or summary_path,
            "legacy_entry": _LEGACY_REVIEW_ENTRY,
            "details": details,
        }
    except Exception as exc:
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc(limit=8)
        return {
            "action": "review",
            "status": "review_legacy_failed",
            "message": "Notebook review bridge was invoked, but the legacy review path could not complete.",
            "query": query,
            "target_notebook_path": preferred_notebook_path,
            "review_report_path": None,
            "legacy_entry": _LEGACY_REVIEW_ENTRY,
            "details": details,
        }
