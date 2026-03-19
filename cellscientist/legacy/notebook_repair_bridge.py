from __future__ import annotations

import os
import re
import traceback
from typing import Any, Dict, List, Optional

import nbformat

from ..runtime.notebook_models import NotebookRunResult
from .notebook_bridge import (
    _derive_notebook_path_for_trial,
    _error_log_path_for_trial,
    _load_legacy_experiment_config,
    _resolve_query_notebook_path,
    _resolve_save_root,
)

_LEGACY_AUTOFIX_ENTRY = "cellscientist.core.notebook_autofix.attempt_fix_notebook"
_ERROR_LOG_RE = re.compile(r'([^\s"\']+\.(?:txt|log))')


def _find_latest_repairable_trial(cfg: Dict[str, Any]) -> Optional[str]:
    save_root = _resolve_save_root(cfg)
    if not os.path.isdir(save_root):
        return None

    error_candidates = []
    notebook_candidates = []
    for entry in os.listdir(save_root):
        trial_dir = os.path.join(save_root, entry)
        if not os.path.isdir(trial_dir):
            continue
        has_notebook = bool(
            _derive_notebook_path_for_trial(trial_dir, executed=True)
            or _derive_notebook_path_for_trial(trial_dir)
        )
        has_error_log = bool(_error_log_path_for_trial(trial_dir))
        if has_error_log:
            error_candidates.append(trial_dir)
        elif has_notebook:
            notebook_candidates.append(trial_dir)

    candidates = error_candidates or notebook_candidates
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _resolve_query_error_log_path(query: str) -> Optional[str]:
    root = os.getcwd()
    for raw_path in _ERROR_LOG_RE.findall(query):
        candidates = [raw_path]
        if not os.path.isabs(raw_path):
            candidates.append(os.path.join(root, raw_path))
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return None


def _parse_dumped_error_log(error_log_path: str) -> List[Dict[str, Any]]:
    if not error_log_path or not os.path.exists(error_log_path):
        return []
    if "error_log_round_" not in os.path.basename(error_log_path):
        return []

    with open(error_log_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    matches = re.finditer(
        r"CELL INDEX:\s*(?P<cell_index>\d+)\n"
        r"ERROR TYPE:\s*(?P<ename>[^\n]*)\n"
        r"MESSAGE\s*:\s*(?P<evalue>[^\n]*)\n"
        r"-+\s*SOURCE CODE\s*-+\n(?P<source>.*?)\n"
        r"-+\s*TRACEBACK\s*-+\n(?P<traceback>.*?)(?=\n-+\nCELL INDEX:|\Z)",
        text,
        re.DOTALL,
    )
    errors: List[Dict[str, Any]] = []
    for match in matches:
        errors.append(
            {
                "cell_index": int(match.group("cell_index")),
                "ename": match.group("ename").strip(),
                "evalue": match.group("evalue").strip(),
                "source": match.group("source").strip(),
                "traceback": match.group("traceback").strip(),
            }
        )
    return errors


def _patched_notebook_path(notebook_path: str) -> str:
    if notebook_path.endswith(".ipynb"):
        return notebook_path[:-6] + "_autofix.ipynb"
    return notebook_path + "_autofix.ipynb"


def bridge_autofix_notebook(
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
        from ..core.executor_engine import collect_cell_errors
        from ..core.notebook_autofix import attempt_fix_notebook
        from ..core.review_workflow import identify_mutable_cells

        cfg, config_paths = _load_legacy_experiment_config()
        details.update(config_paths)

        explicit_notebook_path = _resolve_query_notebook_path(query)
        explicit_error_log_path = _resolve_query_error_log_path(query)
        latest_trial_dir = _find_latest_repairable_trial(cfg) or _get_latest_trial(cfg)
        details["query_notebook_path"] = explicit_notebook_path
        details["query_error_log_path"] = explicit_error_log_path
        details["preferred_notebook_path"] = preferred_notebook_path
        details["preferred_trial_dir"] = preferred_trial_dir
        details["latest_trial_dir"] = latest_trial_dir

        target_notebook_path = explicit_notebook_path
        target_trial_dir = preferred_trial_dir
        error_log_path = explicit_error_log_path

        if preferred_run_result is not None:
            details["recent_run_result"] = {
                "notebook_path": preferred_run_result.notebook_path,
                "trial_dir": preferred_run_result.trial_dir,
                "status": preferred_run_result.status,
                "error_log_path": preferred_run_result.error_log_path,
                "run_log_path": preferred_run_result.run_log_path,
                "metadata": dict(preferred_run_result.metadata or {}),
            }
            if preferred_run_result.notebook_path and not target_notebook_path:
                target_notebook_path = preferred_run_result.notebook_path
            if preferred_run_result.trial_dir and not target_trial_dir:
                target_trial_dir = preferred_run_result.trial_dir
            if preferred_run_result.error_log_path and not error_log_path:
                error_log_path = preferred_run_result.error_log_path

        if not target_notebook_path and preferred_notebook_path and os.path.exists(preferred_notebook_path):
            target_notebook_path = preferred_notebook_path
        if not target_trial_dir and latest_trial_dir:
            target_trial_dir = latest_trial_dir
        if not target_notebook_path and target_trial_dir:
            target_notebook_path = _derive_notebook_path_for_trial(
                target_trial_dir,
                executed=True,
            ) or _derive_notebook_path_for_trial(target_trial_dir)
        if not error_log_path and target_trial_dir:
            error_log_path = _error_log_path_for_trial(target_trial_dir)
        if target_notebook_path and not target_trial_dir:
            target_trial_dir = os.path.dirname(target_notebook_path)

        details["target_trial_dir"] = target_trial_dir

        if not target_notebook_path or not os.path.exists(target_notebook_path):
            return {
                "action": "autofix",
                "status": "autofix_missing_context",
                "message": (
                    "External notebook autofix could not find a recent failed notebook run or explicit notebook path. "
                    "In future multi-turn sessions, this skill will prefer the latest failed run result from session state."
                ),
                "query": query,
                "target_notebook_path": target_notebook_path,
                "error_log_path": error_log_path,
                "patched_notebook_path": None,
                "legacy_entry": _LEGACY_AUTOFIX_ENTRY,
                "details": details,
            }

        nb = nbformat.read(target_notebook_path, as_version=4)
        errors = collect_cell_errors(nb)
        error_source = "notebook_outputs" if errors else None

        if not errors and error_log_path:
            errors = _parse_dumped_error_log(error_log_path)
            if errors:
                error_source = "error_log"

        details["resolved_error_source"] = error_source
        details["error_log_path"] = error_log_path
        details["found_recent_failed_run"] = bool(
            preferred_run_result is not None and preferred_run_result.error_log_path
        )

        if not errors:
            return {
                "action": "autofix",
                "status": "autofix_missing_structured_errors",
                "message": (
                    "External autofix found a notebook context, but it could not recover structured cell errors "
                    "from the notebook outputs or the available error log."
                ),
                "query": query,
                "target_notebook_path": target_notebook_path,
                "error_log_path": error_log_path,
                "patched_notebook_path": None,
                "legacy_entry": _LEGACY_AUTOFIX_ENTRY,
                "details": details,
            }

        mutable_indices = identify_mutable_cells(nb, cfg)
        error_cell_indices = {
            int(error.get("cell_index"))
            for error in errors
            if error.get("cell_index") is not None
        }
        effective_mutable_indices = [idx for idx in mutable_indices if idx in error_cell_indices]
        if not effective_mutable_indices:
            effective_mutable_indices = None

        fixed_nb, changed, method = attempt_fix_notebook(
            nb,
            errors,
            cfg,
            mutable_indices=effective_mutable_indices,
        )

        details["error_count"] = len(errors)
        details["mutable_indices"] = mutable_indices
        details["effective_mutable_indices"] = effective_mutable_indices or []
        details["repair_method"] = method

        if not changed:
            return {
                "action": "autofix",
                "status": "autofix_no_change",
                "message": (
                    "External autofix invoked the legacy repair entry, but it could not derive a valid patch "
                    "from the current notebook and error context."
                ),
                "query": query,
                "target_notebook_path": target_notebook_path,
                "error_log_path": error_log_path,
                "patched_notebook_path": None,
                "legacy_entry": _LEGACY_AUTOFIX_ENTRY,
                "details": details,
            }

        patched_notebook_path = _patched_notebook_path(target_notebook_path)
        nbformat.write(fixed_nb, patched_notebook_path)

        return {
            "action": "autofix",
            "status": "autofix_applied_via_legacy",
            "message": "External notebook autofix applied a legacy repair pass to the current failed notebook.",
            "query": query,
            "target_notebook_path": target_notebook_path,
            "error_log_path": error_log_path,
            "patched_notebook_path": patched_notebook_path,
            "legacy_entry": _LEGACY_AUTOFIX_ENTRY,
            "details": details,
        }
    except Exception as exc:
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc(limit=8)
        return {
            "action": "autofix",
            "status": "autofix_legacy_failed",
            "message": "External notebook autofix bridge was invoked, but the legacy repair entry failed.",
            "query": query,
            "target_notebook_path": preferred_notebook_path,
            "error_log_path": preferred_run_result.error_log_path if preferred_run_result is not None else None,
            "patched_notebook_path": None,
            "legacy_entry": _LEGACY_AUTOFIX_ENTRY,
            "details": details,
        }
