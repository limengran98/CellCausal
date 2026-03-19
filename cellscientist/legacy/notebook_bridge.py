from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, Optional

from ..core.config_loader import _expand_vars, load_yaml_prompts
from ..legacy.llm_resilience import (
    is_retryable_provider_error,
    resolve_bridge_llm_providers,
    summarize_provider_error,
)
from ..pipeline.config import apply_pipeline_overrides, load_pipeline_config, pipeline_extra_env
from ..pipeline.utils import load_json, project_root, resolve_h5_path_unified

_LEGACY_GENERATE_ENTRY = "cellscientist.core.prompt_orchestrator.phase_generate"
_LEGACY_EXECUTE_ENTRY = "cellscientist.core.prompt_orchestrator.phase_execute"


def _load_legacy_experiment_config() -> tuple[Dict[str, Any], Dict[str, str]]:
    root = project_root()
    pipeline_cfg_path = os.path.join(root, "configs", "pipeline_config.json")
    experiment_cfg_path = os.path.join(root, "configs", "experiment_config.json")

    pipe_cfg = load_pipeline_config(pipeline_cfg_path)
    for key, value in (pipeline_extra_env(pipe_cfg) or {}).items():
        if value is not None:
            os.environ[str(key)] = str(value)

    base_cfg = load_json(experiment_cfg_path)
    merged_cfg = apply_pipeline_overrides("Experiment", base_cfg, pipe_cfg)

    expand_env = dict(os.environ)
    for key, value in merged_cfg.items():
        if isinstance(value, (str, int, float, bool)):
            expand_env[str(key)] = str(value)
    cfg = _expand_vars(merged_cfg, expand_env)
    cfg["prompts"] = load_yaml_prompts(os.path.join(root, "prompts"))

    return cfg, {
        "pipeline_config": pipeline_cfg_path,
        "experiment_config": experiment_cfg_path,
    }


def _resolve_prompt_file(cfg: Dict[str, Any]) -> str:
    prompt_file = str(
        ((cfg.get("prompt_branch") or {}) if isinstance(cfg.get("prompt_branch"), dict) else {}).get(
            "prompt_file"
        )
        or "prompts/pipeline_prompt.yaml"
    )
    if os.path.isabs(prompt_file):
        return prompt_file
    return os.path.join(project_root(), prompt_file)


def _resolve_save_root(cfg: Dict[str, Any]) -> str:
    prompt_branch = (cfg.get("prompt_branch") or {}) if isinstance(cfg.get("prompt_branch"), dict) else {}
    save_root = str(prompt_branch.get("save_root") or ((cfg.get("paths") or {}) if isinstance(cfg.get("paths"), dict) else {}).get("design_execution_root") or os.getcwd())
    return os.path.abspath(save_root)


def _build_bridge_run_name(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _resolve_query_notebook_path(query: str) -> Optional[str]:
    matches = re.findall(r'([^\s"\']+\.ipynb)', query)
    if not matches:
        return None

    root = project_root()
    for raw_path in matches:
        candidates = [raw_path]
        if not os.path.isabs(raw_path):
            candidates.append(os.path.join(os.getcwd(), raw_path))
            candidates.append(os.path.join(root, raw_path))
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return None


def _stage_notebook_for_execution(notebook_path: str, cfg: Dict[str, Any]) -> str:
    save_root = _resolve_save_root(cfg)
    os.makedirs(save_root, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(save_root, f"bridge_exec_{timestamp}_{os.getpid()}")
    os.makedirs(trial_dir, exist_ok=True)

    staged_path = os.path.join(trial_dir, "notebook_prompt.ipynb")
    shutil.copy2(notebook_path, staged_path)
    return trial_dir


def _trace_path_for_trial(trial_dir: Optional[str]) -> Optional[str]:
    if not trial_dir:
        return None
    return os.path.join(trial_dir, "intermediate", "trace", "task_trace.json")


def _error_log_path_for_trial(trial_dir: Optional[str]) -> Optional[str]:
    if not trial_dir:
        return None
    candidates = (
        os.path.join(trial_dir, "framework_error_traceback.txt"),
        os.path.join(trial_dir, "analysis_llm_error_traceback.txt"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _derive_notebook_path_for_trial(trial_dir: Optional[str], *, executed: bool = False) -> Optional[str]:
    if not trial_dir:
        return None

    candidates = []
    if executed:
        candidates.extend(
            [
                os.path.join(trial_dir, "notebook_prompt_exec.ipynb"),
                os.path.join(trial_dir, "final_keep", "notebook_prompt_exec.ipynb"),
            ]
        )

    candidates.extend(
        [
            os.path.join(trial_dir, "notebook_prompt.ipynb"),
            os.path.join(trial_dir, "final_keep", "notebook_prompt.ipynb"),
        ]
    )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _write_json_file(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _format_attempt_summary(llm_attempts: list[Dict[str, Any]]) -> str:
    if not llm_attempts:
        return "No provider attempts were recorded."

    lines = []
    for idx, attempt in enumerate(llm_attempts, start=1):
        model = attempt.get("model") or "unknown-model"
        base_url = attempt.get("base_url") or "unknown-base-url"
        status = attempt.get("status") or "unknown"
        error_summary = attempt.get("error_summary")
        line = f"{idx}. {model} @ {base_url} -> {status}"
        if error_summary:
            line += f" | {error_summary}"
        lines.append(line)
    return "\n".join(lines)


def _create_degraded_notebook(
    *,
    cfg: Dict[str, Any],
    query: str,
    prompt_file: str,
    trial_dir: str,
    llm_resolution: Dict[str, Any],
    llm_attempts: list[Dict[str, Any]],
    degraded_reason: str,
) -> Dict[str, Optional[str]]:
    os.makedirs(trial_dir, exist_ok=True)
    final_dir = os.path.join(trial_dir, "final_keep")
    intermediate_dir = os.path.join(trial_dir, "intermediate")
    trace_dir = os.path.join(intermediate_dir, "trace")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    dataset_name = cfg.get("dataset_name") or "unknown_dataset"
    data_h5_path = resolve_h5_path_unified(cfg)
    notebook_path = os.path.join(trial_dir, "notebook_prompt_degraded.ipynb")
    copied_notebook_path = os.path.join(final_dir, "notebook_prompt_degraded.ipynb")
    run_log_path = _trace_path_for_trial(trial_dir)
    attempt_summary = _format_attempt_summary(llm_attempts)

    notebook_doc = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": (
                    "# Degraded Notebook Stub\n\n"
                    "This notebook was generated by the notebook bridge because all configured LLM "
                    "providers for legacy generation failed. It is a minimal artifact for inspection "
                    "and manual continuation."
                ),
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": (
                    "task_context = {\n"
                    f"    'query': {query!r},\n"
                    f"    'dataset_name': {dataset_name!r},\n"
                    f"    'data_h5_path': {data_h5_path!r},\n"
                    f"    'prompt_file': {prompt_file!r},\n"
                    f"    'trial_dir': {trial_dir!r},\n"
                    "}\n"
                    "task_context"
                ),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": (
                    "## TODO / Next Steps\n\n"
                    "- Re-run notebook generation with a healthy provider or corrected endpoint.\n"
                    "- Review the technical specification and domain knowledge artifacts already written for this trial.\n"
                    "- Replace this stub with a full generated notebook once provider access is restored."
                ),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": (
                    "## Provider Failure Summary\n\n"
                    f"Reason: {degraded_reason}\n\n"
                    "Attempt log:\n\n"
                    f"{attempt_summary}"
                ),
            },
        ],
    }

    with open(notebook_path, "w", encoding="utf-8") as handle:
        json.dump(notebook_doc, handle, ensure_ascii=False, indent=2)
    shutil.copy2(notebook_path, copied_notebook_path)

    if run_log_path:
        _write_json_file(
            run_log_path,
            {
                "status": "generated_degraded",
                "trial_dir": trial_dir,
                "notebook_path": notebook_path,
                "degraded_reason": degraded_reason,
                "llm_resolution": llm_resolution,
                "llm_attempts": llm_attempts,
            },
        )

    return {
        "notebook_path": notebook_path,
        "run_log_path": run_log_path,
    }


def bridge_generate_notebook(
    query: str,
    *,
    bridge_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {"query": query}
    cfg: Optional[Dict[str, Any]] = None
    llm_attempts: list[Dict[str, Any]] = []
    try:
        from ..core.execution_workflow import _inject_api_key, _setup_stage1_resources
        from ..core.prompt_orchestrator import phase_generate

        cfg, config_paths = _load_legacy_experiment_config()
        prompt_file = _resolve_prompt_file(cfg)
        trial_dir = os.path.join(_resolve_save_root(cfg), _build_bridge_run_name("bridge_generate"))
        provider_plan = resolve_bridge_llm_providers(cfg, bridge_override=bridge_override)
        providers = provider_plan["providers"]
        details.update(config_paths)
        details["prompt_file"] = prompt_file
        details["trial_dir"] = trial_dir
        details["run_log_path"] = _trace_path_for_trial(trial_dir)
        details["llm_resolution"] = provider_plan["llm_resolution"]

        last_exc: Optional[BaseException] = None
        last_traceback: Optional[str] = None
        non_retryable_failure = False

        for index, provider in enumerate(providers, start=1):
            provider_diag = deepcopy(provider["diagnostic"])
            attempt_record = {
                "provider_name": provider_diag.get("provider_name") or provider_diag.get("name"),
                "provider": provider_diag.get("provider_name") or provider_diag.get("name"),
                "model": provider_diag.get("model"),
                "base_url": provider_diag.get("normalized_base_url") or provider_diag.get("base_url"),
                "source": provider_diag.get("source") or provider_diag.get("config_source"),
                "config_source": provider_diag.get("config_source"),
                "api_key_source": provider_diag.get("api_key_source"),
                "warnings": provider_diag.get("warnings") or [],
                "status": "attempting",
            }

            if not provider_diag.get("enabled", True):
                attempt_record["status"] = "provider_invalid_config"
                attempt_record["error_summary"] = provider_diag.get("skip_reason") or "provider config is incomplete"
                llm_attempts.append(attempt_record)
                continue

            try:
                cfg_attempt = deepcopy(cfg)
                cfg_attempt["llm"] = deepcopy(provider["runtime_llm"])

                _inject_api_key(cfg_attempt)
                _setup_stage1_resources(cfg_attempt, True, spec_path=prompt_file)

                generation_result = phase_generate(
                    cfg_attempt,
                    prompt_file,
                    run_name=os.path.basename(trial_dir),
                )
                resolved_trial_dir = generation_result.get("trial_dir") or trial_dir
                notebook_path = generation_result.get("notebook_path") or _derive_notebook_path_for_trial(
                    resolved_trial_dir
                )
                attempt_record["status"] = "success"
                llm_attempts.append(attempt_record)

                details["trial_dir"] = resolved_trial_dir
                details["run_log_path"] = _trace_path_for_trial(resolved_trial_dir)
                details["llm_attempts"] = llm_attempts
                details["final_provider_used"] = provider_diag

                return {
                    "action": "generate",
                    "status": "generated_via_legacy" if index == 1 else "generated_via_fallback",
                    "message": (
                        "Legacy notebook generation entry completed successfully."
                        if index == 1
                        else "Legacy notebook generation succeeded after provider failover."
                    ),
                    "query": query,
                    "trial_dir": resolved_trial_dir,
                    "notebook_path": notebook_path,
                    "legacy_entry": _LEGACY_GENERATE_ENTRY,
                    "details": details,
                }
            except Exception as exc:
                last_exc = exc
                last_traceback = traceback.format_exc(limit=8)
                attempt_record["status"] = (
                    "provider_retryable_error" if is_retryable_provider_error(exc) else "provider_non_retryable_error"
                )
                attempt_record["error_summary"] = summarize_provider_error(exc)
                llm_attempts.append(attempt_record)

                if not is_retryable_provider_error(exc):
                    non_retryable_failure = True
                    break

        details["llm_attempts"] = llm_attempts
        details["final_provider_used"] = None
        details["traceback"] = last_traceback

        if llm_attempts and not non_retryable_failure:
            degraded_reason = (
                "All configured LLM providers failed with retryable provider/network errors during legacy notebook generation."
            )
            degraded = _create_degraded_notebook(
                cfg=cfg,
                query=query,
                prompt_file=prompt_file,
                trial_dir=trial_dir,
                llm_resolution=details["llm_resolution"],
                llm_attempts=llm_attempts,
                degraded_reason=degraded_reason,
            )
            details["degraded_reason"] = degraded_reason
            details["run_log_path"] = degraded.get("run_log_path")
            details["trial_dir"] = trial_dir

            return {
                "action": "generate",
                "status": "generated_degraded",
                "message": (
                    "Legacy notebook generation could not reach a healthy provider, so the bridge "
                    "generated a degraded stub notebook instead."
                ),
                "query": query,
                "trial_dir": trial_dir,
                "notebook_path": degraded.get("notebook_path"),
                "legacy_entry": _LEGACY_GENERATE_ENTRY,
                "details": details,
            }

        details["error"] = str(last_exc) if last_exc is not None else "unknown_generation_error"
        return {
            "action": "generate",
            "status": "legacy_generation_failed",
            "message": (
                "Legacy notebook generation entry was invoked, but generation could not complete "
                "under the current runtime conditions."
            ),
            "query": query,
            "trial_dir": details.get("trial_dir"),
            "notebook_path": None,
            "legacy_entry": _LEGACY_GENERATE_ENTRY,
            "details": details,
        }
    except Exception as exc:
        if cfg is not None:
            try:
                details["run_log_path"] = _trace_path_for_trial(details.get("trial_dir"))
            except Exception:
                pass
        details["llm_attempts"] = llm_attempts
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc(limit=8)
        return {
            "action": "generate",
            "status": "legacy_generation_failed",
            "message": (
                "Legacy notebook generation entry was invoked, but generation could not complete "
                "under the current runtime conditions."
            ),
            "query": query,
            "trial_dir": details.get("trial_dir"),
            "notebook_path": None,
            "legacy_entry": _LEGACY_GENERATE_ENTRY,
            "details": details,
        }


def bridge_execute_notebook(
    query: str,
    *,
    preferred_notebook_path: Optional[str] = None,
    preferred_trial_dir: Optional[str] = None,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {"query": query}
    try:
        from ..core.execution_workflow import _inject_api_key, _setup_stage1_resources
        from ..core.prompt_orchestrator import _get_latest_trial, phase_execute

        cfg, config_paths = _load_legacy_experiment_config()
        prompt_file = _resolve_prompt_file(cfg)
        details.update(config_paths)
        details["prompt_file"] = prompt_file

        explicit_notebook_path = _resolve_query_notebook_path(query)
        latest_trial_dir = _get_latest_trial(cfg)
        details["query_notebook_path"] = explicit_notebook_path
        details["preferred_notebook_path"] = preferred_notebook_path
        details["preferred_trial_dir"] = preferred_trial_dir
        details["latest_trial_dir"] = latest_trial_dir

        if explicit_notebook_path:
            trial_dir = _stage_notebook_for_execution(explicit_notebook_path, cfg)
            details["staged_trial_dir"] = trial_dir
            notebook_path = explicit_notebook_path
        elif preferred_notebook_path and os.path.exists(preferred_notebook_path):
            trial_dir = _stage_notebook_for_execution(preferred_notebook_path, cfg)
            details["staged_trial_dir"] = trial_dir
            notebook_path = preferred_notebook_path
        elif preferred_trial_dir and os.path.exists(preferred_trial_dir):
            trial_dir = preferred_trial_dir
            notebook_path = _derive_notebook_path_for_trial(preferred_trial_dir)
        else:
            trial_dir = latest_trial_dir
            notebook_path = _derive_notebook_path_for_trial(latest_trial_dir)

        if not trial_dir:
            return {
                "action": "execute",
                "status": "legacy_execute_missing_notebook",
                "message": (
                    "Legacy notebook execution is wired up, but no notebook path was provided "
                    "and no latest legacy trial directory could be found."
                ),
                "query": query,
                "notebook_path": notebook_path,
                "trial_dir": trial_dir,
                "error_log_path": None,
                "run_log_path": None,
                "legacy_entry": _LEGACY_EXECUTE_ENTRY,
                "details": details,
            }

        _inject_api_key(cfg)
        _setup_stage1_resources(cfg, True, spec_path=prompt_file)

        exec_result = phase_execute(cfg, trial_dir)
        executed_trial_dir = exec_result.get("trial_dir") or trial_dir
        exec_notebook = exec_result.get("exec_notebook") or _derive_notebook_path_for_trial(
            executed_trial_dir,
            executed=True,
        )
        metrics = exec_result.get("metrics") or {}
        run_log_path = _trace_path_for_trial(executed_trial_dir)
        error_log_path = _error_log_path_for_trial(executed_trial_dir)
        details.update(
            {
                "trial_dir": executed_trial_dir,
                "exec_notebook": exec_notebook,
                "metrics_present": bool(metrics),
            }
        )

        return {
            "action": "execute",
            "status": "executed_via_legacy",
            "message": "Legacy notebook execution entry completed successfully.",
            "query": query,
            "notebook_path": exec_notebook or notebook_path,
            "trial_dir": executed_trial_dir,
            "error_log_path": error_log_path,
            "run_log_path": run_log_path,
            "legacy_entry": _LEGACY_EXECUTE_ENTRY,
            "details": details,
        }
    except Exception as exc:
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc(limit=8)
        trial_dir = details.get("staged_trial_dir") or details.get("latest_trial_dir")
        notebook_path = preferred_notebook_path or details.get("query_notebook_path")
        if notebook_path is None and trial_dir:
            notebook_path = _derive_notebook_path_for_trial(str(trial_dir))
        return {
            "action": "execute",
            "status": "legacy_execution_failed",
            "message": (
                "Legacy notebook execution entry was invoked, but execution could not complete "
                "under the current runtime conditions."
            ),
            "query": query,
            "notebook_path": notebook_path,
            "trial_dir": str(trial_dir) if trial_dir else None,
            "error_log_path": _error_log_path_for_trial(str(trial_dir)) if trial_dir else None,
            "run_log_path": _trace_path_for_trial(str(trial_dir)) if trial_dir else None,
            "legacy_entry": _LEGACY_EXECUTE_ENTRY,
            "details": details,
        }
