#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified CellScientist pipeline runner.

This is the refactored, single entrypoint based on the original `runners` code.

Key refactor points (per requirements):
- Uses the runner framework (logging, config materialization, metrics, reporting).
- Integrates Design&Execution (Phase 2) and Review&Optimization (Phase 3) into one
  continuous business flow.
- Removes Phase 1 orchestration (not provided) and any runner code that depends on it.
- Keeps all non-Phase-1 logic intact.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from cellscientist.pipeline.utils import (
    ensure_project_cwd,
    project_root,
    results_root_for_dataset,
    run_subprocess_streamed,
    extract_best_path_from_log,
)
from cellscientist.pipeline.config import (
    load_pipeline_config,
    materialize_merged_configs,
    validate_configs,
    pipeline_extra_env,
)
from cellscientist.pipeline.metrics import (
    print_execution_plan,
    parse_phase2_log,
    parse_phase3_log,
    phase2_scores_from_artifacts,
    phase3_scores_from_artifacts,
    rates,
    mean_safe,
    print_final_scoreboard,
)
from cellscientist.pipeline.report import (
    generate_final_report_and_exports,
)
from cellscientist.pipeline.advanced_metrics import (
    maybe_generate_advanced_metrics,
)


def _default_stage_map() -> Dict[str, Dict[str, Any]]:
    """Stage map used by the unified pipeline.

    We keep Phase-numbered keys internally to preserve the runner's log/metric parsers.
    The user-facing file structure and naming are unified (no Phase folders).
    """
    return {
        "Phase 2": {
            "folder": project_root(),
            "module": "cellscientist.core.execution_workflow",
            "config": os.path.join(project_root(), "configs", "experiment_config.json"),
            "entry": ["python", "-m", "cellscientist.core.execution_workflow"],
        },
        "Phase 3": {
            "folder": project_root(),
            "module": "cellscientist.core.review_workflow",
            "config": os.path.join(project_root(), "configs", "review_config.json"),
            "entry": ["python", "-m", "cellscientist.core.review_workflow"],
        },
    }


def _make_logs_dir(dataset_name: str) -> str:
    base = results_root_for_dataset(dataset_name)
    logs = os.path.join(base, "run_logs")
    os.makedirs(logs, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(logs, f"pipeline_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-config",
        default=os.path.join(project_root(), "configs", "pipeline_config.json"),
        help="Optional pipeline-level overrides JSON.",
    )
    parser.add_argument(
        "--experiment-config",
        default=None,
        help="Override experiment (execution) stage config path.",
    )
    parser.add_argument(
        "--review-config",
        default=None,
        help="Override review/optimization stage config path.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Run only the experiment stage (Phase 2).",
    )
    parser.add_argument(
        "--skip-final-report",
        action="store_true",
        help="Skip final report and export logic (runner post-processing).",
    )
    args = parser.parse_args()

    ensure_project_cwd()

    stage_map = _default_stage_map()
    if args.experiment_config:
        stage_map["Phase 2"]["config"] = args.experiment_config
    if args.review_config:
        stage_map["Phase 3"]["config"] = args.review_config

    # 1) Load pipeline_config overrides (optional)
    pipe_cfg = load_pipeline_config(args.pipeline_config)

    # 1b) Extra environment to pass to all subprocesses (GPU selection, custom env vars, etc.)
    extra_env = pipeline_extra_env(pipe_cfg)
    # Also update current process so any config expansion (${VAR}) can see it.
    for _k, _v in (extra_env or {}).items():
        if _v is None:
            continue
        os.environ[str(_k)] = str(_v)


    # 2) Write merged configs (keeps original runner behavior)
    materialize_merged_configs(stage_map, pipe_cfg)

    # 3) Validate dataset_name is consistent
    dataset_name = validate_configs(stage_map)

    # 4) Pretty plan
    print_execution_plan(stage_map, dataset_name)

    # 5) Logs
    logs_dir = _make_logs_dir(dataset_name)

    # 6) Run Phase 2 (Design & Execution)
    p2_cfg = stage_map["Phase 2"]["config"]
    p2_log = os.path.join(logs_dir, "phase2.log")
    p2_t0 = time.time()

    cmd2 = stage_map["Phase 2"]["entry"] + ["--config", p2_cfg, "run"]
    print(f"\n[PIPELINE] ▶ Running Experiment Stage (Phase 2)\n  cmd: {' '.join(cmd2)}\n  log: {p2_log}\n", flush=True)
    with open(p2_log, "w", encoding="utf-8") as fp:
        run_subprocess_streamed(cmd2, cwd=project_root(), phase_fp=fp, extra_env=extra_env)
    p2_t1 = time.time()

    # 7) Discover Phase 2 output dir
    # Prefer config-declared design_execution_root; runner extractor uses it as base_dir.
    p2_loaded = stage_map["Phase 2"].get("_loaded_cfg") or {}
    ge_root = ((p2_loaded.get("paths") or {}).get("design_execution_root")
               or (p2_loaded.get("prompt_branch") or {}).get("save_root")
               or os.path.join(results_root_for_dataset(dataset_name), "generate_execution"))
    ge_root = os.path.abspath(os.path.join(project_root(), ge_root)) if not os.path.isabs(ge_root) else ge_root

    phase2_out = extract_best_path_from_log(p2_log, phase="Phase 2", base_dir=ge_root, t_start=p2_t0)

    if not phase2_out:
        # fall back to ge_root, phase3 has its own selection logic
        phase2_out = ge_root

    print(f"[PIPELINE] Phase 2 output directory: {phase2_out}", flush=True)

    # 8) Run Phase 3 (Review & Optimization)
    p3_log = os.path.join(logs_dir, "phase3.log")
    p3_t0 = time.time()

    if args.skip_review:
        print("\n[PIPELINE] ⏭ Skipping review/optimization stage (--skip-review).", flush=True)
        p3_t1 = p3_t0
    else:
        p3_cfg = stage_map["Phase 3"]["config"]
        cmd3 = stage_map["Phase 3"]["entry"] + ["--config", p3_cfg, "--source_path", phase2_out]
        print(f"\n[PIPELINE] ▶ Running Review/Optimization Stage (Phase 3)\n  cmd: {' '.join(cmd3)}\n  log: {p3_log}\n", flush=True)
        with open(p3_log, "w", encoding="utf-8") as fp:
            run_subprocess_streamed(cmd3, cwd=project_root(), phase_fp=fp, extra_env=extra_env)
        p3_t1 = time.time()

    # 9) Collect metrics summary (runner behavior preserved; Phase 1 excluded)
    summary: Dict[str, Any] = {"dataset": dataset_name, "logs_dir": logs_dir, "stages": {}}

    # Phase 2 metrics
    metric2 = (((stage_map["Phase 2"].get("_loaded_cfg") or {}).get("experiment") or {}).get("primary_metric")
               or "PCC")
    p2_text = Path(p2_log).read_text(encoding="utf-8", errors="ignore") if os.path.exists(p2_log) else ""
    p2_parsed = parse_phase2_log(p2_text, metric=metric2)
    p2_scores = phase2_scores_from_artifacts(ge_root, metric2, p2_t0, p2_t1)

    p2_attempted = int(p2_parsed.get("attempted") or 0)
    p2_succeeded = int(p2_parsed.get("succeeded") or 0)
    p2_bug = int(p2_parsed.get("bug") or 0)
    p2_clean = int(p2_parsed.get("clean_success") or 0)
    sr2, clean2, bug2 = rates(p2_attempted, p2_clean, p2_succeeded, p2_bug)

    summary["stages"]["Phase 2"] = {
        "attempted": p2_attempted,
        "succeeded": p2_succeeded,
        "bug": p2_bug,
        "clean_success": p2_clean,
        "success_rate": sr2,
        "clean_rate": clean2,
        "bug_rate": bug2,
        "avg_at_budget": mean_safe(p2_scores),
        "best_at_budget": max(p2_scores) if p2_scores else None,
        "best_metric": metric2,
        "budget": p2_attempted,
        "time_sec": float(p2_t1 - p2_t0),
    }

    # Phase 3 metrics (if ran)
    if not args.skip_review:
        p3_loaded = stage_map["Phase 3"].get("_loaded_cfg") or {}
        metric3 = (((p3_loaded.get("review") or {}).get("target_metric")) or metric2)
        rf_root = ((p3_loaded.get("paths") or {}).get("review_feedback_root")
                   or os.path.join(results_root_for_dataset(dataset_name), "review_feedback"))
        rf_root = os.path.abspath(os.path.join(project_root(), rf_root)) if not os.path.isabs(rf_root) else rf_root

        p3_text = Path(p3_log).read_text(encoding="utf-8", errors="ignore") if os.path.exists(p3_log) else ""
        p3_parsed = parse_phase3_log(p3_text, metric=metric3)
        p3_scores = phase3_scores_from_artifacts(rf_root, metric3, p3_t0, p3_t1)

        p3_attempted = int(p3_parsed.get("attempted") or 0)
        p3_succeeded = int(p3_parsed.get("succeeded") or 0)
        p3_bug = int(p3_parsed.get("bug") or 0)
        p3_clean = int(p3_parsed.get("clean_success") or 0)
        sr3, clean3, bug3 = rates(p3_attempted, p3_clean, p3_succeeded, p3_bug)

        summary["stages"]["Phase 3"] = {
            "attempted": p3_attempted,
            "succeeded": p3_succeeded,
            "bug": p3_bug,
            "clean_success": p3_clean,
            "success_rate": sr3,
            "clean_rate": clean3,
            "bug_rate": bug3,
            "avg_at_budget": mean_safe(p3_scores),
            "best_at_budget": max(p3_scores) if p3_scores else None,
            "best_metric": metric3,
            "budget": p3_attempted,
            "time_sec": float(p3_t1 - p3_t0),
        }

    # Total row (Phase 2+3)
    total_t = float((p2_t1 - p2_t0) + (p3_t1 - p3_t0))
    all_scores = list(p2_scores) + (list(summary["stages"].get("Phase 3", {}).get("scores", [])) if False else [])
    # keep totals focused on rates/timing; per-stage averages already recorded.
    summary["stages"]["Total"] = {"time_sec": total_t}

    print_final_scoreboard(summary)

    # 10) Final report + advanced metrics (runner post-processing)
    if not args.skip_final_report:
        generate_final_report_and_exports(summary, stage_map, pipe_cfg)
        maybe_generate_advanced_metrics(summary, stage_map, pipe_cfg)

    print(f"\n[PIPELINE] ✅ Completed. Logs: {logs_dir}\n", flush=True)


if __name__ == "__main__":
    main()