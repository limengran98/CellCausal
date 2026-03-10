#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified CellScientist pipeline runner.

This entrypoint uses the original subprocess-based execution style (#16 legacy
workflow): ``execution_workflow.py`` and ``review_workflow.py`` are launched as
subprocesses so that each stage runs in its own process with full stdout/stderr
capture and log-file-based metric parsing.

The multi-agent collaboration framework (agents, orchestrator, message bus) lives
inside those subprocess modules and remains fully intact.
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
    parse_experiment_log,
    parse_review_log,
    experiment_scores_from_artifacts,
    review_scores_from_artifacts,
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
from cellscientist.pipeline.logging_system import create_tiered_logger


def _default_stage_map() -> Dict[str, Dict[str, Any]]:
    """Stage map used by the unified pipeline.

    We keep Phase-numbered keys internally to preserve the runner's log/metric parsers.
    The user-facing file structure and naming are unified (no Phase folders).
    """
    return {
        "Experiment": {
            "folder": project_root(),
            "module": "cellscientist.core.execution_workflow",
            "config": os.path.join(project_root(), "configs", "experiment_config.json"),
            "entry": ["python", "-m", "cellscientist.core.execution_workflow"],
        },
        "Review": {
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
        help="Run only the experiment stage.",
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
        stage_map["Experiment"]["config"] = args.experiment_config
    if args.review_config:
        stage_map["Review"]["config"] = args.review_config

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

    # 4) Create logs directory
    logs_dir = _make_logs_dir(dataset_name)

    # 5) Initialize 4-Tier Logger
    # Gather full config for logger initialization
    # Note: Using lowercase keys ('experiment', 'review') for internal consistency
    # while display names use title case ('Experiment', 'Review')
    full_config = {
        "pipeline": pipe_cfg,
        "stages": {
            "experiment": stage_map.get("Experiment", {}).get("_loaded_cfg", {}),
            "review": stage_map.get("Review", {}).get("_loaded_cfg", {})
        }
    }
    logger = create_tiered_logger(logs_dir, full_config, dataset_name)

    # Log pipeline initialization
    logger.full_log("=" * 80)
    logger.full_log("PIPELINE EXECUTION START")
    logger.full_log("=" * 80)
    logger.full_log(f"Dataset: {dataset_name}")
    logger.full_log(f"Logs Directory: {logs_dir}")

    # 6) Pretty plan (using existing function - still outputs to console)
    print_execution_plan(stage_map, dataset_name)

    # Open additional log files for complete capture
    execution_detail_log = os.path.join(logs_dir, "execution_detail.log")
    console_output_log = os.path.join(logs_dir, "console_output.log")

    detail_fp = None
    console_output_fp = None

    try:
        detail_fp = open(execution_detail_log, "w", encoding="utf-8")
        console_output_fp = open(console_output_log, "w", encoding="utf-8")

        # Write header to detail log
        detail_fp.write("=" * 80 + "\n")
        detail_fp.write("PIPELINE EXECUTION DETAIL LOG\n")
        detail_fp.write("=" * 80 + "\n")
        detail_fp.write(f"Dataset: {dataset_name}\n")
        detail_fp.write(f"Logs Directory: {logs_dir}\n")
        detail_fp.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        detail_fp.write("=" * 80 + "\n\n")
        detail_fp.flush()

        # 7) Run Experiment Stage
        exp_cfg = stage_map["Experiment"]["config"]
        exp_log = os.path.join(logs_dir, "experiment.log")
        exp_t0 = time.time()

        cmd2 = stage_map["Experiment"]["entry"] + ["--config", exp_cfg, "run"]
        logger.console_info("")
        logger.console_info("🔄 EXPERIMENT STAGE", level=0)
        logger.full_log(f"Running Experiment Stage: {' '.join(cmd2)}")
        logger.full_log(f"Experiment log: {exp_log}")

        with open(exp_log, "w", encoding="utf-8") as fp:
            run_subprocess_streamed(
                cmd2,
                cwd=project_root(),
                phase_fp=fp,
                detail_fp=detail_fp,
                console_filter_fp=console_output_fp,
                extra_env=extra_env
            )
        exp_t1 = time.time()

        # 8) Discover Experiment output dir
        # Prefer config-declared design_execution_root; runner extractor uses it as base_dir.
        exp_loaded = stage_map["Experiment"].get("_loaded_cfg") or {}
        ge_root = ((exp_loaded.get("paths") or {}).get("design_execution_root")
                   or (exp_loaded.get("prompt_branch") or {}).get("save_root")
                   or os.path.join(results_root_for_dataset(dataset_name), "generate_execution"))
        ge_root = os.path.abspath(os.path.join(project_root(), ge_root)) if not os.path.isabs(ge_root) else ge_root

        experiment_out = extract_best_path_from_log(exp_log, stage="Experiment", base_dir=ge_root, t_start=exp_t0)

        if not experiment_out:
            # fall back to ge_root, review has its own selection logic
            experiment_out = ge_root

        logger.console_info(f"Output directory: {experiment_out}", level=1, symbol="📁")
        logger.full_log(f"Experiment output directory: {experiment_out}")

        # 9) Run Review Stage
        review_log = os.path.join(logs_dir, "review.log")
        review_t0 = time.time()

        if args.skip_review:
            logger.console_info("")
            logger.console_info("⏭ Skipping review/optimization stage (--skip-review)", level=0)
            logger.full_log("Review stage skipped by user")
            review_t1 = review_t0
        else:
            review_cfg = stage_map["Review"]["config"]
            cmd3 = stage_map["Review"]["entry"] + ["--config", review_cfg, "--source_path", experiment_out]
            logger.console_info("")
            logger.console_info("🔄 REVIEW STAGE", level=0)
            logger.full_log(f"Running Review Stage: {' '.join(cmd3)}")
            logger.full_log(f"Review log: {review_log}")

            with open(review_log, "w", encoding="utf-8") as fp:
                run_subprocess_streamed(
                    cmd3,
                    cwd=project_root(),
                    phase_fp=fp,
                    detail_fp=detail_fp,
                    console_filter_fp=console_output_fp,
                    extra_env=extra_env
                )
            review_t1 = time.time()

    finally:
        # Always close additional log files, even if an error occurred
        if detail_fp:
            try:
                detail_fp.write("\n" + "=" * 80 + "\n")
                detail_fp.write("PIPELINE EXECUTION COMPLETED\n")
                detail_fp.write("=" * 80 + "\n")
                detail_fp.close()
            except Exception as e:
                logger.full_log(f"Warning: Could not close detail log properly: {e}")

        if console_output_fp:
            try:
                console_output_fp.close()
            except Exception as e:
                logger.full_log(f"Warning: Could not close console output log properly: {e}")

    # 10) Collect metrics summary
    summary: Dict[str, Any] = {"dataset": dataset_name, "logs_dir": logs_dir, "stages": {}}

    # Experiment metrics
    metric2 = (((stage_map["Experiment"].get("_loaded_cfg") or {}).get("experiment") or {}).get("primary_metric")
               or "PCC")
    exp_text = Path(exp_log).read_text(encoding="utf-8", errors="ignore") if os.path.exists(exp_log) else ""
    exp_parsed = parse_experiment_log(exp_text, metric=metric2)
    exp_scores = experiment_scores_from_artifacts(ge_root, metric2, exp_t0, exp_t1)

    exp_attempted = int(exp_parsed.get("attempted") or 0)
    exp_succeeded = int(exp_parsed.get("succeeded") or 0)
    exp_bug = int(exp_parsed.get("bug") or 0)
    exp_clean = int(exp_parsed.get("clean_success") or 0)
    sr2, clean2, bug2 = rates(exp_attempted, exp_clean, exp_succeeded, exp_bug)

    summary["stages"]["Experiment"] = {
        "attempted": exp_attempted,
        "succeeded": exp_succeeded,
        "bug": exp_bug,
        "clean_success": exp_clean,
        "success_rate": sr2,
        "clean_rate": clean2,
        "bug_rate": bug2,
        "avg_at_budget": mean_safe(exp_scores),
        "best_at_budget": max(exp_scores) if exp_scores else None,
        "best_metric": metric2,
        "budget": exp_attempted,
        "time_sec": float(exp_t1 - exp_t0),
    }

    # Review metrics (if ran)
    if not args.skip_review:
        review_loaded = stage_map["Review"].get("_loaded_cfg") or {}
        metric3 = (((review_loaded.get("review") or {}).get("target_metric")) or metric2)
        rf_root = ((review_loaded.get("paths") or {}).get("review_feedback_root")
                   or os.path.join(results_root_for_dataset(dataset_name), "review_feedback"))
        rf_root = os.path.abspath(os.path.join(project_root(), rf_root)) if not os.path.isabs(rf_root) else rf_root

        review_text = Path(review_log).read_text(encoding="utf-8", errors="ignore") if os.path.exists(review_log) else ""
        review_parsed = parse_review_log(review_text, metric=metric3)
        review_scores = review_scores_from_artifacts(rf_root, metric3, review_t0, review_t1)

        review_attempted = int(review_parsed.get("attempted") or 0)
        review_succeeded = int(review_parsed.get("succeeded") or 0)
        review_bug = int(review_parsed.get("bug") or 0)
        review_clean = int(review_parsed.get("clean_success") or 0)
        sr3, clean3, bug3 = rates(review_attempted, review_clean, review_succeeded, review_bug)

        summary["stages"]["Review"] = {
            "attempted": review_attempted,
            "succeeded": review_succeeded,
            "bug": review_bug,
            "clean_success": review_clean,
            "success_rate": sr3,
            "clean_rate": clean3,
            "bug_rate": bug3,
            "avg_at_budget": mean_safe(review_scores),
            "best_at_budget": max(review_scores) if review_scores else None,
            "best_metric": metric3,
            "budget": review_attempted,
            "time_sec": float(review_t1 - review_t0),
        }

    # Total row (Experiment+Review)
    total_t = float((exp_t1 - exp_t0) + (review_t1 - review_t0))
    # keep totals focused on rates/timing; per-stage averages already recorded.
    summary["stages"]["Total"] = {"time_sec": total_t}

    # Log timing summary
    logger.console_info("")
    logger.print_timing(f"Experiment completed in {exp_t1 - exp_t0:.1f}s")
    if not args.skip_review:
        logger.print_timing(f"Review completed in {review_t1 - review_t0:.1f}s")
    logger.print_timing(f"Total pipeline time: {total_t:.1f}s")

    # Display final scoreboard
    print_final_scoreboard(summary)

    # 11) Final report + advanced metrics (runner post-processing)
    if not args.skip_final_report:
        logger.console_info("")
        logger.console_info("📝 Generating final reports...", level=0)
        generate_final_report_and_exports(summary, stage_map, pipe_cfg)
        maybe_generate_advanced_metrics(summary, stage_map, pipe_cfg)

    # 12) Finalize logger (saves evidence chain)
    logger.console_info("")
    logger.console_info(f"✅ Pipeline completed. Logs: {logs_dir}", level=0)
    logger.finalize()


if __name__ == "__main__":
    main()