#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified CellScientist pipeline runner — FSM-driven.

This entrypoint delegates all experiment/review logic to the
:class:`~cellscientist.core.orchestrator.PipelineOrchestrator` (a Finite State
Machine).  There are no subprocess calls to ``execution_workflow.py`` or
``review_workflow.py``, and no log-file parsing for metrics — the orchestrator's
internal statistics are used directly for the scoreboard.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict

from cellscientist.pipeline.utils import (
    ensure_project_cwd,
    project_root,
    results_root_for_dataset,
)
from cellscientist.pipeline.config import (
    load_pipeline_config,
    pipeline_extra_env,
)
from cellscientist.pipeline.metrics import (
    print_final_scoreboard,
    scoreboard_from_orchestrator,
)
from cellscientist.pipeline.logging_system import create_tiered_logger
from cellscientist.core.orchestrator import run_orchestrator
from cellscientist.core.agent_monitor import Heartbeat


def _make_logs_dir(dataset_name: str) -> str:
    base = results_root_for_dataset(dataset_name)
    logs = os.path.join(base, "run_logs")
    os.makedirs(logs, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(logs, f"pipeline_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


async def async_main(args: argparse.Namespace) -> None:
    """Async entrypoint: load config, run orchestrator, render scoreboard."""
    ensure_project_cwd()

    # 1) Load pipeline config
    pipe_cfg = load_pipeline_config(args.pipeline_config)

    # 2) Apply environment overrides
    extra_env = pipeline_extra_env(pipe_cfg)
    for k, v in (extra_env or {}).items():
        if v is not None:
            os.environ[str(k)] = str(v)

    # 3) Resolve dataset name
    dataset_name = pipe_cfg.get("dataset_name", "BBBC036")

    # 4) Create logs directory
    logs_dir = _make_logs_dir(dataset_name)

    # 5) Merge CLI overrides into config
    if args.experiment_config:
        pipe_cfg.setdefault("_cli_overrides", {})["experiment_config"] = args.experiment_config
    if args.review_config:
        pipe_cfg.setdefault("_cli_overrides", {})["review_config"] = args.review_config
    if args.skip_review:
        pipe_cfg.setdefault("_cli_overrides", {})["skip_review"] = True

    # Store logs_dir in config so orchestrator/agents can find it
    pipe_cfg["_logs_dir"] = logs_dir

    # 6) Initialize tiered logger
    full_config: Dict[str, Any] = {"pipeline": pipe_cfg, "stages": {}}
    logger = create_tiered_logger(logs_dir, full_config, dataset_name)
    logger.full_log("=" * 80)
    logger.full_log("PIPELINE EXECUTION START (FSM-driven)")
    logger.full_log(f"Dataset: {dataset_name}")
    logger.full_log(f"Logs Directory: {logs_dir}")

    # 7) Run the orchestrator wrapped in an async Heartbeat
    t0 = time.time()
    async with Heartbeat("PipelineOrchestrator", interval=15):
        result = await run_orchestrator(pipe_cfg)
    t1 = time.time()

    # 8) Render scoreboard from the orchestrator's internal statistics
    _render_scoreboard(result, dataset_name, t1 - t0, logger, logs_dir)

    # 9) Optionally generate final report
    if not args.skip_final_report:
        _generate_reports(result, pipe_cfg, dataset_name, logs_dir, logger)

    logger.full_log("=" * 80)
    logger.full_log("PIPELINE EXECUTION COMPLETED")
    logger.console_info("")
    logger.console_info(f"✅ Pipeline completed. Logs: {logs_dir}", level=0)
    logger.finalize()


def _render_scoreboard(
    result: Dict[str, Any],
    dataset_name: str,
    total_time: float,
    logger: Any,
    logs_dir: str,
) -> None:
    """Render the final scoreboard using ONLY the orchestrator's internal stats.

    This permanently fixes the 0% (0/0) bug by reading from the orchestrator's
    summary dict instead of parsing log files.
    """
    status = result.get("status", "unknown")
    max_reached = result.get("max_iterations_reached", False)

    # Build the summary dict using the reusable helper; derive display values from it
    # to avoid any duplication of the success-rate calculation.
    summary = scoreboard_from_orchestrator(result, dataset_name, total_time)
    summary["logs_dir"] = logs_dir

    exp_stage = summary["stages"]["Experiment"]
    best_accuracy = exp_stage.get("best_at_budget") or 0.0
    total_iterations = exp_stage.get("attempted", 0)
    success_count = exp_stage.get("succeeded", 0)
    success_rate_pct = exp_stage["success_rate"] * 100  # decimal → percentage
    metric_name = exp_stage.get("best_metric", "PCC")

    # Print timing and status line
    logger.console_info("")
    logger.print_timing(f"Total pipeline time: {total_time:.1f}s")
    logger.console_info(
        f"📊 Result: status={status} | Best {metric_name}={best_accuracy:.4f} | "
        f"Success: {success_count}/{total_iterations} ({success_rate_pct:.0f}%) | "
        f"Max iterations reached: {max_reached}",
        level=0,
    )

    # Use existing scoreboard printer (handles Rich formatting)
    try:
        print_final_scoreboard(summary)
    except Exception as exc:
        logger.full_log(f"Warning: Could not render Rich scoreboard: {exc}")

    # Save orchestrator result as JSON artifact
    result_path = os.path.join(logs_dir, "orchestrator_result.json")
    try:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.full_log(f"Orchestrator result saved to: {result_path}")
    except Exception as exc:
        logger.full_log(f"Warning: Could not save orchestrator result: {exc}")


def _generate_reports(
    result: Dict[str, Any],
    pipe_cfg: Dict[str, Any],
    dataset_name: str,
    logs_dir: str,
    logger: Any,
) -> None:
    """Generate final reports using orchestrator state instead of log parsing."""
    logger.console_info("")
    logger.console_info("📝 Generating final reports...", level=0)
    try:
        from cellscientist.pipeline.report import generate_report_from_orchestrator
        from cellscientist.pipeline.advanced_metrics import maybe_generate_advanced_metrics

        # Save FSM transitions as a report artifact
        transitions_path = os.path.join(logs_dir, "fsm_transitions.json")
        with open(transitions_path, "w", encoding="utf-8") as f:
            json.dump(result.get("fsm_transitions", []), f, indent=2, default=str)

        generate_report_from_orchestrator(result, pipe_cfg, dataset_name, logs_dir)

        results_root = results_root_for_dataset(dataset_name)
        maybe_generate_advanced_metrics(
            orchestrator_result=result,
            logs_dir=logs_dir,
            results_root=results_root,
            dataset_name=dataset_name,
        )
        logger.console_info("✅ Reports generated.", level=1, symbol="📝")
    except Exception as exc:
        logger.full_log(f"Warning: Report generation failed: {exc}")
        logger.console_info(f"⚠️ Report generation failed: {exc}", level=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CellCausal FSM-driven pipeline runner."
    )
    parser.add_argument(
        "--pipeline-config",
        default=os.path.join(project_root(), "configs", "pipeline_config.json"),
        help="Path to pipeline_config.json.",
    )
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--review-config", default=None)
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--skip-final-report", action="store_true")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()