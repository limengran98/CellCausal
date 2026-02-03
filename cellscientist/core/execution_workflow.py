
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cellscientist_phase_2.py

import sys
import os
import json
import argparse
import glob
import shutil
import datetime
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Force Line Buffering
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from cellscientist.core.config_loader import load_full_config
from cellscientist.core.prompt_orchestrator import (
    phase_generate, phase_execute, phase_analyze, run_full_pipeline
)
# [NEW] Import TokenMeter
from cellscientist.core.llm_client import TokenMeter
from cellscientist.pipeline.utils import project_root

def _setup_stage1_resources(cfg: dict, enable_idea: bool = False):
    paths_cfg = (cfg.get("paths") or {}) if isinstance(cfg.get("paths"), dict) else {}

    # ---------------------------------------------------------------------
    # [UPDATED] Data Resolver
    # ---------------------------------------------------------------------
    explicit_h5 = paths_cfg.get("data_h5_path")
    if explicit_h5 and os.path.exists(str(explicit_h5)):
        h5_path = os.path.abspath(str(explicit_h5))
        os.environ["STAGE1_H5_PATH"] = h5_path
        print(f"[SETUP] Data Anchor (Explicit): {h5_path}", flush=True)
        s1_path = os.path.dirname(h5_path)
        final_ref_dir = s1_path
    else:
        data_root = paths_cfg.get("data_root")
        data_fname = paths_cfg.get("data_h5_filename")
        ds = cfg.get("dataset_name")
        h5_path = None
        final_ref_dir = None

        # 1. Determine Search Roots
        root_candidates: List[str] = []
        
        # A) Configured data_root
        if data_root:
            if os.path.isabs(str(data_root)):
                root_candidates.append(str(data_root))
            else:
                root_candidates.append(os.path.abspath(str(data_root)))
                root_candidates.append(os.path.abspath(os.path.join(project_root(), str(data_root))))

        # B) Standard Project Structure (fallback)
        # Looks in: ./data and ../data (relative to script execution)
        root_candidates.append(os.path.join(os.getcwd(), "data"))
        root_candidates.append(os.path.join(os.path.dirname(os.getcwd()), "data"))
        root_candidates.append(os.path.join(project_root(), "data"))

        # Remove duplicates
        root_candidates = sorted(list(set(root_candidates)), key=len, reverse=True)

        if data_fname:
            tried: List[str] = []
            
            for root_abs in root_candidates:
                if not os.path.exists(root_abs):
                    continue

                # Strategy 1: <root>/<dataset>/<file>
                if ds:
                    cand_ds = os.path.join(root_abs, str(ds), str(data_fname))
                    tried.append(cand_ds)
                    if os.path.exists(cand_ds):
                        h5_path = os.path.abspath(cand_ds)
                        break

                # Strategy 2: <root>/<file> (Directly in data folder - Your Case)
                cand_direct = os.path.join(root_abs, str(data_fname))
                tried.append(cand_direct)
                if os.path.exists(cand_direct):
                    h5_path = os.path.abspath(cand_direct)
                    break
                
                # Strategy 3: Recursive search
                matches = glob.glob(os.path.join(root_abs, "**", str(data_fname)), recursive=True)
                if matches:
                    h5_path = os.path.abspath(matches[0])
                    break

            if h5_path:
                os.environ["STAGE1_H5_PATH"] = h5_path
                print(f"[SETUP] Data Anchor Resolved: {h5_path}", flush=True)
                final_ref_dir = os.path.dirname(h5_path)
            else:
                print("[SETUP][WARN] Could not find data H5. Tried locations:", flush=True)
                for t in tried:
                    print(f"  - {t}", flush=True)

        # Legacy Phase-1-compatible path (optional fallback)
        if not h5_path:
            s1_dir_str = paths_cfg.get("stage1_analysis_dir")
            if s1_dir_str:
                s1_path = os.path.abspath(s1_dir_str)
                final_ref_dir = s1_path
                
                # Search for H5 in legacy dir
                if os.path.exists(os.path.join(s1_path, "REFERENCE_DATA.h5")):
                    h5_path = os.path.join(s1_path, "REFERENCE_DATA.h5")
                elif os.path.isdir(s1_path):
                     # Check subdirs
                    subdirs = sorted([os.path.join(s1_path, d) for d in os.listdir(s1_path) if os.path.isdir(os.path.join(s1_path, d))])
                    if subdirs:
                        final_ref_dir = subdirs[-1]
                        h5s = glob.glob(os.path.join(final_ref_dir, "*.h5"))
                        if h5s: h5_path = h5s[0]

            if h5_path:
                os.environ["STAGE1_H5_PATH"] = h5_path
                print(f"[SETUP] Data Anchor (Legacy): {h5_path}", flush=True)

    # Idea loading relative to the resolved H5 path
    if enable_idea:
        base_dir = os.path.dirname(h5_path) if h5_path else (final_ref_dir or os.getcwd())
        idea_path = os.path.join(base_dir, "idea.json")
        
        if os.path.exists(idea_path):
            os.environ["STAGE1_IDEA_PATH"] = idea_path
            print(f"[SETUP] Idea File: {idea_path}", flush=True)
        else:
            custom_idea = cfg.get("prompt_branch", {}).get("idea_file")
            if custom_idea and os.path.exists(custom_idea):
                os.environ["STAGE1_IDEA_PATH"] = os.path.abspath(custom_idea)
                print(f"[SETUP] Config Idea File: {os.environ['STAGE1_IDEA_PATH']}", flush=True)
            else:
                print("[SETUP][WARN] --use-idea ON but no idea.json found.", flush=True)
    else:
        if "STAGE1_IDEA_PATH" in os.environ:
            del os.environ["STAGE1_IDEA_PATH"]
        print("[SETUP] Idea Mode: OFF", flush=True)

def _inject_api_key(cfg: dict):
    """Ensure API Key is loaded into environment for llm_utils to find."""
    llm = cfg.get("llm", {})
    key = llm.get("api_key")
    base = llm.get("base_url")
    
    if key:
        os.environ["OPENAI_API_KEY"] = key
    if base:
        os.environ["OPENAI_BASE_URL"] = base
        
    print(f"[SETUP] LLM Config: URL={os.environ.get('OPENAI_BASE_URL', 'default')}, Key={'***' if key else 'None'}", flush=True)

def _check_success(metrics: dict, threshold: float, metric_key: str) -> Tuple[bool, float]:
    if not metrics:
        return False, -999.0

    winner = metrics.get("winner")
    if not winner:
        models = [k for k in (metrics.get("models") or {}).keys() if k != "config"]
        if not models:
            return False, -999.0
        winner = models[0]

    m_data = metrics.get(winner, (metrics.get("models") or {}).get(winner, {}))

    val = None
    if isinstance(m_data, dict) and "aggregate" in m_data and isinstance(m_data["aggregate"], dict):
        val = m_data["aggregate"].get(metric_key)

    if val is None and isinstance(m_data, dict) and "per_fold" in m_data and isinstance(m_data["per_fold"], dict):
        vals = []
        for f in m_data["per_fold"].values():
            if not isinstance(f, dict):
                continue
            v = f.get(metric_key)
            if v is None and isinstance(f.get("metrics"), dict):
                v = f["metrics"].get(metric_key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if vals:
            val = float(np.mean(vals))

    try:
        score = float(val) if val is not None else -999.0
    except Exception:
        score = -999.0

    print(f"[CHECK] {winner} | {metric_key}: {score:.4f} (Target > {threshold})", flush=True)
    return score > threshold, score

def _archive_run(trial_dir: str) -> Optional[str]:
    """
    Renames the given prompt/workspace directory to prompt_run_TIMESTAMP.

    NOTE: This keeps the existing "main" time-stamped artifacts behavior.
    """
    if not trial_dir or not os.path.exists(trial_dir):
        return None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = os.path.dirname(trial_dir)

    new_name = f"prompt_run_{ts}"
    new_path = os.path.join(parent, new_name)

    try:
        os.rename(trial_dir, new_path)
        print(f"[ARCHIVE] Saved run to: {new_name}", flush=True)
        return new_path
    except OSError:
        try:
            shutil.move(trial_dir, new_path)
            print(f"[ARCHIVE] Moved run to: {new_name}", flush=True)
            return new_path
        except Exception as e:
            print(f"[ARCHIVE][WARN] Failed to archive run: {e}", flush=True)
            return None

def _atomic_write_json(path: str, data: dict):
    """Best-effort atomic JSON write (won't raise)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[LOOP][WARN] Failed to write loop summary to {path}: {e}", flush=True)

def _get_save_root(cfg: Dict[str, Any]) -> str:
    # Follow prompt_orchestrator's logic
    return (cfg.get("prompt_branch", {}) or {}).get("save_root", (cfg.get("paths", {}) or {}).get("design_execution_root", os.getcwd()))

def run_loop(cfg: dict, prompt_file: Optional[str], use_idea: bool):
    exp_cfg = cfg.get("experiment", {}) or {}
    max_iters = int(exp_cfg.get("max_iterations", 1) or 1)
    threshold = float(exp_cfg.get("success_threshold", 0.0) or 0.0)
    pm = exp_cfg.get("primary_metric", "PCC") or "PCC"

    # [BUGFIX] Iteration data-loss fix:
    # Instead of reusing the same workspace that gets wiped each iteration,
    # we give each iteration a unique run_name so every iteration's artifacts are preserved.
    start_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    workspace_prefix = f"workspace_{start_ts}_{pid}"

    p_path = (prompt_file or
              (cfg.get("prompt_branch") or {}).get("prompt_file") or
              "prompts/pipeline_prompt.yaml")

    out_root = _get_save_root(cfg)
    summary_path = os.path.join(out_root, f"phase2_loop_summary_{start_ts}_{pid}.json")

    print(f"\n[LOOP] Max Iters: {max_iters} | Target: {pm} > {threshold}", flush=True)
    print(f"[LOOP] Workspace Prefix: {workspace_prefix}", flush=True)
    print(f"[LOOP] Save Root: {out_root}", flush=True)
    print(f"[LOOP] Loop Summary: {summary_path}", flush=True)

    # [TELEM] Reset meter before starting loop to clear any setup noise
    TokenMeter.get_and_reset()

    loop_started_ts = time.time()
    loop_started_iso = datetime.datetime.now().isoformat()

    iter_logs: List[Dict[str, Any]] = []
    best_score = -9999.0
    best_trial_dir: Optional[str] = None
    archived_dir: Optional[str] = None

    def _write_summary(final: bool = False):
        executed = len(iter_logs)
        valid = sum(1 for r in iter_logs if r.get("status") in {"VALID", "SUCCESS"} or (isinstance(r.get("score"), (int, float)) and r.get("score") > -999.0))
        criteria_met_n = sum(1 for r in iter_logs if r.get("criteria_met") is True)
        best = max([r.get("score", -999.0) for r in iter_logs] + [-999.0])
        
        # [TELEM] Aggregate Total Cost for the whole loop
        total_prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in iter_logs)
        total_completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in iter_logs)
        total_llm_time = sum(r.get("usage", {}).get("total_latency_sec", 0.0) for r in iter_logs)

        _atomic_write_json(summary_path, {
            "dataset": cfg.get("dataset_name"),
            "started_at": loop_started_iso,
            "finished_at": datetime.datetime.now().isoformat() if final else None,
            "primary_metric": pm,
            "success_threshold": threshold,
            "max_iterations": max_iters,
            "iterations_executed": executed,
            "valid_iterations": int(valid),
            "criteria_met_iterations": int(criteria_met_n),
            "success_rate": (criteria_met_n / float(executed)) if executed > 0 else 0.0,
            "validity_rate": (valid / float(executed)) if executed > 0 else 0.0,
            "best_score": float(best),
            "best_at_budget": float(best),
            "best_trial_dir": best_trial_dir,
            "archived_dir": archived_dir,
            # [TELEM] Add top-level cost stats
            "total_cost_tokens": total_prompt + total_completion,
            "total_cost_latency": total_llm_time,
            "iterations": iter_logs,
            "total_time_sec": float(time.time() - loop_started_ts),
        })

    try:
        for i in range(1, max_iters + 1):
            run_name = f"{workspace_prefix}_iter{i:03d}"
            print(f"\n{'='*40}\n🔄 ITERATION {i}/{max_iters} | run_name={run_name}\n{'='*40}", flush=True)

            # [TELEM] Reset meter at start of iteration to capture ONLY this iteration's usage
            TokenMeter.get_and_reset()

            iter_started = time.time()
            iter_trial_dir = None
            iter_score = -999.0
            iter_status = "UNKNOWN"
            criteria_met = False

            try:
                _setup_stage1_resources(cfg, use_idea)
                res = run_full_pipeline(cfg, p_path, run_name=run_name)

                iter_trial_dir = res.get("trial_dir")
                success, score = _check_success(res.get("metrics", {}), threshold, pm)
                iter_score = float(score)
                iter_status = "VALID" if score > -999.0 else "NO_METRIC"
                criteria_met = bool(success)

                # [TELEM] Capture Usage specific to this iteration
                usage_stats = TokenMeter.get_and_reset()
                print(f"[COST] Iteration {i}: {usage_stats['total_tokens']} tokens | {usage_stats['total_latency_sec']:.2f}s LLM time", flush=True)

                if score > -999.0 and score > best_score:
                    prev_best = best_score
                    best_score = score
                    best_trial_dir = iter_trial_dir
                    print(f"📈 [IMPROVEMENT] New best score: {score:.4f} (Prev: {prev_best:.4f}).", flush=True)

                iter_logs.append({
                    "iter": i,
                    "run_name": run_name,
                    "status": iter_status,
                    "score": float(iter_score),
                    "trial_dir": iter_trial_dir,
                    "duration_sec": float(time.time() - iter_started),
                    "criteria_met": criteria_met,
                    "usage": usage_stats, # [TELEM] Save detailed usage
                })

                # Persist loop summary continuously (not just at end)
                _write_summary(final=False)

                if success and iter_trial_dir:
                    print(f"\n🎉 [SUCCESS] Criteria Met! Archiving and stopping.", flush=True)
                    archived_dir = _archive_run(iter_trial_dir)
                    _write_summary(final=True)
                    return

                print("⚠️ [CONTINUE] Threshold not met.", flush=True)

            except KeyboardInterrupt:
                print("\n[INTERRUPT] KeyboardInterrupt received. Saving loop summary and exiting.", flush=True)
                # Capture partial usage
                usage_stats = TokenMeter.get_and_reset()
                iter_logs.append({
                    "iter": i,
                    "run_name": run_name,
                    "status": "INTERRUPTED",
                    "score": float(iter_score),
                    "trial_dir": iter_trial_dir,
                    "duration_sec": float(time.time() - iter_started),
                    "criteria_met": False,
                    "usage": usage_stats,
                })
                _write_summary(final=True)
                raise

            except Exception as e:
                print(f"❌ [ERROR] Iteration {i} crashed: {e}", flush=True)
                import traceback
                traceback.print_exc()

                # Capture partial usage
                usage_stats = TokenMeter.get_and_reset()
                iter_logs.append({
                    "iter": i,
                    "run_name": run_name,
                    "status": "CRASH",
                    "score": float(iter_score),
                    "trial_dir": iter_trial_dir,
                    "duration_sec": float(time.time() - iter_started),
                    "criteria_met": False,
                    "error": str(e),
                    "usage": usage_stats,
                })
                _write_summary(final=False)

        print(f"\n{'='*40}\n🏁 LOOP FINISHED (No immediate success)\n{'='*40}", flush=True)

        # Archive BEST run found (keeps time-stamped prompt_run_* behavior)
        if best_trial_dir and os.path.exists(best_trial_dir):
            print(f"[LOOP] Archiving BEST run found (Score: {best_score:.4f}).", flush=True)
            archived_dir = _archive_run(best_trial_dir)
        else:
            print("[LOOP] ❌ No valid runs completed successfully to archive.", flush=True)

        _write_summary(final=True)

    finally:
        # Ensure final summary exists even if unexpected error occurs
        if not os.path.exists(summary_path):
            _write_summary(final=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_common(p):
        p.add_argument("--prompt-file", type=str)
        p.add_argument("--use-idea", action="store_true")

    cmd_run = sub.add_parser("run")
    _add_common(cmd_run)

    cmd_gen = sub.add_parser("generate")
    _add_common(cmd_gen)

    sub.add_parser("execute")
    sub.add_parser("analyze")

    args = parser.parse_args()
    cfg = load_full_config(args.config)
    _inject_api_key(cfg)

    use_idea = getattr(args, "use_idea", False)

    if args.cmd == "run":
        run_loop(cfg, args.prompt_file, use_idea)
    elif args.cmd == "generate":
        _setup_stage1_resources(cfg, use_idea)
        p_path = args.prompt_file or (cfg.get("prompt_branch", {})).get("prompt_file") or "prompts/pipeline_prompt.yaml"
        phase_generate(cfg, p_path)
    elif args.cmd == "execute":
        phase_execute(cfg)
    elif args.cmd == "analyze":
        phase_analyze(cfg)

if __name__ == "__main__":
    main()