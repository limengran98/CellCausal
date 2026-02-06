#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics extraction + log parsing + scoreboard printing.

Refactored out of run_cellscientist.py.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import get_nested
from .utils import safe_read_json, read_text


# =============================================================================
# Rich (optional)
# =============================================================================


def _maybe_console():
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return None


def print_execution_plan(stage_map: Dict[str, Dict[str, Any]], dataset_name: str, console=None) -> None:
    """Pretty print stage configuration overview (unified pipeline).

    Phase 1 (Design) is intentionally excluded per refactor requirements.
    Displays comprehensive configuration including literature, BioKB, GPU, timeout, cache, etc.
    """
    console = console if console is not None else _maybe_console()
    if not console:
        print(f"Plan for dataset: {dataset_name}")
        return

    from rich.table import Table

    table = Table(title=f"🧬 CellScientist Pipeline Configuration ([bold green]{dataset_name}[/])", 
                  border_style="bright_cyan")
    table.add_column("Stage", style="cyan bold", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Literature", style="blue", justify="center")
    table.add_column("BioKB", style="blue", justify="center")
    table.add_column("GPU", style="green", justify="center")
    table.add_column("Timeout", style="yellow", justify="right")
    table.add_column("Max Iters", style="white", justify="center")
    table.add_column("Cache", style="cyan", justify="center")

    def _pick(*keys):
        for k in keys:
            if k in stage_map and isinstance(stage_map.get(k), dict):
                return stage_map[k]
        return None
    
    def _format_knowledge(cfg: dict, key_path: List[str], top_k_path: List[str]) -> str:
        """Format knowledge source status with top-k value."""
        enabled = get_nested(cfg, key_path)
        top_k = get_nested(cfg, top_k_path)
        if enabled:
            return f"✓ ({top_k})" if top_k else "✓"
        return "✗"
    
    def _format_timeout(seconds: Optional[int]) -> str:
        """Format timeout in human-readable form."""
        if not seconds:
            return "-"
        try:
            sec = int(seconds)
            hours = sec / 3600
            if hours >= 1:
                return f"{hours:.0f}h"
            return f"{sec}s"
        except (ValueError, TypeError):
            return str(seconds)

    exp = _pick("Experiment", "Phase 2")
    if exp:
        c2 = exp.get("_loaded_cfg", {})
        p2_model = get_nested(c2, ["llm", "model"]) or "-"
        p2_iters = get_nested(c2, ["experiment", "max_iterations"]) or "-"
        
        # Literature and BioKB
        lit_status = _format_knowledge(c2, ["external_knowledge", "enable_literature"], 
                                      ["external_knowledge", "literature_top_k"])
        biokb_status = _format_knowledge(c2, ["external_knowledge", "enable_biokb"], 
                                        ["external_knowledge", "biokb_top_k"])
        
        # GPU device
        gpu_device = get_nested(c2, ["execution", "gpu_device"])
        if gpu_device is not None:
            gpu_str = f"GPU{gpu_device}"
        else:
            gpu_str = "CPU"
        
        # Timeout
        timeout = get_nested(c2, ["execution", "timeout"]) or get_nested(c2, ["experiment", "timeout"])
        timeout_str = _format_timeout(timeout)
        
        # Cache
        cache_enabled = get_nested(c2, ["llm", "cache_enabled"]) or get_nested(c2, ["cache", "enabled"])
        cache_str = "✓" if cache_enabled else "✗"
        
        table.add_row("Experiment", str(p2_model), lit_status, biokb_status, 
                     gpu_str, timeout_str, str(p2_iters), cache_str)
    else:
        table.add_row("Experiment", "-", "-", "-", "-", "-", "-", "-")

    rev = _pick("Review", "Phase 3")
    if rev:
        c3 = rev.get("_loaded_cfg", {})
        p3_model = get_nested(c3, ["llm", "model"]) or "-"
        p3_iters = get_nested(c3, ["review", "max_iterations"]) or "-"
        
        # Literature and BioKB (typically reused from experiment)
        lit_status = _format_knowledge(c3, ["external_knowledge", "enable_literature"], 
                                      ["external_knowledge", "literature_top_k"])
        biokb_status = _format_knowledge(c3, ["external_knowledge", "enable_biokb"], 
                                        ["external_knowledge", "biokb_top_k"])
        
        # GPU device
        gpu_device = get_nested(c3, ["execution", "gpu_device"])
        if gpu_device is not None:
            gpu_str = f"GPU{gpu_device}"
        else:
            gpu_str = "CPU"
        
        # Timeout
        timeout = get_nested(c3, ["execution", "timeout"]) or get_nested(c3, ["review", "timeout"])
        timeout_str = _format_timeout(timeout)
        
        # Cache
        cache_enabled = get_nested(c3, ["llm", "cache_enabled"]) or get_nested(c3, ["cache", "enabled"])
        cache_str = "✓" if cache_enabled else "✗"
        
        table.add_row("Review", str(p3_model), lit_status, biokb_status, 
                     gpu_str, timeout_str, str(p3_iters), cache_str)
    else:
        table.add_row("Review", "-", "-", "-", "-", "-", "-", "-")

    console.print(table)
    console.print("")


# =============================================================================
# Metrics core
# =============================================================================


def extract_primary_score(metrics: Dict[str, Any], metric_key: str) -> Optional[float]:
    if not metrics or not isinstance(metrics, dict):
        return None
    winner = metrics.get("winner")
    models = metrics.get("models") if isinstance(metrics.get("models"), dict) else None

    if not winner:
        if models:
            mk = [k for k in models.keys() if k != "config"]
            winner = mk[0] if mk else None
        else:
            mk = [k for k in metrics.keys() if k not in {"winner", "config", "models", "methods"}]
            winner = mk[0] if mk else None
    if not winner:
        return None

    m_data = metrics.get(winner)
    if not isinstance(m_data, dict) and models:
        m_data = models.get(winner)
    if not isinstance(m_data, dict):
        return None

    if isinstance(m_data.get("aggregate"), dict):
        val = m_data["aggregate"].get(metric_key)
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    pf = m_data.get("per_fold")
    if isinstance(pf, dict):
        vals = []
        for fold in pf.values():
            if isinstance(fold, dict):
                v = fold.get(metric_key)
                if v is None and isinstance(fold.get("metrics"), dict):
                    v = fold["metrics"].get(metric_key)
                try:
                    if v is not None:
                        vals.append(float(v))
                except Exception:
                    continue
        if vals:
            return sum(vals) / float(len(vals))
    return None


def mean_safe(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def find_scores_in_json(obj: Any) -> List[float]:
    out: List[float] = []
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(find_scores_in_json(x))
    elif isinstance(obj, dict):
        if isinstance(obj.get("scores"), list):
            out.extend(find_scores_in_json(obj["scores"]))
        else:
            for v in obj.values():
                out.extend(find_scores_in_json(v))
    return out


def pick_best(scores: List[float], direction: str = "maximize") -> Optional[float]:
    # Allow -999 and inf if that's all we have, otherwise filter
    valid = [s for s in scores if isinstance(s, (int, float))]
    if not valid:
        return None
    
    # Filter out sentinels for calculation if possible
    clean = [s for s in valid if s != -999 and s != float("inf") and s != -float("inf")]
    
    if not clean:
        # If we only have errors/defaults, return the 'best' of the errors
        if direction.lower() == "minimize":
            return min(valid)
        return max(valid)
        
    if direction.lower() == "minimize":
        return min(clean)
    return max(clean)


# =============================================================================
# Stage log parsers
# =============================================================================


def parse_experiment_log(log_text: str, metric: str) -> Dict[str, Any]:
    # [FIX] Markers updated to match execution_workflow.py output
    success_markers = [
        "Success threshold met",
        "Target metric reached",
        "Early stop triggered",
        "Optimization converged",
        "Stopping early",
        "Success!",
        "Criteria Met!", # Matches execution_workflow.py
        "Archiving and stopping"
    ]
    
    # [FIX] Bug markers refined. "Traceback" isn't always a pipeline bug if auto-fixed.
    # We look for "CRASH" or "failed" explicitly.
    bug_markers = [
        "Notebook Generation Failed",
        "LLM_GEN_FAILURE",
        "CRITICAL PARSE FAILURE",
        "Framework Error",
        "Critical Framework Error",
        "[ERROR] Iteration", # Catch explicit crash logs
    ]
    
    iters: Dict[int, Dict[str, Any]] = {}
    cur_iter: Optional[int] = None
    global_early_success = False

    for ln in log_text.splitlines():
        if any(x in ln for x in success_markers):
            global_early_success = True

        m = re.search(r"ITERATION\s+(\d+)/(\d+)", ln)
        if m:
            cur_iter = int(m.group(1))
            iters.setdefault(cur_iter, {"bug": False, "score": None, "explicit_success": False})
            continue

        if cur_iter is None:
            continue
            
        rec = iters.setdefault(cur_iter, {"bug": False, "score": None, "explicit_success": False})

        if any(x in ln for x in bug_markers):
            rec["bug"] = True

        # [FIX] Robust Regex for Experiment Score: [CHECK] <Any text> | <Metric>: <Score>
        # Handles: "[CHECK] Baseline | PCC: 0.1234" and "[CHECK] Winner | PCC: -999.0"
        mm = re.search(r"\[CHECK\].*?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", ln)
        if mm:
            try:
                rec["score"] = float(mm.group(1))
            except Exception:
                pass
        
        if any(x in ln for x in success_markers):
            rec["explicit_success"] = True

    attempted = len(iters)
    scores = [v["score"] for v in iters.values() if isinstance(v.get("score"), (int, float))]
    
    bug = 0
    clean_success = 0
    succeeded = 0

    for idx, v in iters.items():
        is_bug = v["bug"]
        has_score = isinstance(v.get("score"), (int, float))
        is_success_stop = v["explicit_success"] or (global_early_success and idx == max(iters.keys()))
        
        if is_bug:
            bug += 1
        
        # We consider it "succeeded" if we got a score OR explicitly finished
        if has_score or is_success_stop:
            succeeded += 1
            if not is_bug:
                clean_success += 1

    succeeded = min(succeeded, attempted)
    clean_success = min(clean_success, attempted)
    
    return {
        "attempted": attempted, 
        "succeeded": succeeded, 
        "bug": bug, 
        "clean_success": clean_success, 
        "scores": scores
    }


def parse_review_log(log_text: str, metric: str) -> Dict[str, Any]:
    success_markers = [
        "Goal Reached!",
        "Success threshold",
        "Optimization finished early",
        "Success!",
        "NEW BEST!"
    ]
    bug_markers = [
        "LLM Generation Failed",
        "Report generation failed",
        "Errors Found",
        "Final Execution Failed",
        "Logic Error",
        "Traceback",
        "Exception:",
        "[FATAL]",
    ]
    iters: Dict[int, Dict[str, Any]] = {}
    cur_iter: Optional[int] = None
    global_early_success = False

    for ln in log_text.splitlines():
        if any(x in ln for x in success_markers):
            global_early_success = True

        # Matches: "ITERATION 1/3" or "optimization (Iter 1)"
        m = re.search(r"(?:ITERATION|Iter)\s+(\d+)", ln, re.I)
        if m:
            cur_iter = int(m.group(1))
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            continue

        if cur_iter is None:
            continue
            
        if any(x in ln for x in bug_markers):
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            iters[cur_iter]["bug"] = True

        # [FIX] Robust Regex for Review: matches "> Candidate Score: 0.1234"
        # Also supports "Global Best Score: 0.1234"
        ms = re.search(r"(?:Candidate Score|Global Best Score|Score)[:\s]+\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", ln)
        if ms:
            try:
                iters.setdefault(cur_iter, {"bug": False, "score": None})
                # Only update if we don't have a score or this looks like a specific iteration score
                if iters[cur_iter]["score"] is None or "Candidate" in ln:
                    iters[cur_iter]["score"] = float(ms.group(1))
            except Exception:
                pass

    attempted = len(iters)
    # Include -999 in scores list to calculate avg correctly, but filter for best
    scores = [v["score"] for v in iters.values() if isinstance(v.get("score"), (int, float))]
    
    bug = 0
    clean_success = 0
    succeeded = 0
    
    for idx, v in iters.items():
        is_bug = v["bug"]
        # Success if we got ANY score (even -999 means it ran and reported back)
        has_score = (isinstance(v.get("score"), (int, float))) 
        is_success_stop = (global_early_success and idx == max(iters.keys())) if iters else False

        if is_bug:
            bug += 1
        
        if has_score or is_success_stop:
            succeeded += 1
            if not is_bug:
                clean_success += 1
    
    return {
        "attempted": attempted, 
        "succeeded": succeeded, 
        "bug": bug, 
        "clean_success": clean_success, 
        "scores": scores
    }


# =============================================================================
# Artifact-based score extraction
# =============================================================================


def experiment_scores_from_artifacts(ge_dir: str, metric: str, t_start: float, t_end: float) -> List[float]:
    scores: List[float] = []
    
    # Support multiple folder structures (prompt_run or just workspace)
    prompt_root = os.path.join(ge_dir, "prompt")
    search_dirs = [prompt_root, ge_dir]
    
    run_dirs = []
    
    for base in search_dirs:
        if not os.path.exists(base): continue
        # Look for both timestamped folders and "workspace_" folders
        for d in glob.glob(os.path.join(base, "*")):
            if not os.path.isdir(d): continue
            name = os.path.basename(d)
            if name.startswith("prompt_run_") or name.startswith("workspace_"):
                try:
                    mt = os.path.getmtime(d)
                    # Relaxed time window
                    if t_start - 60 <= mt <= t_end + 60:
                        run_dirs.append(d)
                except Exception:
                    continue

    run_dirs.sort(key=lambda p: os.path.getmtime(p))

    for d in run_dirs:
        met_path = os.path.join(d, "metrics.json")
        met = safe_read_json(met_path) or {}
        v = extract_primary_score(met, metric)
        if isinstance(v, (int, float)):
            scores.append(float(v))
    return scores


def review_scores_from_artifacts(rf_dir: str, metric: str, t_start: float, t_end: float) -> List[float]:
    scores: List[float] = []
    best_path = None
    best_mtime = -1.0
    
    if not os.path.exists(rf_dir): return scores

    for name in os.listdir(rf_dir):
        p = os.path.join(rf_dir, name)
        if not os.path.isdir(p) or not name.startswith("review_run_"):
            continue
        try:
            mt = os.path.getmtime(p)
        except Exception:
            continue
        # Find the specific run matching this execution time
        if t_start - 60 <= mt <= t_end + 60 and mt > best_mtime:
            best_mtime = mt
            best_path = p
            
    if not best_path:
        return scores

    # 1. Try history state first (detailed iteration scores)
    hist = safe_read_json(os.path.join(best_path, "history_state.json"))
    if isinstance(hist, list):
        for rec in hist:
            if not isinstance(rec, dict):
                continue
            sc = rec.get("score")
            if isinstance(sc, (int, float)):
                scores.append(float(sc))
        return scores

    # 2. Fallback to single best metric
    mb = safe_read_json(os.path.join(best_path, "metrics_best.json")) or {}
    v = extract_primary_score(mb, metric)
    if isinstance(v, (int, float)):
        scores.append(float(v))
    return scores


def rates(attempted: int, clean_success: int, succeeded: int, bug: int) -> Tuple[float, float, float]:
    """
    Revised logic:
    - Success Rate (total_success_rate): All successful runs / Attempted (Includes Auto-Fixed).
    - Clean Rate (zero_shot_sr): Runs that succeeded WITHOUT auto-fix / Attempted.
    - Bug Rate: Runs that triggered auto-fix / Attempted.
    """
    if attempted <= 0:
        return 0.0, 0.0, 0.0
    
    total_success_rate = succeeded / float(attempted)
    zero_shot_sr = clean_success / float(attempted)
    bug_rate = bug / float(attempted)
    
    return total_success_rate, zero_shot_sr, bug_rate


def print_final_scoreboard(summary: Dict[str, Any], console=None) -> None:
    """Print final success rate scoreboard with enhanced formatting."""
    stages = summary.get("stages", {})
    console = console if console is not None else _maybe_console()
    if console:
        from rich.table import Table

        table = Table(title=f"📊 Success Rate Scoreboard ([bold]{summary.get('dataset')}[/])", 
                     border_style="bright_green")
        table.add_column("Stage", style="bold")
        table.add_column("Success ↑", justify="right", style="green")
        table.add_column("Best@Budget", justify="right", style="bright_cyan bold")
        table.add_column("Time (s)", justify="right", style="yellow")
        table.add_column("Token Cost", justify="right", style="magenta")

        for stage_name in ["Experiment", "Review", "Total"]:
            row = stages.get(stage_name, {})
            
            # Success rate with attempted count
            sr = row.get("success_rate")
            attempted = row.get("attempted") or 0
            succeeded = row.get("succeeded") or 0
            if stage_name == "Total":
                sr_s = "━━━━━━"
            else:
                sr_pct = f"{sr*100:.0f}%" if isinstance(sr, (int, float)) else "-"
                sr_s = f"{sr_pct} ({succeeded}/{attempted})"
            
            # Best score
            best = row.get("best_at_budget")
            metric = str(row.get("best_metric", ""))
            if stage_name == "Total":
                # For total row, show overall best from both stages
                exp_best = stages.get("Experiment", {}).get("best_at_budget")
                rev_best = stages.get("Review", {}).get("best_at_budget")
                candidates = [x for x in [exp_best, rev_best] if isinstance(x, (int, float))]
                if candidates:
                    best = max(candidates)
                    best_s = f"[bold]{best:.4f}[/bold]" if isinstance(best, (int, float)) else "-"
                else:
                    best_s = "-"
            else:
                best_s = f"{metric}={best:.4f}" if isinstance(best, (int, float)) and metric else "-"
            
            # Time
            tsec = row.get("time_sec")
            if stage_name == "Total":
                tsec_s = f"[bold]{tsec:.1f}[/bold]" if isinstance(tsec, (int, float)) else "-"
            else:
                tsec_s = f"{tsec:.1f}" if isinstance(tsec, (int, float)) else "-"
            
            # Token cost placeholder - will be integrated with TokenMeter in future
            # when LLM client metrics are passed to pipeline summary
            token_cost = "-"
            
            # Style Total row differently
            if stage_name == "Total":
                table.add_row(f"[bold]{stage_name}[/bold]", sr_s, best_s, tsec_s, f"[bold]{token_cost}[/bold]")
            else:
                table.add_row(stage_name, sr_s, best_s, tsec_s, token_cost)

        console.print("")
        console.print(table)
    else:
        print("\n=== Scoreboard ===")
        for stage_name in ["Experiment", "Review", "Total"]:
            row = stages.get(stage_name, {})
            print(
                f"{stage_name}: Success={row.get('success_rate')}, ZeroShot={row.get('clean_rate')}, "
                f"BugRate={row.get('bug_rate')}, Avg={row.get('avg_at_budget')}, Best={row.get('best_at_budget')}, "
                f"Metric={row.get('best_metric')}, Time={row.get('time_sec')}"
            )