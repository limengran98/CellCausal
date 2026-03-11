#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import re
import glob
import shutil
import nbformat
import numpy as np
import time
from copy import deepcopy
from datetime import datetime, timezone

# =============================================================================
# [ARCH UPGRADE] Import Centralized Utilities
# =============================================================================
from cellscientist.core.config_loader import load_full_config
# [UPDATED] Import TokenMeter here
from cellscientist.core.llm_client import chat_json, chat_text, extract_json_from_text, resolve_llm_config, TokenMeter
from cellscientist.core.external_knowledge_mirothink import retrieve_external_knowledge, knowledge_pack_to_markdown
from cellscientist.core.task_logger import get_task_logger

# [UPDATED] Import decoupled execution and error handling functions
from cellscientist.core.executor_engine import run_notebook_pure, attempt_fix_notebook, dump_error_log
from cellscientist.pipeline.utils import project_root, resolve_h5_path_unified

from cellscientist.core.task_graph import (
    init_task_graph_from_config,
    load_task_graph,
    save_task_graph,
    apply_decomposition_updates,
    route_active_tasks,
    update_after_iteration,
    to_prompt_summary,
)

try:
    from cellscientist.core.experiment_report import write_experiment_report
except ImportError:
    print("[WARN] 'experiment_report.py' not found. Report generation will be disabled.")
    write_experiment_report = None

# =============================================================================
# Unified Logging Helper for Subprocess Module
# =============================================================================

def _log(msg: str, *, console: bool = False):
    """Unified logging output for subprocess execution.
    
    All messages go through print (captured by parent's run_cmd_streamed).
    - If console=True: Adds [CELL_CONSOLE] prefix → shown in console + all logs
    - If console=False: Adds [DETAIL] prefix → only in detail logs, not console
    
    Args:
        msg: Message to log
        console: If True, message appears in console. If False, only in detail logs.
    """
    summary_only = str(os.environ.get("CELL_SUMMARY_ONLY", "0")).lower() in {"1", "true", "yes"}
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    elif not summary_only:
        print(f"[DETAIL] {msg}", flush=True)

# =============================================================================
# 1. Helper: Resource Resolver (Legacy Wrapper)
# =============================================================================

def _resolve_h5_path(cfg):
    """
    Resolves the absolute path of the STAGE 1 H5 Data.
    Delegates to the unified utility in pipeline/utils.py to avoid redundancy.
    """
    return resolve_h5_path_unified(cfg)


def _inject_llm_env(cfg: dict) -> None:
    """Ensure LLM credentials / base_url are available via env vars."""
    llm = (cfg.get("llm") or {}) if isinstance(cfg.get("llm"), dict) else {}
    
    key = llm.get("api_key")
    if key:
        os.environ["OPENAI_API_KEY"] = str(key)
        
    base = llm.get("base_url")
    if base:
        os.environ["OPENAI_BASE_URL"] = str(base)
    
    # [FIX] Check and Warn if default
    current_base = os.environ.get("OPENAI_BASE_URL", "default")
    if "api.openai.com" in current_base or current_base == "default":
        _log(f"[LLM] ⚠️ WARNING: Using default/OpenAI Base URL ({current_base}).", console=True)
        _log(f"       If you need a custom endpoint (e.g. yi-zhan), ensure 'base_url' is in your config.", console=True)
    else:
        _log(f"[LLM] ✅ Using Custom Base URL: {current_base}", console=True)


def _resolve_relative_resources(cfg):
    """(Legacy wrapper, kept for backward compatibility)"""
    path = _resolve_h5_path(cfg)
    if path:
        os.environ["STAGE1_H5_PATH"] = path

# =============================================================================
# 2. Logging & Analysis Logic (Enhanced for Reflection & Visualization)
# =============================================================================

def _log_tree_visual(workspace, iteration, strategy, decision, focus, status, score):
    """
    Generates, prints, and appends a hierarchical ASCII tree visualization.
    """
    try:
        score_display = f"{score:.4f}" if score != -999 and score != float('inf') else "N/A"
    except Exception:
        score_display = "N/A"
    timestamp = datetime.now().strftime("%H:%M:%S")

    lines = []
    if status == "PENDING":
        lines.append(f"\n╔════════════════════════════════════════════════════════════════╗")
        lines.append(f"║ 🧬 CellScientist Dual-Space Optimization (Iter {iteration})             ║")
        lines.append(f"╚════════════════════════════════════════════════════════════════╝")
        lines.append(f"└── 🌍 [L0: Global Strategy Space]")
        lines.append(f"    │   Target Strategy: {strategy}")

        mode_icon = "🔭" if decision == "EXPLORE" else "🔬"
        lines.append(f"    ├── {mode_icon} [L1: Optimization Mode]: {decision}")

        if decision == "REFINE":
            lines.append(f"    │   └── 🎯 [L2: Sub-Task Focus]: {focus}")
            lines.append(f"    │       └── 🔒 [Constraint Projection]: Locking non-{focus} parameters.")
            lines.append(f"    │       └── 📉 [Alternating Opt]: Computing Semantic Gradient...")
        else:
            lines.append(f"    │   └── 🌐 [L2: Global Instantiation]: Re-initializing Task Hypergraph.")
        lines.append(f"    │")
    else:
        res_icon = "✅" if status == "IMPROVED" else "❌" if status == "CRASH" else "⚠️"
        lines.append(f"    └── 🏁 [Result]: {status} (Score: {score_display}) {res_icon}")
        lines.append(f"        (Time: {timestamp})")

    tree_str = "\n".join(lines)

    # 1. Print to Terminal
    _log(tree_str, console=True)

    # 2. Save to File in Workspace (append)
    try:
        os.makedirs(workspace, exist_ok=True)
        log_file = os.path.join(workspace, "optimization_tree.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(tree_str + "\n")
    except Exception as e:
        _log(f"[WARN] Failed to append optimization tree: {e}", console=False)

def _compute_semantic_gradient(candidate_metrics, baseline_metrics):
    """
    Calculates 'Semantic Gradient' based on metric divergence.
    """
    if not candidate_metrics or not baseline_metrics:
        return "Analysis: Initial run or missing metrics. No gradient available."

    def get_val(m, k):
        try:
            if isinstance(m, dict) and "aggregate" in m and isinstance(m["aggregate"], dict):
                return float(m["aggregate"].get(k, 0) or 0)
            if isinstance(m, dict):
                return float(m.get(k, 0) or 0)
            return 0.0
        except Exception:
            return 0.0

    cand_pcc = get_val(candidate_metrics, "PCC")
    cand_mse = get_val(candidate_metrics, "MSE")
    cand_deg = get_val(candidate_metrics, "DEG_RMSE_50") or get_val(candidate_metrics, "DEG_RMSE_20")

    base_pcc = get_val(baseline_metrics, "PCC")
    base_mse = get_val(baseline_metrics, "MSE")
    base_deg = get_val(baseline_metrics, "DEG_RMSE_50") or get_val(baseline_metrics, "DEG_RMSE_20")

    feedback = []

    if cand_pcc > base_pcc and cand_mse > base_mse:
        feedback.append(
            "⚠️ **Gradient Alert (Scale Mismatch)**: PCC improved (Ranking is better), but MSE worsened. "
            "Model captures the trend but has scaling errors. "
            "-> **Constraint**: Fix output scaling/bias. Do NOT change architecture."
        )
    elif cand_mse < base_mse and cand_pcc < base_pcc:
        feedback.append(
            "⚠️ **Gradient Alert (Conservative Fit)**: MSE improved, but PCC worsened. "
            "Model is predicting mean values, losing biological variance. "
            "-> **Constraint**: Switch to Rank-aware Loss (RankNet/Contrastive)."
        )

    if cand_deg > base_deg and base_deg > 0:
        feedback.append(
            f"⚠️ **Mechanism Failure**: Error on Top Diff-Expressed Genes increased ({cand_deg:.4f}). "
            "Model ignores critical biological signals. "
            "-> **Correction**: Increase weight on high-variance genes."
        )

    if not feedback:
        feedback.append("✅ **Gradient Stable**: Balanced improvement. -> **Strategy**: Continue fine-tuning.")

    return "\n".join(feedback)

def write_review_log(workspace, iteration, suggestion, score, current_best, static_baseline, status):
    """
    Writes a log comparing Candidate vs Best vs Baseline.
    Records Strategy, Reflection, Decision Type, Focus Area, and Semantic Gradient.
    """
    log_path = os.path.join(workspace, "optimization_history.md")
    timestamp = datetime.now().strftime("%H:%M:%S")

    reflection = suggestion.get("reflection_on_history", "No reflection provided.")
    strategy = suggestion.get("selected_strategy", "Unknown Strategy")
    critique = suggestion.get("critique", "No critique provided.")
    edits = suggestion.get("edits", [])

    decision = suggestion.get("decision_type", "EXPLORE")
    focus = suggestion.get("focus_area", "All")
    sem_grad = suggestion.get("semantic_gradient_analysis", "N/A")
    active_tasks = suggestion.get("active_tasks", [])

    icon = "✅" if status == "IMPROVED" else "❌" if status == "FAILED" else "⚠️"

    delta_str = "-"
    if static_baseline is not None and score != -999 and score != float('inf'):
        try:
            diff = float(score) - float(static_baseline)
            delta_str = f"{diff:+.4f}"
        except Exception:
            delta_str = "-"

    try:
        cand_score_disp = f"{float(score):.4f}" if score != -999 and score != float('inf') else "N/A"
    except Exception:
        cand_score_disp = "N/A"

    log_entry = f"""
## Iteration {iteration} {icon} (Time: {timestamp})
**Status**: {status} 
**Action**: `{decision}` | **Focus**: `{focus}`
**Active Tasks**: {", ".join([t.get("name", t.get("id","?")) for t in active_tasks]) if active_tasks else "N/A"}
**Strategy**: {strategy}

| Metric Type | Score |
| :--- | :--- |
| **Candidate (This Run)** | **{cand_score_disp}** |
| Current Best (Evolution) | {current_best:.4f} |
| Original Baseline | {static_baseline if static_baseline is not None else 'N/A'} |
| **Delta (vs Baseline)** | **{delta_str}** |

### 📉 Semantic Gradient Analysis
> {sem_grad}

### 🧠 Reflection on History
> {reflection}

### 💡 Rationale (Critique)
> {critique}

### 🛠 Code Changes
"""
    if not edits:
        log_entry += "\n*No code edits applied.*\n"
    else:
        for edit in edits:
            idx = edit.get("cell_index")
            src = edit.get("source", "")
            log_entry += f"\n**Cell {idx}**:\n```python\n{src}\n```\n"

    log_entry += "\n---\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        _log(f"[LOG] Review log updated: {log_path}", console=False)
    except Exception as e:
        _log(f"[WARN] Failed to write review log: {e}", console=False)

# =============================================================================
# 3. Path & Resource Management
# =============================================================================

def _get_paths(cfg: dict) -> dict:
    return cfg.get("paths", {}) or {}

def find_best_experiment_trial(cfg, explicit_path=None):
    """
    Finds the source trial directory.
    """
    # 1. Check argument explicit path
    if explicit_path:
        abs_path = os.path.abspath(explicit_path) if not os.path.isabs(explicit_path) else explicit_path
        _log(f"[INIT] Checking specified path: {abs_path}", console=False)
        
        # Check if this IS the run directory
        if os.path.exists(os.path.join(abs_path, "metrics.json")):
             _log(f"[INIT] Locked to specified source (Direct): {abs_path}", console=False)
             return abs_path

    # 2. Check config explicit path (bridge from run_cellscientist.py)
    paths = _get_paths(cfg)
    cfg_explicit = paths.get("explicit_source_path")
    
    if not cfg_explicit:
        candidates = [paths.get("generate_execution_dir"), paths.get("execution_dir")]
        for c in candidates:
            if c and os.path.exists(os.path.join(c, "metrics.json")):
                cfg_explicit = c
                break

    if cfg_explicit:
        abs_path = os.path.abspath(cfg_explicit)
        if os.path.exists(os.path.join(abs_path, "metrics.json")):
            _log(f"[INIT] Locked to config explicit source: {abs_path}", console=False)
            return abs_path

    # 3. Fallback: Search in default locations (Auto-detection)
    gen_root = paths.get("generate_execution_root") or paths.get("generate_execution_dir") or paths.get("generate_execution") \
               or os.path.join("results", cfg.get("dataset_name", "dataset"), "generate_execution")
    
    if os.path.exists(os.path.join(gen_root, "metrics.json")):
        return gen_root

    # [FIX] Look for both flattened workspaces and legacy prompt folder
    prompt_root = os.path.join(os.path.abspath(gen_root), "prompt")
    
    search_paths = []
    if os.path.exists(os.path.abspath(gen_root)):
        search_paths.extend(glob.glob(os.path.join(os.path.abspath(gen_root), "workspace_*")))
        search_paths.extend(glob.glob(os.path.join(os.path.abspath(gen_root), "prompt_run_*")))
    
    if os.path.exists(prompt_root):
        search_paths.extend(glob.glob(os.path.join(prompt_root, "prompt_run_*")))
        search_paths.extend(glob.glob(os.path.join(prompt_root, "workspace_*")))

    runs = sorted(list(set(search_paths)), reverse=True)

    _log(f"[INIT] Searching for Experiment stage results in: {gen_root}", console=False)
    for run in runs:
        nb_path = os.path.join(run, "notebook_prompt_exec.ipynb")
        metrics_path = os.path.join(run, "metrics.json")
        if os.path.exists(nb_path) and os.path.exists(metrics_path):
            _log(f"[INIT] Found valid Experiment source (Auto-detected): {run}", console=False)
            return run

    raise FileNotFoundError("No valid Experiment execution results found (Auto-detection failed).")

def setup_phase3_workspace(cfg, source_trial_path):
    # [FIX] Add PID to prevent concurrency collision
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    
    paths = _get_paths(cfg)
    out_root = paths.get("review_feedback_root") or paths.get("review_feedback_dir") \
               or os.path.join("results", cfg.get("dataset_name", "dataset"), "review_feedback")
    out_root = os.path.abspath(out_root)

    workspace = os.path.join(out_root, f"review_run_{ts}_{pid}")
    os.makedirs(workspace, exist_ok=True)

    src_nb = os.path.join(source_trial_path, "notebook_prompt_exec.ipynb")
    dst_nb = os.path.join(workspace, "notebook_base.ipynb")
    shutil.copy(src_nb, dst_nb)

    src_met = os.path.join(source_trial_path, "metrics.json")
    shutil.copy(src_met, os.path.join(workspace, "metrics_base.json"))

    with open(os.path.join(workspace, "optimization_history.md"), "w", encoding="utf-8") as f:
        f.write(f"# Optimization History\nSource Trial: {source_trial_path}\n\n")

    return workspace, dst_nb

# =============================================================================
# 4. Notebook Analysis
# =============================================================================

def identify_mutable_cells(nb, cfg):
    """
    Identifies mutable cells, explicitly supporting Strategy (Markdown) + Model (Code).
    """
    force_indices = (cfg.get("review", {}) or {}).get("force_cells")
    if force_indices and isinstance(force_indices, list) and len(force_indices) > 0:
        valid_indices = [i for i in force_indices if 0 <= i < len(nb.cells)]
        return valid_indices

    protected_keywords = (cfg.get("review", {}) or {}).get("protected_sections", [])

    mutable_indices = []

    for i, cell in enumerate(nb.cells):
        src = cell.source

        if "StratifiedKFold" in src or "train_test_split" in src:
            continue

        if cell.cell_type == "markdown":
            if any(k in src for k in ["Strategy", "Idea", "Hypothesis", "Research Plan"]):
                mutable_indices.append(i)
            continue

        if cell.cell_type == "code":
            is_protected = any(k in src for k in protected_keywords)
            if is_protected:
                continue

            is_model = "import torch.nn" in src or ("class " in src and "(nn.Module)" in src)
            is_digest = "Digest" in src and "Strategy" in src

            if is_model or is_digest:
                mutable_indices.append(i)

    _log(f"[ROUTER] Mutable Cell Indices (Auto): {mutable_indices}", console=False)
    return mutable_indices

# =============================================================================
# 5. Metric Extraction
# =============================================================================

def _extract_val_from_model_data(model_data, metric_name):
    """Helper to dig metric from model dict."""
    val = None
    try:
        if isinstance(model_data, dict) and isinstance(model_data.get("aggregate"), dict):
            val = model_data["aggregate"].get(metric_name)
        if val is None and isinstance(model_data, dict) and isinstance(model_data.get("per_fold"), dict):
            vals = []
            for x in model_data["per_fold"].values():
                if isinstance(x, dict) and metric_name in x:
                    try:
                        vals.append(float(x.get(metric_name, 0)))
                    except Exception:
                        continue
            if vals:
                val = float(np.mean(vals))
        return float(val) if val is not None else None
    except Exception:
        return None

def get_candidate_metric_value(metrics_path, metric_name="PCC"):
    if not os.path.exists(metrics_path):
        return -999.0
    try:
        with open(metrics_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", {})
        if not models:
            return -999.0

        non_baseline_keys = [k for k in models.keys() if "baseline" not in k.lower() and "reference" not in k.lower()]
        target_key = non_baseline_keys[-1] if non_baseline_keys else (data.get("winner") or list(models.keys())[0])

        val = _extract_val_from_model_data(models.get(target_key, {}), metric_name)
        return val if val is not None else -999.0
    except Exception:
        return -999.0

def get_baseline_metric_value(metrics_path, metric_name="PCC"):
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", {})
        baseline_keys = [k for k in models.keys() if "baseline" in k.lower() or "reference" in k.lower()]
        if baseline_keys:
            return _extract_val_from_model_data(models.get(baseline_keys[0], {}), metric_name)
        return None
    except Exception:
        return None

# =============================================================================
# 6. LLM Logic (Enhanced for Context & Memory)
# =============================================================================

def classify_failure_type(history_summary, curr_metrics_obj, last_error=None):
    """Classify the failure type based on history and current state
    
    Returns:
        tuple: (failure_type, query_hint)
        failure_type: F1-F5 classification
        query_hint: Specific query template for this failure type
    """
    if not history_summary:
        return "F0", None  # First iteration, no failure
    
    last_iter = history_summary[-1]
    status = last_iter.get("status", "").upper()
    
    # F1: Execution Error (CRASH)
    if status == "CRASH" or last_error:
        error_msg = str(last_error) if last_error else last_iter.get("critique", "")
        
        # Check for common error types
        if any(kw in error_msg.upper() for kw in ["IMPORTERROR", "MODULENOTFOUND", "IMPORT"]):
            hint = "python ImportError fix module installation package manager"
        elif any(kw in error_msg.upper() for kw in ["KEYERROR", "ATTRIBUTEERROR", "NAMEERROR"]):
            hint = "python KeyError AttributeError debugging variable scope"
        elif any(kw in error_msg.upper() for kw in ["OOM", "OUT OF MEMORY", "MEMORY"]):
            hint = "python out of memory optimization batch size tensor management"
        elif any(kw in error_msg.upper() for kw in ["NAN", "INF", "INVALID"]):
            hint = "python NaN inf numerical stability machine learning"
        elif any(kw in error_msg.upper() for kw in ["SHAPE", "DIMENSION", "SIZE MISMATCH"]):
            hint = "python tensor shape mismatch PyTorch dimension error"
        else:
            hint = f"{error_msg[:100]} fix debugging python issue"
        
        return "F1", hint
    
    # F4: Mechanism Conflict (check if we have BioKB evidence and results contradict)
    # This is a simplified heuristic - in practice would need more sophisticated analysis
    if "mechanism" in last_iter.get("critique", "").lower() or "conflict" in last_iter.get("critique", "").lower():
        return "F4", "perturbation mechanism pathway compensation MAPK PI3K p53 EMT biological validation"
    
    # F3: Direction Error
    score = last_iter.get("score", -999)
    if score != -999 and score != float('inf'):
        sem_grad = last_iter.get("semantic_gradient", "")
        if "direction" in sem_grad.lower() or "inconsistent" in sem_grad.lower():
            return "F3", "perturbation response direction consistency validation biological interpretation"
    
    # F2: Metric Issues (not improving, converging badly)
    if status == "FAILED":
        # Check if metrics are present but poor
        try:
            if curr_metrics_obj:
                return "F2", "cell painting perturbation modeling metric optimization baseline evaluation performance"
        except Exception:
            pass
    
    # F5: Data Issues (check for keywords in history)
    critique = last_iter.get("critique", "")
    if any(kw in critique.lower() for kw in ["batch effect", "leakage", "data split", "distribution", "overfitting"]):
        return "F5", "data preprocessing batch effect normalization cross-validation split strategy"
    
    # Default to F2 if failed but unclear why
    if status == "FAILED":
        return "F2", "cell painting perturbation modeling metric optimization baseline evaluation"
    
    return "F0", None


def _expand_vars(text, context):
    if not text:
        return ""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(context.get(m.group(1), m.group(0))), text)

def generate_optimization_suggestion(cfg, nb, mutable_indices, current_metrics, iteration, best_metric_val, workspace, history_summary, task_graph_state_text=None):
    """
    Generates optimization suggestions.
    Builds a richer history context including strategies, decisions and reflections.
    Implements failure type classification and targeted query hints.
    """
    mutable_content = ""
    immutable_context_content = ""

    for idx, cell in enumerate(nb.cells):
        src = cell.source

        label = "CODE"
        if cell.cell_type == "markdown":
            if any(k in src for k in ["Strategy", "Idea", "Hypothesis", "Research Plan"]):
                label = "STRATEGY_DOC"
            else:
                label = "MARKDOWN_NOTE"
        elif cell.cell_type == "code":
            if "class " in src and "(nn.Module)" in src:
                label = "MODEL_DEF"
            elif "Strategy Digest" in src:
                label = "STRATEGY_DIGEST"
            elif idx < 5 and ("import " in src or "from " in src):
                label = "SETUP/IMPORTS"

        if idx in mutable_indices:
            mutable_content += f"\n# --- CELL INDEX {idx} (TARGET TO OPTIMIZE: {label}) ---\n{src}\n"
        else:
            if cell.cell_type == "code":
                immutable_context_content += f"\n# --- CELL INDEX {idx} (READ-ONLY CONTEXT) ---\n{src}\n"
    
    # Classify failure type for targeted external knowledge retrieval
    failure_type, query_hint = classify_failure_type(history_summary, current_metrics)

    history_text = ""
    if not history_summary:
        history_text = "No previous iterations. This is the first attempt."
    else:
        history_text = "--- PREVIOUS EXPERIMENTS HISTORY (Read Carefully to Avoid Repeats) ---\n"
        for h in history_summary:
            iter_num = h.get('iter')
            strat = h.get('strategy', 'Unknown')
            decision = h.get('decision', 'N/A')
            focus = h.get('focus', 'N/A')
            reflect = (h.get('reflection', 'N/A') or 'N/A')[:300].replace('\n', ' ')
            score = h.get('score', -999.0)
            status = h.get('status', 'UNKNOWN')

            try:
                score_disp = f"{float(score):.4f}" if score != -999 and score != float('inf') else "N/A"
            except Exception:
                score_disp = "N/A"

            history_text += f"\n[Iteration {iter_num}]\n"
            history_text += f"  - Strategy: {strat}\n"
            history_text += f"  - Action: {decision} (Focus: {focus})\n"
            history_text += f"  - Score: {score_disp} ({status})\n"
            history_text += f"  - Reflection: {reflect}...\n"

    models_data = current_metrics.get("models", {}) if isinstance(current_metrics, dict) else {}
    baseline_key = next((k for k in models_data.keys() if "baseline" in k.lower()), None)
    best_key = next((k for k in models_data.keys() if "baseline" not in k.lower()), None)

    base_metrics = models_data.get(baseline_key, {}) if baseline_key else {}
    cand_metrics = models_data.get(best_key, {}) if best_key else {}

    semantic_gradient_text = _compute_semantic_gradient(cand_metrics, base_metrics)

    # Resolve prompt path robustly (works regardless of current working directory)
    candidates = [
        os.path.join(os.getcwd(), "prompts", "review_optimize.yaml"),
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "prompts", "review_optimize.yaml"),
        os.path.join(os.path.dirname(__file__), "prompts", "review_optimize.yaml"),  # legacy fallback
    ]
    prompt_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    sys_prompt = "You are a code optimizer. Return JSON with 'edits' key."
    user_prompt = f"Optimize this code:\n{mutable_content}"

    if os.path.exists(prompt_path):
        try:
            import yaml
            with open(prompt_path, 'r', encoding='utf-8') as f:
                p_data = yaml.safe_load(f) or {}

            ctx = {
                "target_metric": (cfg.get("review", {}) or {}).get("target_metric"),
                "current_best": str(best_metric_val),
                "iteration": str(iteration),
                "mutable_cells_content": mutable_content,
                "immutable_context_content": immutable_context_content,
                "metrics_json": json.dumps(current_metrics, indent=2, ensure_ascii=False),
                "history_summary": history_text,
                "experiment_history": history_text,
                "semantic_gradient_analysis": semantic_gradient_text,
                "task_graph_state": task_graph_state_text or "{}"
            }

            # [NEW] External knowledge retrieval (optional)
            # Uses MiroThink-derived Serper search + Jina Reader scraping. Controlled by cfg["literature"]["enabled"].
            # Pass query_hint based on failure type classification
            if failure_type != "F0" and query_hint:
                _log(f"[REVIEW] Failure type: {failure_type}, using targeted query hint", console=True)
            
            try:
                pack = retrieve_external_knowledge(
                    cfg=cfg,
                    context_text=(immutable_context_content or "") + "\n\n" + (mutable_content or ""),
                    stage="review",
                    query_hint=query_hint,  # Pass the failure-specific query hint
                    workspace_dir=workspace,
                    tag=f"iter_{iteration}",
                )
                ctx["external_knowledge_md"] = knowledge_pack_to_markdown(
                    pack,
                    max_chars=int(((cfg.get("literature") or {}) if isinstance(cfg.get("literature"), dict) else {}).get("prompt_max_chars", 6000) or 6000),
                )
            except Exception as e:
                _log(f"[REVIEW][LIT][WARN] External knowledge retrieval failed: {e}", console=False)
                ctx["external_knowledge_md"] = ""

            if "system" in p_data:
                sys_prompt = _expand_vars(p_data["system"], ctx)
            if "user_template" in p_data:
                user_prompt = _expand_vars(p_data["user_template"], ctx)

            if "${semantic_gradient_analysis}" not in (p_data.get("user_template", "") or ""):
                user_prompt += f"\n\n**DATA FEEDBACK (SEMANTIC GRADIENT)**:\n{semantic_gradient_text}"

        except Exception as e:
            _log(f"[WARN] Failed to load prompt yaml: {e}", console=False)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # [NEW] Snapshot the prompts for traceability
    try:
        debug_iter_dir = os.path.join(workspace, "intermediate", "debug_review", f"iter_{iteration}")
        os.makedirs(debug_iter_dir, exist_ok=True)
        with open(os.path.join(debug_iter_dir, "review_prompt_system.txt"), "w", encoding="utf-8") as f:
            f.write(sys_prompt)
        with open(os.path.join(debug_iter_dir, "review_prompt_user.txt"), "w", encoding="utf-8") as f:
            f.write(user_prompt)
        _log(f"[REVIEW] 🧾 Saved prompt snapshot to: {debug_iter_dir}", console=True)
    except Exception as e:
        _log(f"[REVIEW][WARN] Failed to save prompt snapshot: {e}", console=False)

    _log(f"[REVIEW] Requesting optimization (Iter {iteration})...", console=False)

    llm_cfg = (cfg.get("llm", {}) or {}) if isinstance(cfg.get("llm", {}), dict) else {}
    # Review prompts can be very long (mutable+immutable cells + metrics + history + RAG).
    # Use a larger default than chat_json's 4096 to reduce truncation-induced JSON parse failures.
    review_max_tokens = int(llm_cfg.get("review_max_tokens") or llm_cfg.get("max_tokens") or 12000)
    review_max_tokens = max(4096, review_max_tokens)

    try:
        return chat_json(
            messages,
            cfg=cfg,
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=review_max_tokens,
            max_retries=int(llm_cfg.get("json_max_retries") or 4),
        )
    except Exception as e:
        _log(
            f"[REVIEW] chat_json failed (max_tokens={review_max_tokens}); attempting text fallback parse: {e}",
            console=False,
        )

    try:
        fallback_text = chat_text(
            messages,
            cfg=cfg,
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=review_max_tokens,
        )
        parsed = extract_json_from_text(fallback_text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"edits": parsed}
        return None
    except Exception as e:
        _log(f"[REVIEW] chat_json failed, attempting text fallback parse: {e}", console=False)

    try:
        fallback_text = chat_text(
            messages,
            cfg=cfg,
            temperature=(cfg.get("llm", {}) or {}).get("temperature", 0.7),
        )
        parsed = extract_json_from_text(fallback_text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"edits": parsed}
        return None
    except Exception as e:
        _log(f"[REVIEW] LLM Interaction Failed: {e}", console=False)
        return None

# =============================================================================
# 7. Execution Loop Logic
# =============================================================================

def execute_and_recover(nb_path, workdir, cfg, mutable_indices=None, extra_env=None):
    """
    Manages the Execution -> Error Reporting -> Fix Loop lifecycle.
    Accepts mutable_indices (locks) and extra_env (paths).
    Note: LLM calls inside attempt_fix_notebook will also be metered by TokenMeter.
    """
    timeout = (cfg.get("exec", {}) or {}).get("timeout_seconds", 3600)
    max_fixes = (cfg.get("exec", {}) or {}).get("max_fix_rounds", 1)

    current_nb_path = nb_path

    nb = nbformat.read(current_nb_path, as_version=4)
    _log(f"[EXEC] Running Notebook: {current_nb_path}", console=True)

    executed_nb, errors = run_notebook_pure(
        nb,
        workdir,
        timeout,
        cuda_device_id=None,
        extra_env=extra_env
    )

    if not errors:
        out_path = current_nb_path.replace(".ipynb", "_exec.ipynb")
        nbformat.write(executed_nb, out_path)
        return out_path, True

    fix_round = 0
    current_nb_obj = executed_nb

    while errors and fix_round < max_fixes:
        fix_round += 1
        _log(f"\n{'!'*40}", console=True)
        _log(f"[EXEC] Errors Found (Round {fix_round}) - Initiating Recovery...", console=True)

        dump_error_log(workdir, errors, round_idx=fix_round)

        fixed_nb, changed, method = attempt_fix_notebook(
            current_nb_obj,
            errors,
            cfg,
            mutable_indices=mutable_indices
        )

        if not changed:
            _log(f"[EXEC] Could not generate valid fix ({method}). Stopping.", console=True)
            break

        _log(f"[EXEC] Applying Fix ({method}) -> Rerunning...", console=True)

        current_nb_obj, errors = run_notebook_pure(
            fixed_nb,
            workdir,
            timeout,
            cuda_device_id=None,
            extra_env=extra_env
        )

        if not errors:
            _log(f"[EXEC] Fixed successfully!", console=True)
            break

    final_out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    nbformat.write(current_nb_obj, final_out_path)

    if errors:
        _log(f"[EXEC] Final Execution Failed. Remaining Errors: {len(errors)}", console=True)
        dump_error_log(workdir, errors, round_idx=999)
        return final_out_path, False

    return final_out_path, True

# =============================================================================
# 8. Main Optimization Loop
# =============================================================================

def optimize_loop(cfg, workspace_dir, base_nb_path):
    # Resolve H5 Path & Prepare Env
    h5_path = _resolve_h5_path(cfg)
    exec_env = {}
    if h5_path:
        exec_env["STAGE1_H5_PATH"] = h5_path
    else:
        _log("[ERROR] STAGE1_H5_PATH could not be resolved! Notebook execution may fail.", console=True)

    review_cfg = cfg.get("review", {}) or {}
    target_metric = review_cfg.get("target_metric", "PCC")

    # [TRACE] Structured task trace (causal traceability)
    tlog = get_task_logger(workspace_dir)
    try:
        tlog.log_step('review.start', 'Begin review/optimization loop', workspace_dir=os.path.abspath(workspace_dir), base_notebook=os.path.abspath(base_nb_path), target_metric=target_metric)
        tlog.log_artifact('notebook_base', base_nb_path, 'Baseline notebook')
    except Exception:
        pass
    
    # [FIX] Read direction
    direction = review_cfg.get("direction", "maximize").lower() 
    
    threshold = float(review_cfg.get("pass_threshold", review_cfg.get("success_threshold", 0.0) or 0.0))
    max_iters = int(review_cfg.get("max_iterations", 3) or 3)

    # [FIX] Define WORST_SCORE and is_better helper
    WORST_SCORE = float('inf') if direction == "minimize" else -999.0

    def is_better(new_val, old_val):
        if new_val == -999.0 or new_val == float('inf'): 
            return False
        if old_val == WORST_SCORE: 
            return True
        if direction == "minimize":
            return new_val < old_val
        return new_val > old_val

    # Initialization
    current_metrics_path = os.path.join(workspace_dir, "metrics_base.json")
    
    # [FIX] Logic for initial score
    init_val = get_candidate_metric_value(current_metrics_path, target_metric)
    
    if direction == "minimize" and init_val == -999.0:
        best_score_so_far = float('inf')
    else:
        best_score_so_far = init_val

    static_baseline_score = get_baseline_metric_value(current_metrics_path, target_metric)
    if static_baseline_score is None:
        static_baseline_score = best_score_so_far

    best_nb_path = base_nb_path

    # History Persistence: Load state if exists
    history_state_path = os.path.join(workspace_dir, "history_state.json")
    history_summary = []

    if os.path.exists(history_state_path):
        try:
            with open(history_state_path, 'r', encoding="utf-8") as f:
                history_summary = json.load(f)
            if not isinstance(history_summary, list):
                history_summary = []
            _log(f"[INIT] Loaded {len(history_summary)} history records from state file.", console=True)
        except Exception as e:
            _log(f"[WARN] Failed to load history state: {e}", console=False)

    # Task-Graph State
    task_graph_state_path = os.path.join(workspace_dir, "task_graph_state.json")
    if os.path.exists(task_graph_state_path):
        try:
            task_graph_state = load_task_graph(task_graph_state_path)
            _log(f"[INIT] Loaded task graph with {len(task_graph_state.get('tasks', {}))} tasks.", console=True)
        except Exception as e:
            _log(f"[WARN] Failed to load task graph state: {e}. Re-initializing.", console=False)
            task_graph_state = init_task_graph_from_config(review_cfg.get("optimization_hierarchy", []))
    else:
        task_graph_state = init_task_graph_from_config(review_cfg.get("optimization_hierarchy", []))

    try:
        save_task_graph(task_graph_state, task_graph_state_path)
    except Exception as e:
        _log(f"[WARN] Failed to save task graph state: {e}", console=False)

    baseline_display = f"{static_baseline_score:.4f}" if static_baseline_score is not None else "N/A"

    _log(f"\n[LOOP] Starting Optimization.", console=True)
    _log(f"       🎯 Target: {target_metric} ({direction})", console=True)
    _log(f"       🏁 Original Baseline: {baseline_display}", console=True)
    _log(f"       🥇 Current Best:      {best_score_so_far:.4f}", console=True)
    
    # [TELEM] Reset Meter Global
    TokenMeter.get_and_reset()

    try:
        for i in range(1, max_iters + 1):
            # [TELEM] Reset Meter for this iteration
            TokenMeter.get_and_reset()
            iter_started = time.time()
            try:
                tlog.log_step('review.iter_start', 'Iteration start', iteration=i, best_score=float(best_score_so_far), target_metric=str(target_metric))
            except Exception:
                pass

            nb = nbformat.read(best_nb_path, as_version=4)
            mutable_indices = identify_mutable_cells(nb, cfg)
            if not mutable_indices:
                _log("[ERROR] No mutable cells found.", console=True)
                break

            try:
                with open(current_metrics_path, 'r', encoding="utf-8") as f:
                    curr_metrics_obj = json.load(f)
            except Exception:
                curr_metrics_obj = {}

            task_graph_state_text = to_prompt_summary(task_graph_state)

            suggestion = generate_optimization_suggestion(
                cfg, nb, mutable_indices, curr_metrics_obj, i, best_score_so_far, workspace_dir, history_summary,
                task_graph_state_text=task_graph_state_text
            )

            if not suggestion or "edits" not in suggestion:
                _log("[WARN] Invalid LLM response. Skipping.", console=False)
                continue
            
            # [NEW] Validate used_evidence_ids (mandatory tracking)
            # Allow empty list if no external knowledge was consulted
            if "used_evidence_ids" not in suggestion or not isinstance(suggestion.get("used_evidence_ids"), list):
                _log("[WARN] Missing or invalid 'used_evidence_ids' field. Skipping this iteration for evidence traceability.", console=False)
                # Save incomplete suggestion for debugging
                incomplete_path = os.path.join(workspace_dir, f"suggestion_iter_{i}_incomplete.json")
                try:
                    with open(incomplete_path, 'w', encoding='utf-8') as f:
                        json.dump(suggestion, f, indent=2, ensure_ascii=False)
                    _log(f"[WARN] Incomplete suggestion saved to: {incomplete_path}", console=False)
                except Exception:
                    pass
                continue
            
            used_evidence_ids = suggestion.get("used_evidence_ids", [])
            if used_evidence_ids:
                _log(f"[REVIEW] ✅ Evidence IDs used: {used_evidence_ids}", console=True)
            else:
                _log(f"[REVIEW] ⚠️ No external evidence consulted (empty evidence list)", console=True)
            suggestion["used_evidence_ids"] = used_evidence_ids  # Ensure it's in the saved suggestion

            # [TRACE] Persist suggestion for causal traceability
            suggestion_path = os.path.join(workspace_dir, f"suggestion_iter_{i}.json")
            try:
                with open(suggestion_path, 'w', encoding='utf-8') as f:
                    json.dump(suggestion, f, indent=2, ensure_ascii=False)
                tlog.log_artifact('suggestion', suggestion_path, 'LLM suggestion (edits/strategy)')
            except Exception:
                pass

            # [TRACE] External knowledge artifacts (latest pointers)
            lit_dir = os.path.join(workspace_dir, 'external_knowledge')
            lit_md = os.path.join(lit_dir, 'external_knowledge_review.md')
            lit_json = os.path.join(lit_dir, 'external_knowledge_review.json')
            try:
                if os.path.exists(lit_md):
                    tlog.log_artifact('external_knowledge_md', lit_md, 'External knowledge (review)')
                if os.path.exists(lit_json):
                    tlog.log_artifact('external_knowledge_json', lit_json, 'External knowledge (review)')
            except Exception:
                pass

            # Apply decomposition updates (compat with different signatures)
            try:
                apply_decomposition_updates(task_graph_state, suggestion)
            except TypeError:
                try:
                    updates = suggestion.get("updates", [])
                    if updates:
                        task_graph_state = apply_decomposition_updates(task_graph_state, updates)
                except Exception as e:
                    _log(f"[WARN] Failed to apply decomposition updates: {e}", console=False)
            except Exception as e:
                _log(f"[WARN] Failed to apply decomposition updates: {e}", console=False)

            # Route active tasks (compat)
            try:
                active_task_ids = route_active_tasks(task_graph_state, suggestion)
            except TypeError:
                try:
                    active_task_ids = route_active_tasks(task_graph_state)
                except Exception:
                    active_task_ids = []
            except Exception:
                active_task_ids = []

            active_task_names = []
            for tid in active_task_ids or []:
                t = (task_graph_state.get("tasks", {}) or {}).get(tid, {})
                active_task_names.append(t.get("name", tid))
            suggestion["active_tasks"] = [{"id": tid, "name": name} for tid, name in zip(active_task_ids or [], active_task_names)]

            critique = suggestion.get("critique", "")
            reflection = suggestion.get("reflection_on_history", "No reflection.")
            strategy_tag = suggestion.get("selected_strategy", "Unknown")

            decision_tag = suggestion.get("decision_type", "EXPLORE")
            focus_tag = suggestion.get("focus_area", "All")
            sem_grad = suggestion.get("semantic_gradient_analysis", "N/A")

            _log_tree_visual(workspace_dir, i, strategy_tag, decision_tag, focus_tag, "PENDING", best_score_so_far)

            _log(f"[STRATEGY] {strategy_tag}", console=True)
            _log(f"[ACTION]   {decision_tag} (Focus: {focus_tag})", console=True)
            _log(f"[REFLECTION] {reflection[:100]}...", console=True)
            _log(f"[CRITIQUE] {critique[:100]}...", console=True)

            nb_next = deepcopy(nb)
            applied_count = 0
            for edit in suggestion.get("edits", []):
                try:
                    idx = int(edit.get("cell_index"))
                    if idx in mutable_indices:
                        nb_next.cells[idx].source = edit.get("source", "")
                        applied_count += 1
                except Exception:
                    continue

            if applied_count == 0:
                _log("[WARN] No edits applied.", console=False)
                continue

            candidate_nb_path = os.path.join(workspace_dir, f"notebook_iter_{i}.ipynb")
            nbformat.write(nb_next, candidate_nb_path)

            try:
                tlog.log_artifact('candidate_notebook', candidate_nb_path, f'Candidate notebook for iter {i}')
            except Exception:
                pass

            _log(f"[EXEC] Running Candidate {i}...", console=True)

            executed_nb_path, success = execute_and_recover(
                nb_path=candidate_nb_path,
                workdir=workspace_dir,
                cfg=cfg,
                mutable_indices=mutable_indices,
                extra_env=exec_env
            )
            
            # [TELEM] Capture stats for this iteration
            usage_stats = TokenMeter.get_and_reset()
            usage_stats["duration_sec"] = time.time() - iter_started
            _log(f"[COST] Iteration {i}: {usage_stats['total_tokens']} tokens | {usage_stats['total_latency_sec']:.2f}s LLM time", console=True)

            try:
                iter_metrics_path = os.path.join(workspace_dir, f"metrics_iter_{i}.json")
                generated_metrics = os.path.join(workspace_dir, "metrics.json")
                if os.path.exists(generated_metrics):
                    shutil.copy(generated_metrics, iter_metrics_path)

                try:
                    tlog.log_artifact('iter_metrics', iter_metrics_path, f'Iteration metrics {i}', compute_hash=False)
                except Exception:
                    pass

                candidate_score = get_candidate_metric_value(iter_metrics_path, target_metric)
                
                # [FIX] Handle invalid score for minimize logic
                if direction == "minimize" and candidate_score == -999.0:
                    candidate_score = float('inf')

                current_run_baseline = get_baseline_metric_value(iter_metrics_path, target_metric)
                comparison_baseline = current_run_baseline if current_run_baseline is not None else static_baseline_score

                # [FIX] Use generic is_better comparison
                status = "FAILED"
                if not success:
                    status = "CRASH"
                elif is_better(candidate_score, best_score_so_far):
                    status = "IMPROVED"

                _log(f"-"*40, console=True)
                _log(f"[RESULT] Iteration {i} Summary", console=True)
                _log(f"  > Candidate Score: {candidate_score:.4f}", console=True)
                comp_base_disp = f"{comparison_baseline:.4f}" if comparison_baseline is not None else "N/A"
                _log(f"  > Baseline Score:  {comp_base_disp}", console=True)
                _log(f"  > Best Previous:   {best_score_so_far:.4f}", console=True)
                _log(f"  > Verdict:         {status}", console=True)
                _log(f"-"*40, console=True)

                _log_tree_visual(workspace_dir, i, strategy_tag, decision_tag, focus_tag, status, candidate_score)

                write_review_log(workspace_dir, i, suggestion, candidate_score, best_score_so_far, comparison_baseline, status)
                
                # [NEW] Gate Decision: Persist decision and rollback logic
                gate_decision = {
                    "iteration": i,
                    "primary_metric": target_metric,
                    "direction": direction,
                    "candidate_score": candidate_score,
                    "prev_best_score": best_score_so_far,
                    "baseline_score": comparison_baseline,
                    "status": status,
                    "improved": (status == "IMPROVED"),
                    "reason": "",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                if status == "IMPROVED":
                    gate_decision["reason"] = f"Candidate score {candidate_score:.4f} is better than previous best {best_score_so_far:.4f}"
                    gate_decision["action"] = "ACCEPT"
                elif status == "CRASH":
                    gate_decision["reason"] = "Execution failed with errors"
                    gate_decision["action"] = "REJECT_CRASH"
                else:
                    gate_decision["reason"] = f"Candidate score {candidate_score:.4f} did not improve over previous best {best_score_so_far:.4f}"
                    gate_decision["action"] = "REJECT_NO_IMPROVEMENT"
                
                # Save gate decision
                gate_path = os.path.join(workspace_dir, f"gate_decision_iter_{i}.json")
                try:
                    with open(gate_path, 'w', encoding='utf-8') as f:
                        json.dump(gate_decision, f, indent=2, ensure_ascii=False)
                    _log(f"[GATE] 📋 Decision saved: {gate_path} | Action: {gate_decision['action']}", console=True)
                except Exception as e:
                    _log(f"[WARN] Failed to save gate decision: {e}", console=False)

                # Update task graph with evidence (compat)
                try:
                    update_after_iteration(
                        task_graph_state,
                        iteration=i,
                        active_task_ids=[t.get('id') for t in suggestion.get('active_tasks', [])],
                        improved=(status == 'IMPROVED'),
                        score=candidate_score,
                        target_metric=target_metric,
                        executed_notebook_path=executed_nb_path,
                        metrics_path=iter_metrics_path,
                        reflection=suggestion.get('reflection_on_history', None),
                        observation={
                            'decision_type': decision_tag,
                            'focus_area': focus_tag,
                            'semantic_gradient_analysis': sem_grad,
                            'baseline_score': comparison_baseline,
                            'best_score_so_far': best_score_so_far,
                        },
                        intervention={
                            'applied_edits': [
                                {
                                    'cell_index': e.get('cell_index'),
                                    'reason': e.get('reason') or '',
                                } for e in (suggestion.get('edits', []) or [])
                            ],
                            'applied_count': applied_count,
                            'candidate_notebook_path': candidate_nb_path,
                        },
                        evidence={
                            'external_knowledge_latest_md': os.path.join(workspace_dir, 'external_knowledge', 'external_knowledge_review.md'),
                            'external_knowledge_latest_json': os.path.join(workspace_dir, 'external_knowledge', 'external_knowledge_review.json'),
                        },
                        trace_path=getattr(tlog, 'trace_path', None),
                    )
                except TypeError:
                    try:
                        update_after_iteration(task_graph_state, status=status, score=candidate_score, rationale=critique)
                    except Exception:
                        pass
                except Exception as e:
                    _log(f"[WARN] Failed to update task graph state: {e}", console=False)

                try:
                    save_task_graph(task_graph_state, task_graph_state_path)
                except Exception as e:
                    _log(f"[WARN] Failed to save task graph state: {e}", console=False)

                history_summary.append({
                    "iter": i,
                    "strategy": strategy_tag,
                    "decision": decision_tag,
                    "focus": focus_tag,
                    "reflection": reflection,
                    "critique": critique,
                    "semantic_gradient": sem_grad,
                    "status": status,
                    "score": candidate_score,
                    "tasks": [t.get("id") for t in suggestion.get("active_tasks", [])],
                    "task_names": [t.get("name") for t in suggestion.get("active_tasks", [])],
                    "usage": usage_stats # [TELEM] Save cost data
                })

                try:
                    tlog.log_step('review.iter_end', 'Iteration complete', iteration=i, status=status, score=float(candidate_score))
                except Exception:
                    pass

                # Save history immediately
                try:
                    with open(history_state_path, "w", encoding="utf-8") as f:
                        json.dump(history_summary, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    _log(f"[WARN] Failed to save history state: {e}", console=False)

                if status == "IMPROVED":
                    _log(f"🎉 NEW BEST! Updating baseline for next iteration.", console=True)
                    best_score_so_far = candidate_score
                    best_nb_path = executed_nb_path
                    current_metrics_path = iter_metrics_path

                    if write_experiment_report:
                        try:
                            with open(iter_metrics_path, 'r', encoding="utf-8") as f:
                                m_obj = json.load(f)
                            write_experiment_report(workspace_dir, m_obj, cfg, primary_metric=target_metric)
                        except Exception as e:
                            _log(f"[WARN] Report gen failed: {e}", console=False)

                    # [FIX] Final threshold checks with direction
                    beats_threshold = False
                    if direction == "minimize":
                        beats_threshold = (best_score_so_far <= threshold)
                    else:
                        beats_threshold = (best_score_so_far >= threshold)
                    
                    beats_baseline = True
                    if static_baseline_score is not None and static_baseline_score != WORST_SCORE:
                        if direction == "minimize":
                            beats_baseline = (best_score_so_far < static_baseline_score)
                        else:
                            beats_baseline = (best_score_so_far > static_baseline_score)

                    if beats_threshold and beats_baseline:
                        _log(f"\n[SUCCESS] Goal Reached! Score {best_score_so_far:.4f} vs Threshold ({threshold})", console=True)
                        break
                    elif beats_threshold and not beats_baseline:
                        _log(f"\n[CONTINUE] Threshold passed ({best_score_so_far:.4f} vs {threshold}), but NOT beating Baseline ({static_baseline_score:.4f}). Continuing...", console=True)
                else:
                    _log(f"📉 Improvement failed. Reverting to previous best.", console=True)

            except Exception as e:
                _log(f"[ERROR] Logic Error after execution: {e}", console=True)
                
                # [TELEM] Recover usage even on error
                usage_stats = TokenMeter.get_and_reset()
                
                history_summary.append({
                    "iter": i,
                    "strategy": strategy_tag,
                    "decision": decision_tag,
                    "focus": focus_tag,
                    "critique": "Execution Logic Crash",
                    "status": "CRASH",
                    "score": -999,
                    "reflection": "Code crashed before metrics could be read.",
                    "usage": usage_stats # [TELEM]
                })
                try:
                    with open(history_state_path, "w", encoding="utf-8") as f:
                        json.dump(history_summary, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

    except KeyboardInterrupt:
        _log("[INTERRUPT] KeyboardInterrupt received. Saving history_state.json and exiting...", console=True)
        raise
    finally:
        # [RF-01 FIX] Ensure history/task-graph are persisted even on crash/interrupt
        try:
            with open(history_state_path, "w", encoding="utf-8") as f:
                json.dump(history_summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _log(f"[WARN] Failed to save history state in finally: {e}", console=False)
        try:
            save_task_graph(task_graph_state, task_graph_state_path)
        except Exception as e:
            _log(f"[WARN] Failed to save task graph state in finally: {e}", console=False)

    # =============================================================================
    # 9. Finalize: Save Best Artifacts
    # =============================================================================
    _log(f"\n🏁 Optimization Finished. Global Best Score: {best_score_so_far:.4f}", console=True)

    if best_nb_path and os.path.exists(best_nb_path):
        final_nb_path = os.path.join(workspace_dir, "notebook_best.ipynb")
        try:
            shutil.copy(best_nb_path, final_nb_path)
            _log(f"✅ Saved BEST Notebook to: {final_nb_path}", console=True)
        except Exception as e:
            _log(f"[WARN] Failed to save BEST Notebook: {e}", console=False)
    else:
        _log(f"⚠️ Could not locate best notebook source: {best_nb_path}", console=True)

    if current_metrics_path and os.path.exists(current_metrics_path):
        final_met_path = os.path.join(workspace_dir, "metrics_best.json")
        try:
            shutil.copy(current_metrics_path, final_met_path)
            _log(f"✅ Saved BEST Metrics to: {final_met_path}", console=True)
        except Exception as e:
            _log(f"[WARN] Failed to save BEST Metrics: {e}", console=False)

        if write_experiment_report:
            try:
                with open(final_met_path, 'r', encoding="utf-8") as f:
                    m_obj = json.load(f)
                write_experiment_report(workspace_dir, m_obj, cfg, primary_metric=target_metric)
                _log(f"✅ Regenerated Experiment Report based on BEST metrics.", console=True)
            except Exception as e:
                _log(f"[WARN] Final Report gen failed: {e}", console=False)
    else:
        _log(f"⚠️ Could not locate best metrics source: {current_metrics_path}", console=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="review_feedback_config.json")
    parser.add_argument("--source_path", default=None,
                        help="Explicit path to the previous run directory (overrides config)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        _log(f"[FATAL] Config not found: {args.config}", console=True)
        return
    cfg = load_full_config(args.config)
    _inject_llm_env(cfg)

    try:
        explicit_source = args.source_path
        if not explicit_source:
            explicit_source = (cfg.get("paths", {}) or {}).get("explicit_source_path")

        source_trial = find_best_experiment_trial(cfg, explicit_source)
        workspace, base_nb = setup_phase3_workspace(cfg, source_trial)
        optimize_loop(cfg, workspace, base_nb)
    except Exception as e:
        _log(f"[FATAL] {e}", console=True)

if __name__ == "__main__":
    main()
