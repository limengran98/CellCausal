# design_execution/prompt_orchestrator.py
import os, json, shutil, datetime
from typing import Dict, Any, Optional
import nbformat

from .prompt_generator import generate_notebook_content
from .prompt_executor import run_notebook_with_autofix
from ..pipeline.utils import export_notebook_as_py, safe_copy
from .experiment_report import write_experiment_report
from .task_logger import get_task_logger

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

# Import Visualization Tool
try:
    from .prompt_viz import write_hypergraph_viz
except ImportError:
    write_hypergraph_viz = None

# =============================================================================
# Helper: Path Management
# =============================================================================

def _get_save_root(cfg: Dict[str, Any]) -> str:
    return cfg.get("prompt_branch", {}).get("save_root", cfg["paths"]["design_execution_root"])

def _get_latest_trial(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Robust discovery of the latest trial directory.
    Priority:
    1. Direct 'workspace_*' or 'prompt_run_*' in save_root (New flattened structure).
    2. Legacy 'prompt/prompt_run_*' (Backward compatibility).
    """
    root = _get_save_root(cfg)
    if not os.path.exists(root): return None
    
    # 1. Search in root
    candidates = []
    for p in os.listdir(root):
        full_p = os.path.join(root, p)
        if os.path.isdir(full_p) and (p.startswith("workspace_") or p.startswith("prompt_run_")):
            candidates.append(full_p)
            
    # 2. Search in legacy 'prompt' subdir
    legacy_root = os.path.join(root, "prompt")
    if os.path.exists(legacy_root):
        for p in os.listdir(legacy_root):
            full_p = os.path.join(legacy_root, p)
            if os.path.isdir(full_p) and (p.startswith("workspace_") or p.startswith("prompt_run_")):
                candidates.append(full_p)
    
    if not candidates:
        return None
        
    # Sort by modification time, newest first
    candidates.sort(key=lambda x: os.path.getmtime(x))
    return candidates[-1]

def _audit_intermediate_files(trial_dir: str):
    """
    Force audit and print files in intermediate directory to ensure visibility of the process.
    """
    inter_dir = os.path.join(trial_dir, "intermediate")
    _log(f"\n[ORCH] 🔎 Auditing Intermediate Results in: {inter_dir}", console=True)
    
    if not os.path.exists(inter_dir):
        _log("   ⚠️ Directory NOT created by Notebook. (Did the model skip saving?)", console=True)
        return

    files = []
    for root, _, filenames in os.walk(inter_dir):
        for f in filenames:
            path = os.path.join(root, f)
            try:
                size_kb = os.path.getsize(path) / 1024
            except OSError:
                size_kb = 0
            rel_path = os.path.relpath(path, trial_dir)
            files.append((rel_path, size_kb))
    
    if not files:
        _log("   ⚠️ Directory exists but is EMPTY.", console=True)
    else:
        # Sort by name
        files.sort()
        for fname, fsize in files:
            _log(f"   📄 {fname:<40} | {fsize:>6.1f} KB", console=False)
    _log("", console=False) # Spacer


def _ensure_result_folders(trial_dir: str) -> Dict[str, str]:
    """Create and return the canonical result folders for a trial."""
    final_dir = os.path.join(trial_dir, "final_keep")
    inter_dir = os.path.join(trial_dir, "intermediate")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    return {"final": final_dir, "intermediate": inter_dir}

# =============================================================================
# Phases
# =============================================================================

def phase_generate(
    cfg: Dict[str, Any], 
    spec_path: str, 
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """Generate Notebook (Experiment stage)."""
    
    out_root = _get_save_root(cfg)
    ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()

    # [REFACTOR] 1. Determine Trial Directory FIRST
    if run_name:
        # Loop mode: Use fixed workspace
        trial_dir = os.path.join(out_root, run_name)
        # Clean up previous run data if exists to avoid mixing
        if os.path.exists(trial_dir):
            try:
                shutil.rmtree(trial_dir)
                _log(f"[ORCH] 🧹 Cleaned up previous workspace: {trial_dir}", console=False)
            except OSError as e:
                _log(f"[ORCH][WARN] Failed to clean workspace {trial_dir}: {e}", console=False)
    else:
        # Standard mode: Timestamp + PID for safety (No 'prompt' subdir nesting)
        trial_dir = os.path.join(out_root, f"prompt_run_{ts_now}_{pid}")
        
    os.makedirs(trial_dir, exist_ok=True)
    
    # [REFACTOR] 2. Create structure immediately
    dirs = _ensure_result_folders(trial_dir)
    
    # [REFACTOR] 3. Define Debug Dir INSIDE intermediate (No external redundancy)
    debug_dir = os.path.join(dirs["intermediate"], "debug_prompt")
    os.makedirs(debug_dir, exist_ok=True)

    # [TRACE] Structured task trace
    tlog = get_task_logger(trial_dir)
    tlog.log_step('phase_generate.start', 'Begin notebook generation', spec_path=os.path.abspath(spec_path), run_name=run_name or '')
    tlog.log_artifact('spec', spec_path, 'Technical specification')

    # [GENERATE] Write artifacts directly into the trial's debug_dir
    nb, _user_prompt, strategy_md = generate_notebook_content(cfg, spec_path, debug_dir)
    
    _log(f"[ORCH] 🧾 Debug artifacts written to: {debug_dir}", console=False)
    try:
        tlog.log_artifact('debug_prompt', debug_dir, 'Debug prompt artifacts')
    except Exception:
        pass
    
    nb_path = os.path.join(trial_dir, "notebook_prompt.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    try:
        tlog.log_artifact('notebook', nb_path, 'Generated notebook prompt')
    except Exception:
        pass

    # Immediately snapshot to final_keep
    safe_copy(nb_path, dirs["final"], "notebook_prompt.ipynb")

    # Also export a .py for quick inspection/diffing
    try:
        export_notebook_as_py(nb_path, nb_path.replace(".ipynb", ".py"))
    except Exception:
        pass

    # Immediately snapshot the .py as well
    safe_copy(nb_path.replace(".ipynb", ".py"), dirs["final"], "notebook_prompt.py")
        
    if strategy_md:
        strat_path = os.path.join(trial_dir, 'research_strategy.md')
        with open(strat_path, 'w', encoding='utf-8') as f:
            f.write(strategy_md)
        try:
            tlog.log_artifact('strategy', strat_path, 'Synthesized research strategy')
        except Exception:
            pass

    try:
        tlog.log_step('phase_generate.end', 'Generation complete', trial_dir=os.path.abspath(trial_dir))
    except Exception:
        pass
    _log(f"[ORCH] Generation Complete. Trial: {trial_dir}", console=False)
    return {"trial_dir": trial_dir, "notebook_path": nb_path}

def phase_execute(cfg: Dict[str, Any], trial_dir: Optional[str] = None) -> Dict[str, Any]:
    """Experiment stage: Execute Notebook with Auto-Fix."""
    tdir = trial_dir or _get_latest_trial(cfg)
    if not tdir:
        raise RuntimeError("No trial directory found.")
        
    nb_path = os.path.join(tdir, "notebook_prompt.ipynb")
    if not os.path.exists(nb_path):
        raise RuntimeError(f"Notebook not found: {nb_path}")
        
    # [TRACE] Structured task trace
    tlog = get_task_logger(tdir)
    tlog.log_step('phase_execute.start', 'Begin notebook execution', trial_dir=os.path.abspath(tdir))

    # Run
    dirs = _ensure_result_folders(tdir)
    final_exec = run_notebook_with_autofix(nb_path, tdir, cfg)

    # Snapshot executed notebook immediately
    if final_exec and os.path.exists(final_exec):
        safe_copy(final_exec, dirs["final"], "notebook_prompt_exec.ipynb")

    # Export executed notebook into .py for convenience
    try:
        export_notebook_as_py(final_exec, final_exec.replace(".ipynb", ".py"))
    except Exception:
        pass

    # Snapshot .py version as well (best-effort)
    if final_exec:
        safe_copy(final_exec.replace(".ipynb", ".py"), dirs["final"], "notebook_prompt_exec.py")

    # Export executed notebook into .py as well
    try:
        export_notebook_as_py(final_exec, final_exec.replace(".ipynb", ".py"))
    except Exception:
        pass
    
    # Generate Hypergraph Visualization
    if write_hypergraph_viz:
        _log(f"[ORCH] Generating Hypergraph Visualization...", console=False)
        viz_out = write_hypergraph_viz(tdir, nb_path, fmt="mermaid")
        if viz_out.get("mermaid"):
            _log(f"[ORCH] Viz saved: {viz_out['mermaid']}", console=False)
            
    # Audit Intermediate Files
    _audit_intermediate_files(tdir)
    
    # Load Metrics
    metrics = {}
    m_path = os.path.join(tdir, "metrics.json")
    if os.path.exists(m_path):
        try:
            with open(m_path, "r") as f: metrics = json.load(f)
        except: pass

        # Snapshot metrics immediately
        safe_copy(m_path, dirs["final"], "metrics.json")
        
    try:
        if final_exec:
            tlog.log_artifact('notebook_exec', final_exec, 'Executed notebook')
        if os.path.exists(m_path):
            tlog.log_artifact('metrics', m_path, 'Execution metrics')
        tlog.log_step('phase_execute.end', 'Execution complete', has_metrics=bool(metrics))
    except Exception:
        pass

    return {"trial_dir": tdir, "exec_notebook": final_exec, "metrics": metrics}

def phase_analyze(cfg: Dict[str, Any], trial_dir: Optional[str] = None) -> Dict[str, Any]:
    """Review stage: Generate Report."""
    tdir = trial_dir or _get_latest_trial(cfg)
    if not tdir:
        raise RuntimeError("No trial directory found.")
    tlog = get_task_logger(tdir)
    tlog.log_step('phase_analyze.start', 'Begin report generation', trial_dir=os.path.abspath(tdir))
    m_path = os.path.join(tdir, "metrics.json")
    if not os.path.exists(m_path):
        _log(f"[ORCH] No metrics.json in {tdir}. Skipping report.", console=False)
        return {}
        
    try:
        dirs = _ensure_result_folders(tdir)
        with open(m_path, "r") as f: metrics = json.load(f)
        
        pm = ((cfg.get("experiment") or {}).get("primary_metric") or "PCC")
        
        report_path = write_experiment_report(tdir, metrics, cfg, primary_metric=pm)
        _log(f"[ORCH] Report written: {report_path}", console=False)
        try:
            tlog.log_artifact('report', report_path, 'Experiment report')
        except Exception:
            pass
        # Snapshot report and any analysis errors immediately
        safe_copy(report_path, dirs["final"], "experiment_report.md")
        safe_copy(os.path.join(tdir, "analysis_llm_error_traceback.txt"), dirs["final"], "analysis_llm_error_traceback.txt")
        try:
            tlog.log_step('phase_analyze.end', 'Report generation complete')
        except Exception:
            pass
        return {"report_path": report_path}
    except Exception as e:
        _log(f"[ORCH] Analysis failed: {e}", console=False)
        return {}

def run_full_pipeline(
    cfg: Dict[str, Any], 
    spec_path: str,
    run_name: Optional[str] = None # Pass down run_name
) -> Dict[str, Any]:
    
    _log("├─ 🔎 STEP 1: GENERATE", console=True)
    # Pass run_name to control folder creation/overwriting
    gen_res = phase_generate(cfg, spec_path, run_name=run_name)
    
    _log("├─ 🔎 STEP 2: EXECUTE", console=True)
    exec_res = phase_execute(cfg, gen_res["trial_dir"])
    
    _log("└─ 🔎 STEP 3: ANALYZE", console=True)
    phase_analyze(cfg, gen_res["trial_dir"])
    
    return exec_res