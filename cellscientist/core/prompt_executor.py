# design_execution/prompt_executor.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, json, re, hashlib, ast, shutil, threading, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from copy import deepcopy
import traceback

# Import robust chat utilities
from .llm_client import chat_json, chat_text

# Unified auto-fix utilities
from . import notebook_autofix as _autofix

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
# 0. Robust Helper Tools (The "Nuclear" Parser v2.0)
# =============================================================================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Compatibility wrapper.

    The original Phase-2 implementation contained a large, robust parser.
    That logic is now centralized in `notebook_autofix.extract_edits_spec`
    to eliminate duplication while preserving behavior.
    """
    return _autofix.extract_edits_spec(text)


def _dump_graph_error_log(workdir: str, task_id: str, errors: List[Dict[str, Any]], round_idx: int = 0) -> str:
    """Persist detailed error info for graph executor runs.

    This complements console prints, ensuring users can inspect full tracebacks
    even when running long pipelines.
    """
    os.makedirs(workdir, exist_ok=True)
    safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(task_id))
    log_path = os.path.join(workdir, f"graph_error_{safe_task}_round_{round_idx}.txt")
    lines = [f"=== GRAPH ERROR REPORT | task={task_id} | round={round_idx} ===\n"]
    for e in errors:
        lines.append("-" * 80 + "\n")
        lines.append(f"CELL INDEX: {e.get('cell_index')}\n")
        lines.append(f"ERROR TYPE: {e.get('ename')}\n")
        lines.append(f"MESSAGE   : {e.get('evalue')}\n")
        lines.append("\n--- SOURCE ---\n")
        lines.append((e.get("source") or "") + "\n")
        lines.append("\n--- TRACEBACK ---\n")
        lines.append((e.get("traceback") or "") + "\n")
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    _log(f"[GRAPH][ERROR DUMP] Saved detailed error log: {log_path}", console=False)
    return log_path

# =============================================================================
# 1. Heuristic Logic (Static Analysis)
# =============================================================================

def _get_heuristic_patch(evalue: str) -> Optional[str]:
    """Compatibility wrapper to unified heuristic patch selection.

    The original inline heuristic logic is centralized in `notebook_autofix`.
    """
    return _autofix._get_heuristic_patch(evalue)

def _apply_heuristics(nb: nbformat.NotebookNode, errors: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
    """Compatibility wrapper for unified heuristics.

    Returns:
      (changed_count, patched_cell_indices)
    """
    return _autofix.apply_heuristics(nb, errors)

# =============================================================================
# 2. LLM Auto-Fix Logic (Enhanced for Long Code)
# =============================================================================

def _compute_cell_hash(source: str) -> str:
    """Compute MD5 hash of cell content ignoring whitespace.

    Wrapper to unified implementation in notebook_autofix.
    """
    return _autofix._compute_cell_hash(source)


def _apply_llm_edits(nb: nbformat.NotebookNode, edits: List[Dict[str, Any]]) -> int:
    """Apply edits and return number of effective changes (Hash Verified).

    Wrapper to unified implementation in notebook_autofix.
    """
    return _autofix.apply_llm_edits(nb, edits)


def _llm_auto_fix_once(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    llm_cfg: Dict[str, Any],
    autofix_system_prompt: str
) -> Tuple[nbformat.NotebookNode, bool]:
    """
    Wrapper to unified Phase-2 style LLM auto-fix (triple-quoted Python dict output).
    """
    return _autofix.llm_autofix_once_design(nb, errors, llm_cfg, autofix_system_prompt)

# =============================================================================
# 3. Graph Executor (Adaptive Node-by-Node Execution)
# =============================================================================

class GraphExecutor(NotebookClient):
    """
    Advanced Executor that runs the notebook as a directed graph of cells.
    Allows interrupting execution on error, fixing the specific node (cell),
    and retrying IN-PLACE without restarting the kernel.
    """
    def __init__(
        self,
        nb,
        workdir,
        llm_config,
        autofix_prompt,
        max_fix_rounds=3,
        *,
        checkpoint_each_cell: bool = True,
        snapshot_files: bool = True,
        snapshot_max_bytes: int = 200 * 1024 * 1024,
        heartbeat_seconds: int = 120,
        **kwargs,
    ):
        super().__init__(nb, **kwargs)
        self.nb = nb
        self.workdir = os.path.abspath(workdir)
        self.llm_config = llm_config or {}
        self.autofix_prompt = autofix_prompt
        self.max_fix_rounds = int(max_fix_rounds)

        # New: aggressive persistence (requested)
        self.checkpoint_each_cell = bool(checkpoint_each_cell)
        self.snapshot_files = bool(snapshot_files)
        self.snapshot_max_bytes = int(snapshot_max_bytes) if snapshot_max_bytes is not None else 0
        self.heartbeat_seconds = max(15, int(heartbeat_seconds or 120))

        self.global_errors = []
        self._cell_fix_counts = {}  # (cell_idx, task_id) -> total patches attempted
        self._cell_last_error_sig = {}
        self.autofix_attempted_cells = 0
        self.autofix_success_cells = 0
        self.execution_stats = {}

        self._ensure_artifact_dirs()
        self._file_manifest = self._scan_files() if self.snapshot_files else {}



    def execute_graph(self):
        """Main entry point for graph execution."""
        _log(f"[GRAPH] 🚀 Initializing Kernel in {self.workdir}", console=False)
        
        # 1. Setup Environment & Guard Code
        self._inject_setup_cells()
        
        # 2. Start Persistent Kernel
        self.create_kernel_manager()
        self.start_new_kernel()
        self.start_new_kernel_client()

        # Track execution statistics
        total_cells = sum(1 for c in self.nb.cells if c.cell_type == 'code')
        executed_cells = 0
        failed_cells = 0
        autofix_used = False

        try:
            # 3. Iterate Cells (Nodes)
            cell_idx = 0
            while cell_idx < len(self.nb.cells):
                cell = self.nb.cells[cell_idx]
                
                if cell.cell_type != 'code':
                    cell_idx += 1
                    continue

                # Identify Task
                task_meta = cell.metadata.get("subtask", {})
                task_id = task_meta.get("id", f"Cell_{cell_idx}")
                task_name = task_meta.get("name", "Unnamed")
                
                _log(f"├─ ⚛ Execute: Cell {executed_cells + 1}/{total_cells} [{task_name}]", console=True)

                try:
                    # Execute single cell using nbclient's low-level method
                    self._execute_cell_with_heartbeat(cell, cell_idx, task_name, executed_cells + 1, total_cells)

                    # If we are here, execution was successful
                    self._after_cell_success(cell_idx, task_id)
                    executed_cells += 1
                    cell_idx += 1 
                
                except CellExecutionError:
                    _log(f"├─ ❌ Error in Cell {executed_cells + 1} [{task_name}]", console=True)
                    self._after_cell_error(cell_idx, task_id)
                    
                    # Try to fix IN-PLACE
                    self.autofix_attempted_cells += 1
                    fixed = self._attempt_node_fix(cell_idx, task_id)
                    autofix_used = True
                    if fixed:
                        self.autofix_success_cells += 1
                    
                    if fixed:
                        _log(f"├─ 🔧 Auto-fix applied → retrying cell", console=True)
                        # We do NOT increment cell_idx, so the loop will re-execute the SAME cell index
                        # but with the new source code we just patched into self.nb
                        continue 
                    else:
                        _log(f"├─ 🛑 Auto-fix failed after {self.max_fix_rounds} rounds", console=True)
                        self.global_errors.append(f"Task {task_id} Failed.")
                        failed_cells += 1
                        # Stop execution here to preserve partial results or debug
                        break
        
        finally:
            _log("[GRAPH] 🛑 Shutting down kernel.", console=False)
            self._cleanup_kernel()
            
            # Print execution summary
            success_status = "✅ Success" if failed_cells == 0 else f"❌ Failed ({failed_cells} errors)"
            autofix_note = " (with autofix)" if autofix_used and failed_cells == 0 else " (no autofix)" if not autofix_used else ""
            _log(f"└─ ⚛ Execute: {executed_cells} cells → {success_status}{autofix_note}", console=True)

            fix_rounds = [int(v) for v in self._cell_fix_counts.values() if int(v) > 0]
            self.execution_stats = {
                "total_cells": int(total_cells),
                "executed_cells": int(executed_cells),
                "failed_cells": int(failed_cells),
                "notebook_success": bool(failed_cells == 0),
                "autofix_attempted_cells": int(self.autofix_attempted_cells),
                "autofix_success_cells": int(self.autofix_success_cells),
                "autofix_success_rate": float(self.autofix_success_cells / self.autofix_attempted_cells) if self.autofix_attempted_cells > 0 else None,
                "avg_fix_rounds": float(sum(fix_rounds) / len(fix_rounds)) if fix_rounds else 0.0,
                "max_fix_rounds_used": int(max(fix_rounds)) if fix_rounds else 0,
                "unresolved_error_count": int(len(self.global_errors)),
            }

        return self.nb

    def _execute_cell_with_heartbeat(self, cell, cell_idx: int, task_name: str, display_idx: Optional[int] = None, total_cells: Optional[int] = None):
        """Execute one notebook cell and emit periodic heartbeat logs while it runs."""
        started_at = time.time()
        stop_event = threading.Event()
        display_idx = int(display_idx) if display_idx is not None else (int(cell_idx) + 1)
        total_cells = int(total_cells) if total_cells is not None else max(display_idx, 1)

        def _heartbeat_loop():
            while not stop_event.wait(self.heartbeat_seconds):
                elapsed = int(time.time() - started_at)
                timeout_sec = int(getattr(self, "timeout", 0) or 0)
                if timeout_sec > 0:
                    _log(
                        f"├─ ⏱️ Running: Cell {display_idx}/{total_cells} [{task_name}] | elapsed={elapsed}s | timeout={timeout_sec}s",
                        console=True,
                    )
                else:
                    _log(
                        f"├─ ⏱️ Running: Cell {display_idx}/{total_cells} [{task_name}] | elapsed={elapsed}s",
                        console=True,
                    )

        beat = threading.Thread(target=_heartbeat_loop, daemon=True)
        beat.start()
        try:
            return self.execute_cell(cell, cell_idx)
        finally:
            stop_event.set()
            beat.join(timeout=0.1)

    def _inject_setup_cells(self):
        """Inject setup code (Env vars, Guard) at the top."""
        inter_dir = os.path.join(self.workdir, "intermediate")
        final_dir = os.path.join(self.workdir, "final_keep")

        guard_code = f"""# [AUTO-FIX] Guard Cell & Env Setup
import sys, os, json, pickle, pathlib, time

os.environ['OUTPUT_DIR'] = r'{self.workdir}'
os.environ['INTERMEDIATE_DIR'] = r'{inter_dir}'
os.environ['FINAL_KEEP_DIR'] = r'{final_dir}'

os.makedirs(os.environ['INTERMEDIATE_DIR'], exist_ok=True)
os.makedirs(os.environ['FINAL_KEEP_DIR'], exist_ok=True)

def _guard_exit(*args, **kwargs):
    raise RuntimeError("SysExitBlocked: Use raise ValueError() instead.")

sys.exit = _guard_exit

# Lightweight helpers notebooks can use to save frequently
def _atomic_write_bytes(path, data: bytes):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, str(p))

def save_json(obj, name: str, subdir: str = ""):
    base = os.environ.get("INTERMEDIATE_DIR") or os.environ.get("OUTPUT_DIR") or "."
    out = os.path.join(base, subdir, name)
    if not out.endswith(".json"):
        out += ".json"
    _atomic_write_bytes(out, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))
    return out

print("[SETUP] Environment Configured.", "OUTPUT_DIR=", os.environ.get("OUTPUT_DIR"), "INTERMEDIATE_DIR=", os.environ.get("INTERMEDIATE_DIR"))
"""

        self.nb.cells.insert(0, nbformat.v4.new_code_cell(guard_code))

    # ------------------------------------------------------------------
    # Aggressive persistence utilities (checkpoint + snapshot)
    # ------------------------------------------------------------------

    def _ensure_artifact_dirs(self):
        self.intermediate_dir = os.path.join(self.workdir, "intermediate")
        self.final_keep_dir = os.path.join(self.workdir, "final_keep")
        self.checkpoints_dir = os.path.join(self.intermediate_dir, "checkpoints")
        self.snapshots_dir = os.path.join(self.intermediate_dir, "snapshots")
        for d in (self.intermediate_dir, self.final_keep_dir, self.checkpoints_dir, self.snapshots_dir):
            os.makedirs(d, exist_ok=True)

    def _scan_files(self):
        """Return a manifest of files under workdir excluding intermediate/final_keep."""
        manifest = {}
        skip_names = {"intermediate", "final_keep", "__pycache__", ".ipynb_checkpoints"}
        for root, dirs, files in os.walk(self.workdir):
            dirs[:] = [d for d in dirs if d not in skip_names and not d.startswith(".")]
            for fn in files:
                # [UPDATED] Include dotfiles generally, exclude specific ones if needed
                if fn == ".DS_Store": continue 
                
                abs_p = os.path.join(root, fn)
                if abs_p.startswith(self.intermediate_dir) or abs_p.startswith(self.final_keep_dir):
                    continue
                try:
                    st = os.stat(abs_p)
                except OSError:
                    continue
                rel = os.path.relpath(abs_p, self.workdir)
                manifest[rel] = (float(st.st_mtime), int(st.st_size))
        return manifest

    def _safe_task(self, task_id: str) -> str:
        s = str(task_id or "cell")
        s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
        return s[:64] if s else "cell"

    def _write_notebook_checkpoint(self, cell_idx: int, task_id: str, *, tag: str = "after"):
        try:
            safe = self._safe_task(task_id)
            ck = os.path.join(self.checkpoints_dir, f"nb_{tag}_{cell_idx:03d}_{safe}.ipynb")
            nbformat.write(self.nb, ck)
            latest = os.path.join(self.final_keep_dir, "notebook_latest.ipynb")
            nbformat.write(self.nb, latest)
        except Exception as e:
            _log(f"[EXEC][WARN] Failed to checkpoint notebook at cell {cell_idx}: {e}", console=False)

    def _copy_changed_files(self, old_manifest: dict, new_manifest: dict, cell_idx: int, task_id: str, *, tag: str = "after"):
        # [UPDATED] Force copy ANY file in new_manifest to intermediate dir to guarantee persistence
        # We ignore old_manifest for the copy to intermediate/ to be safe (overwrite/update)
        
        safe = self._safe_task(task_id)
        # We don't create a new folder per cell anymore to avoid massive duplication,
        # instead we sync to 'intermediate/' directly so user sees latest state.
        
        out_dir = self.intermediate_dir # Sync to root of intermediate

        copied = 0
        skipped = 0
        
        for rel, meta in new_manifest.items():
            abs_src = os.path.join(self.workdir, rel)
            
            # Size check
            try:
                size = int(meta[1])
            except Exception:
                size = 0
            if self.snapshot_max_bytes and size > self.snapshot_max_bytes:
                skipped += 1
                continue

            abs_dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
            try:
                shutil.copy2(abs_src, abs_dst)
                copied += 1
            except Exception:
                skipped += 1

        if copied > 0:
            _log(f"   💾 Synced {copied} files to intermediate/", console=False)

    def _after_cell_success(self, cell_idx: int, task_id: str):
        if self.checkpoint_each_cell:
            self._write_notebook_checkpoint(cell_idx, task_id, tag="after")
        if self.snapshot_files:
            new_manifest = self._scan_files()
            # [UPDATED] Trigger copy logic
            self._copy_changed_files(self._file_manifest, new_manifest, cell_idx, task_id, tag="after")
            self._file_manifest = new_manifest

    def _after_cell_error(self, cell_idx: int, task_id: str):
        if self.checkpoint_each_cell:
            self._write_notebook_checkpoint(cell_idx, task_id, tag="error")
        if self.snapshot_files:
            new_manifest = self._scan_files()
            self._copy_changed_files(self._file_manifest, new_manifest, cell_idx, task_id, tag="error")
            self._file_manifest = new_manifest

    def _get_cell_errors(self, cell_idx: int) -> List[Dict[str, Any]]:
        """Extract error details from the cell outputs."""
        cell = self.nb.cells[cell_idx]
        errs = []
        for out in (cell.get("outputs") or []):
            if out.get("output_type") == "error":
                tb = out.get("traceback", [])
                tb_str = "\n".join(tb) if isinstance(tb, list) else str(tb)
                tb_clean = re.sub(r'\x1b\[[0-9;]*m', '', tb_str)
                errs.append({
                    "cell_index": cell_idx,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": tb_clean,
                    "source": cell.get("source", ""),
                })
        return errs

    def _dump_graph_error_log(self, workdir: str, task_id: str, round_idx: int, errors: List[Dict[str, Any]]) -> str:
        """Persist detailed error info to disk (so users don't have to scroll logs)."""
        try:
            os.makedirs(workdir, exist_ok=True)
            safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(task_id))
            log_path = os.path.join(workdir, f"graph_error_{safe_task}_round_{round_idx}.txt")
            report = f"=== GRAPH ERROR REPORT | task={task_id} | round={round_idx} ===\n"
            report += f"errors={len(errors)}\n"
            for e in errors:
                report += "\n" + ("-" * 60) + "\n"
                report += f"CELL INDEX: {e.get('cell_index')}\n"
                report += f"ERROR TYPE: {e.get('ename')}\n"
                report += f"MESSAGE   : {e.get('evalue')}\n"
                report += ("-" * 20) + " SOURCE " + ("-" * 20) + "\n"
                report += (e.get("source") or "") + "\n"
                report += ("-" * 20) + " TRACEBACK " + ("-" * 20) + "\n"
                report += (e.get("traceback") or "") + "\n"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(report)
            _log(f"[GRAPH][ERROR] Saved detailed error log: {log_path}", console=False)
            return log_path
        except Exception:
            return ""

    def _attempt_node_fix(self, cell_idx: int, task_id: str) -> bool:
        """
        The Local Fix Loop.
        Returns True if the cell source was modified and should be retried.
        """
        key = (cell_idx, task_id)
        base = int(self._cell_fix_counts.get(key, 0) or 0)
        if base >= self.max_fix_rounds:
            return False

        # Always dump current error details for this task/round
        errs = self._get_cell_errors(cell_idx)
        self._dump_graph_error_log(self.workdir, task_id, base + 1, errs)

        for attempt in range(base, self.max_fix_rounds):
            errors = self._get_cell_errors(cell_idx)
            if not errors:
                return False

            # fingerprint current error to help debugging / avoid runaway retries
            try:
                sig_parts = []
                for e in errors:
                    en = str(e.get("ename", ""))
                    ev = str(e.get("evalue", ""))
                    tb = str(e.get("traceback", ""))[:300]
                    sig_parts.append(en + ":" + ev + ":" + tb)
                err_sig = "|".join(sig_parts)
            except Exception:
                err_sig = "unknown"

            self._cell_last_error_sig[key] = err_sig

            round_no = attempt + 1
            _log(f"   🔧 Auto-fix (Round {round_no}/{self.max_fix_rounds})", console=True)

            # 1. Heuristics (Fast path)
            h_changes, _ = _apply_heuristics(self.nb, errors)
            if h_changes > 0:
                _log(f"   ✅ Heuristic patch applied", console=False)
                self._cell_fix_counts[key] = round_no
                return True

            # 2. LLM (Slow path)
            # Pass a COPY of notebook to LLM func to avoid partial mutations if it fails
            nb_copy = deepcopy(self.nb)
            
            # NOTE: We only send the CURRENT failing cell errors to the LLM
            nb_fixed, patched = _llm_auto_fix_once(
                nb_copy, 
                errors, 
                self.llm_config, 
                self.autofix_prompt
            )

            if patched:
                # Update our live notebook's specific cell
                new_source = nb_fixed.cells[cell_idx].source
                if self.nb.cells[cell_idx].source != new_source:
                    self.nb.cells[cell_idx].source = new_source
                    _log(f"   ✅ LLM patch applied", console=False)
                    self._cell_fix_counts[key] = round_no
                    return True
                else:
                    _log(f"   ⚠️ LLM returned identical code", console=False)
            
        return False

# =============================================================================
# 4. Main Entry Point
# =============================================================================

def run_notebook_with_autofix(
    nb_path: str,
    workdir: str,
    cfg: Dict[str, Any]
) -> str:
    """
    Executes notebook using the Adaptive Graph Executor.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    
    # Load Notebook
    nb_orig = nbformat.read(nb_path, as_version=4)
    
    # Config
    exec_cfg = cfg.get("exec", {})
    max_rounds = int(exec_cfg.get("max_fix_rounds", 3))
    
    prompts_map = cfg.get("prompts", {})
    autofix_prompt = prompts_map.get("autofix", {}).get("system_prompt", 
        "You are a Python Expert. Fix the code errors provided. Return a Python Dictionary with 'edits'.")

    # Instantiate Graph Executor
    executor = GraphExecutor(
        nb=nb_orig,
        workdir=workdir,
        llm_config=cfg.get("llm", {}),
        autofix_prompt=autofix_prompt,
        max_fix_rounds=max_rounds,
        checkpoint_each_cell=bool(exec_cfg.get("checkpoint_each_cell", True)),
        snapshot_files=bool(exec_cfg.get("snapshot_files", True)),
        snapshot_max_bytes=int(exec_cfg.get("snapshot_max_bytes", 200 * 1024 * 1024)),
        # nbclient args
        timeout=int(exec_cfg.get("timeout_seconds", 3600)),
        heartbeat_seconds=int(exec_cfg.get("heartbeat_seconds", 120)),
        kernel_name="python3",
        allow_errors=False, # We handle errors manually
        resources={"metadata": {"path": workdir}}
    )
    
    # Run
    _log(f"[EXEC] Starting Adaptive Graph Execution: {nb_path}", console=True)
    framework_recovered = False
    try:
        final_nb = executor.execute_graph()
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"[EXEC] ☢️ Critical Framework Error: {e}", console=True)
        _log(tb, console=False)
        try:
            os.makedirs(workdir, exist_ok=True)
            with open(os.path.join(workdir, "framework_error_traceback.txt"), "w", encoding="utf-8") as f:
                f.write(tb)
        except Exception:
            pass
        final_nb = executor.nb
        framework_recovered = True

    # Save Result
    out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    
    # Remove the injected setup cell (index 0) before saving
    if len(final_nb.cells) > 0 and "[AUTO-FIX] Guard Cell" in final_nb.cells[0].source:
        final_nb.cells.pop(0)

    nbformat.write(final_nb, out_path)
    _log(f"[EXEC] Saved final state: {out_path}", console=True)
    
    if executor.global_errors:
        _log(f"[EXEC] Finished with unresolved errors: {executor.global_errors}", console=False)
    else:
        _log(f"[EXEC] ✅ Execution completed successfully.", console=True)
        
    # Attach execution-level robustness stats to metrics.json (existing or stub).
    m_path = os.path.join(workdir, "metrics.json")
    execution_stats = dict(getattr(executor, "execution_stats", {}) or {})
    execution_stats["framework_recovered"] = bool(framework_recovered)
    execution_stats["crash_recovered_ratio"] = 1.0 if framework_recovered else 0.0
    try:
        if os.path.exists(m_path):
            with open(m_path, "r", encoding="utf-8") as f:
                m_obj = json.load(f) if f.readable() else {}
            if not isinstance(m_obj, dict):
                m_obj = {}
        else:
            m_obj = {}
        m_obj["execution_stats"] = execution_stats
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(m_obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # If notebook did not create metrics.json, write a stub so downstream stages don't silently show -999.
    m_path = os.path.join(workdir, "metrics.json")
    if not os.path.exists(m_path):
        metrics_stub = {
            "status": "MISSING_METRICS_JSON",
            "note": "Notebook did not write metrics.json. Check data loading / evaluation cells.",
            "global_errors": executor.global_errors,
        }
        try:
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(metrics_stub, f, ensure_ascii=False, indent=2)
            try:
                os.makedirs(os.path.join(workdir, "final_keep"), exist_ok=True)
                shutil.copy2(m_path, os.path.join(workdir, "final_keep", "metrics.json"))
            except Exception:
                pass
            _log(f"[EXEC][WARN] Wrote stub metrics.json: {m_path}", console=False)
        except Exception:
            pass

    return out_path