from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import re
import hashlib
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from copy import deepcopy

# Import centralized LLM utilities
from .llm_client import chat_json
from .notebook_autofix import attempt_fix_notebook

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
    if console:
        print(f"[CELL_CONSOLE] {msg}", flush=True)
    else:
        print(f"[DETAIL] {msg}", flush=True)

# =============================================================================
# 0. Verbose Client
# =============================================================================

class VerboseNotebookClient(NotebookClient):
    """
    A wrapper that prints real-time progress AND cell outputs (stdout/stderr).
    """
    
    def setup_kernel(self, **kwargs):
        _log(f"[EXEC] 🚀 Starting Kernel Manager...", console=True)
        return super().setup_kernel(**kwargs)

    def _log_start(self, cell, cell_index):
        cell_type = cell.get("cell_type", "unknown")
        snippet = cell.get("source", "").strip().split("\n")[0][:50]
        if cell_type == "code":
            _log(f"[EXEC] ⏳ Cell {cell_index} [CODE] Running... | {snippet}...", console=True)

    def _log_outputs(self, cell, cell_index):
        """Helper to print stdout/stderr from the cell."""
        if cell.get("cell_type") != "code": return
        
        outputs = cell.get("outputs", [])
        for out in outputs:
            
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                text = out.get("text", "").strip()
                if text:
                    _log(f"       📝 [STDOUT]: {text}", console=False)
            

            elif out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = out.get("text", "").strip()
                if text:
                    _log(f"       ⚠️ [STDERR]: {text}", console=False)
            

            elif out.get("output_type") == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "Unknown")
                _log(f"       ❌ [ERROR]: {ename} - {evalue}", console=True)

    def _log_end(self, cell, cell_index, success=True):
        if cell.get("cell_type") == "code":
            icon = "✅" if success else "❌"
            _log(f"[EXEC] {icon} Cell {cell_index} Done.", console=True)


    def execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        self._log_start(cell, cell_index)
        try:
            result = super().execute_cell(cell, cell_index, execution_count, store_history)
            self._log_outputs(cell, cell_index) 
            self._log_end(cell, cell_index, success=True)
            return result
        except Exception as e:
            self._log_outputs(cell, cell_index) 
            self._log_end(cell, cell_index, success=False)
            raise e


    async def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        self._log_start(cell, cell_index)
        try:
            result = await super().async_execute_cell(cell, cell_index, execution_count, store_history)
            self._log_outputs(cell, cell_index) 
            self._log_end(cell, cell_index, success=True)
            return result
        except Exception as e:
            self._log_outputs(cell, cell_index) 
            self._log_end(cell, cell_index, success=False)
            raise e

# =============================================================================
# 1. Error Handling & Reporting
# =============================================================================

def collect_cell_errors(nb: nbformat.NotebookNode) -> List[Dict[str, Any]]:
    """Extracts errors from the notebook object."""
    errs = []
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code": continue
        for out in (c.get("outputs") or []):
            if out.get("output_type") == "error":
                tb_str = "\n".join(out.get("traceback", []))
                tb_clean = re.sub(r'\x1b\[[0-9;]*m', '', tb_str)
                errs.append({
                    "cell_index": i,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": tb_clean,
                    "source": c.get("source", "")
                })
    return errs

def dump_error_log(workdir: str, errors: List[Dict], round_idx: int = 0) -> str:
    log_path = os.path.join(workdir, f"error_log_round_{round_idx}.txt")
    report = f"=== ERROR REPORT (Round {round_idx}) ===\n"
    _log(f"\n{'!'*60}", console=False)
    _log(f"[ERROR DUMP] Found {len(errors)} errors. Details saved to: {log_path}", console=False)
    
    for e in errors:
        idx = e['cell_index']
        _log(f"\n>> Cell {idx} Error: {e['ename']} - {e['evalue']}", console=False)

        trace_tail = e['traceback'][-500:] if len(e['traceback']) > 500 else e['traceback']
        _log(f"   [Traceback]:\n{trace_tail}", console=False)
        
        entry = (
            f"\n{'-'*60}\n"
            f"CELL INDEX: {idx}\n"
            f"ERROR TYPE: {e['ename']}\n"
            f"MESSAGE   : {e['evalue']}\n"
            f"{'-'*20} SOURCE CODE {'-'*20}\n"
            f"{e['source']}\n"
            f"{'-'*20} TRACEBACK {'-'*20}\n"
            f"{e['traceback']}\n"
        )
        report += entry

    _log(f"{'!'*60}\n", console=False)
    with open(log_path, "w", encoding="utf-8") as f: f.write(report)
    return log_path

# =============================================================================
# 2. Pure Execution (Code Injection + Verbose Output)
# =============================================================================

def run_notebook_pure(nb: nbformat.NotebookNode, 
                      workdir: str, 
                      timeout: int = 1800, 
                      cuda_device_id: Optional[int] = None,
                      extra_env: Optional[Dict[str, str]] = None) -> Tuple[nbformat.NotebookNode, List[Dict]]:
    """
    Executes the notebook using VerboseNotebookClient.
    Uses Code Injection for reliable environment variables.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    
    nb_run = deepcopy(nb)
    
    # --- [Code Injection] ---
    setup_code = [
        "# [SYSTEM] Auto-Injected Configuration",
        "import os",
        f"os.environ['OUTPUT_DIR'] = r'{workdir}'"
    ]
    
    if extra_env:
        _log(f"[EXEC] 💉 Injecting Code Variables: {list(extra_env.keys())}", console=False)
        for k, v in extra_env.items():
            setup_code.append(f"os.environ['{k}'] = {repr(v)}")
            
    # Inject Guard (Keep sys.exit patched for now, but watch logs!)
    setup_code.append("import sys; sys.exit = lambda *x: print('SYS.EXIT TRIGGERED (IGNORED)')")
    
    setup_cell = nbformat.v4.new_code_cell("\n".join(setup_code))
    nb_run.cells.insert(0, setup_cell)
    # ------------------------

    # Env Setup
    env_vars = os.environ.copy()
    if cuda_device_id is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)

    # Use Verbose Client
    client = VerboseNotebookClient(
        nb_run, 
        timeout=timeout, 
        kernel_name="python3", 
        allow_errors=True,
        resources={"metadata": {"path": workdir}},
        env=env_vars
    )
    
    try: 
        client.execute()
    except Exception as e: 
        _log(f"[EXEC] Kernel Exception (Partial): {e}", console=False)
    
    if len(client.nb.cells) > 0: client.nb.cells.pop(0) 
        
    errors = collect_cell_errors(client.nb)
    return client.nb, errors

# =============================================================================
# 3. Auto-Fix Logic (Unified)
# =============================================================================

# NOTE: Auto-fix logic is centralized in `notebook_autofix.py` and imported above
# as `attempt_fix_notebook`. This preserves legacy behavior (heuristics + LLM)
# while removing duplicated implementations.


# =============================================================================
# 4. Main Execution Loop
# =============================================================================

def execute_and_recover(nb_path: str, workdir: str, cfg: Dict[str, Any], mutable_indices: Optional[List[int]] = None, extra_env: Optional[Dict[str, str]] = None) -> Tuple[str, bool]:
    timeout = cfg["exec"]["timeout_seconds"]
    max_fixes = cfg["exec"]["max_fix_rounds"]
    current_nb_path = nb_path
    
    nb = nbformat.read(current_nb_path, as_version=4)
    _log(f"\n[EXEC] 🟢 Starting Run: {os.path.basename(current_nb_path)}", console=True)
    
    executed_nb, errors = run_notebook_pure(nb, workdir, timeout, cuda_device_id=None, extra_env=extra_env)
    
    if not errors:
        out_path = current_nb_path.replace(".ipynb", "_exec.ipynb")
        nbformat.write(executed_nb, out_path)
        _log(f"[EXEC] 🏁 Run Completed Successfully.", console=True)
        return out_path, True
    
    fix_round = 0
    current_nb_obj = executed_nb
    
    while errors and fix_round < max_fixes:
        fix_round += 1
        _log(f"\n[EXEC] 🔴 Errors Found (Round {fix_round}) - Initiating Recovery...", console=True)
        dump_error_log(workdir, errors, round_idx=fix_round)
        
        fixed_nb, changed, method = attempt_fix_notebook(current_nb_obj, errors, cfg, mutable_indices)
        if not changed:
            _log(f"[EXEC] ⚠️ Could not generate valid fix ({method}). Stopping.", console=False)
            break
        _log(f"[EXEC] 🔧 Applying Fix ({method}) -> Rerunning...", console=True)
        current_nb_obj, errors = run_notebook_pure(fixed_nb, workdir, timeout, cuda_device_id=None, extra_env=extra_env)
        if not errors:
            _log(f"[EXEC] ✅ Fixed successfully!", console=True)
            break

    final_out_path = nb_path.replace(".ipynb", "_exec.ipynb")
    nbformat.write(current_nb_obj, final_out_path)
    if errors:
        _log(f"[EXEC] ❌ Final Execution Failed.", console=True)
        return final_out_path, False
    return final_out_path, True