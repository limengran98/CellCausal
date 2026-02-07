#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for run_cellscientist.

Contains project-root helpers, IO helpers, streamed subprocess runner,
explicit path extraction logic, AND unified resource resolution.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import glob
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Project root / CWD
# =============================================================================


def project_root() -> str:
    """Return repository root.

    File layout: <repo>/cellscientist/pipeline/utils.py
    So repo root is two parents above this file.
    """
    return str(Path(__file__).resolve().parents[2])


def ensure_project_cwd() -> None:
    """Force CWD to repo root so relative paths behave consistently."""
    root = project_root()
    marker = os.path.join(root, "cellscientist")
    if not os.path.exists(marker):
        print(
            f"[WARN] Script location '{root}' may not be project root (missing 'cellscientist').\n"
            "       Continuing anyway and forcing CWD to script directory."
        )
    os.chdir(root)


# =============================================================================
# IO helpers
# =============================================================================


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"❌ Error: Config file not found: {path}")
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_text(path: str) -> str:
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def read_text_limited(path: str, *, max_chars: int = 120_000) -> str:
    raw = read_text(path)
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw
    head = raw[: max_chars // 2]
    tail = raw[-max_chars // 2 :]
    skipped = max(0, len(raw) - len(head) - len(tail))
    return head + f"\n\n... [TRUNCATED: {skipped} chars] ...\n\n" + tail


def read_head_tail_lines(
    path: str, *, head: int = 200, tail: int = 400, max_chars: int = 140_000
) -> str:
    raw = read_text(path)
    if not raw:
        return ""
    lines = raw.splitlines()
    if len(lines) <= head + tail:
        out = raw
    else:
        out = "\n".join(lines[:head]) + "\n\n... [SNIP] ...\n\n" + "\n".join(lines[-tail:])
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n\n... [TRUNCATED to {max_chars} chars] ..."
    return out


# =============================================================================
# UNIFIED RESOURCE RESOLUTION (H5 Path)
# =============================================================================

def resolve_h5_path_unified(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Centralized logic to find the H5 data file.
    Used by execution_workflow.py and review_workflow.py.
    """
    paths_cfg = (cfg.get("paths") or {}) if isinstance(cfg.get("paths"), dict) else {}

    # 1. Explicit Configuration (Highest Priority)
    explicit_h5 = paths_cfg.get("data_h5_path")
    if explicit_h5 and os.path.exists(str(explicit_h5)):
        cand = os.path.abspath(str(explicit_h5))
        print(f"[DATA] Found Stage 1 Data (Explicit): {cand}", flush=True)
        return cand

    data_root = paths_cfg.get("data_root")
    data_fname = paths_cfg.get("data_h5_filename")
    ds = cfg.get("dataset_name")

    # 2. Construct Search Candidates
    root_candidates: List[str] = []
    
    if data_root:
        if os.path.isabs(str(data_root)):
            root_candidates.append(str(data_root))
        else:
            root_candidates.append(os.path.abspath(str(data_root)))
            root_candidates.append(os.path.abspath(os.path.join(project_root(), str(data_root))))

    # Standard Fallbacks
    root_candidates.append(os.path.join(os.getcwd(), "data"))
    root_candidates.append(os.path.join(os.path.dirname(os.getcwd()), "data"))
    root_candidates.append(os.path.join(project_root(), "data"))
    
    # Deduplicate while preserving order
    root_candidates = sorted(list(set(root_candidates)), key=len, reverse=True)

    if data_fname:
        tried = []
        for root_abs in root_candidates:
            if not os.path.exists(root_abs): continue

            # Strategy A: <root>/<dataset>/<file>
            if ds:
                cand_a = os.path.join(root_abs, str(ds), str(data_fname))
                tried.append(cand_a)
                if os.path.exists(cand_a):
                    cand = os.path.abspath(cand_a)
                    print(f"[DATA] Found Stage 1 Data (Dataset Subdir): {cand}", flush=True)
                    return cand

            # Strategy B: <root>/<file>
            cand_b = os.path.join(root_abs, str(data_fname))
            tried.append(cand_b)
            if os.path.exists(cand_b):
                cand = os.path.abspath(cand_b)
                print(f"[DATA] Found Stage 1 Data (Root Dir): {cand}", flush=True)
                return cand
            
            # Strategy C: Recursive
            try:
                matches = glob.glob(os.path.join(root_abs, "**", str(data_fname)), recursive=True)
                if matches:
                    cand = os.path.abspath(matches[0])
                    print(f"[DATA] Found Stage 1 Data (Recursive): {cand}", flush=True)
                    return cand
            except Exception:
                pass

        print("[DATA][WARN] Could not resolve data H5 via data_root/filename. Tried:", flush=True)
        for p in tried:
            print(f"  - {p}", flush=True)

    # 3. Legacy Fallback (stage1_analysis_dir)
    s1_dir_str = paths_cfg.get("stage1_analysis_dir")
    if s1_dir_str:
        s1_path = os.path.abspath(s1_dir_str)
        final_ref_dir = s1_path

        # Auto-discovery if path is a parent dir
        if not os.path.exists(os.path.join(s1_path, "REFERENCE_DATA.h5")):
            if os.path.isdir(s1_path):
                subdirs = sorted([
                    os.path.join(s1_path, d) 
                    for d in os.listdir(s1_path) 
                    if os.path.isdir(os.path.join(s1_path, d)) and not d.startswith(".")
                ])
                if subdirs:
                    final_ref_dir = subdirs[-1]
                    print(f"[DATA] 🔎 Auto-detected latest reference run: {os.path.basename(final_ref_dir)}", flush=True)

        cand_h5 = os.path.join(final_ref_dir, "REFERENCE_DATA.h5")
        if os.path.exists(cand_h5):
            print(f"[DATA] Found Stage 1 Data (Legacy Explicit): {cand_h5}", flush=True)
            return cand_h5

        h5_files = glob.glob(os.path.join(final_ref_dir, "*.h5"))
        if h5_files:
            target_h5 = h5_files[0]
            print(f"[DATA] Found Stage 1 Data (Legacy Auto): {target_h5}", flush=True)
            return target_h5

    print(f"[DATA][WARN] No .h5 files found via any method.", flush=True)
    return None


# =============================================================================
# Path Extraction (Robust Strategy)
# =============================================================================


def find_recent_output_dir(base_dir: str, prefix: str, t_start: float) -> Optional[str]:
    """
    Fallback: Search for the most recently created directory in base_dir 
    that matches the prefix and was created AFTER t_start.
    """
    if not os.path.exists(base_dir):
        return None
    
    candidates = []
    # Allow generous clock skew/filesystem delay
    safe_start = t_start - 120.0 
    
    try:
        for name in os.listdir(base_dir):
            if not name.startswith(prefix) and not name.startswith("workspace_"):
                continue
            full_path = os.path.join(base_dir, name)
            if not os.path.isdir(full_path):
                continue
            
            try:
                # Use getmtime (modification time)
                mtime = os.path.getmtime(full_path)
                if mtime >= safe_start:
                    candidates.append((mtime, full_path))
            except OSError:
                continue
    except Exception:
        return None

    if not candidates:
        return None
    
    # Sort by time descending (newest first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_best_path_from_log(log_path: str, stage: str = None, phase: str = None, base_dir: str = "", t_start: float = 0.0) -> Optional[str]:
    """
    Extracts the output directory path for the current run.
    
    Args:
        log_path: Path to the log file
        stage: Stage name ("Experiment" or "Review") - preferred parameter
        phase: Legacy parameter for backward compatibility ("Phase 2" or "Phase 3")
        base_dir: Base directory for searching
        t_start: Start time for filtering recent outputs
    """
    # Handle backward compatibility
    if stage is None and phase is not None:
        # Map old phase names to new stage names
        if phase == "Phase 2":
            stage = "Experiment"
        elif phase == "Phase 3":
            stage = "Review"
    
    if stage is None:
        raise ValueError("Either 'stage' or 'phase' parameter must be provided")
    
    # --- Strategy 1: Explicit Pointer File (High Robustness) ---
    # Run pipeline creates a pointer file in the logs dir or base dir
    if base_dir:
        pointer_path = os.path.join(base_dir, "latest_run_pointer.json")
        if os.path.exists(pointer_path):
            try:
                with open(pointer_path, "r") as f:
                    data = json.load(f)
                    path = data.get("latest_trial_dir")
                    if path and os.path.exists(path):
                        return path
            except Exception:
                pass

    # --- Strategy 2: Log Parsing ---
    text = read_text(log_path)
    if text:
        if stage == "Experiment":
            # Match: [ARCHIVE] Saved run to: prompt_run_xxxx
            m_best = re.search(r"\[ARCHIVE\] Saved run to:\s*(.+)", text)
            if m_best:
                path_str = m_best.group(1).strip()
                if not os.path.isabs(path_str) and base_dir:
                    # Check nested prompt directory
                    prompt_path = os.path.join(base_dir, "prompt", path_str)
                    if os.path.exists(prompt_path): return prompt_path
                    # Check direct directory
                    direct_path = os.path.join(base_dir, path_str)
                    if os.path.exists(direct_path): return direct_path
                elif os.path.exists(path_str):
                    return path_str
            
            # Backup Match: Saved final state
            matches = list(re.finditer(r"\[EXEC\] Saved final state:\s*(.+)", text))
            if matches:
                last_file = matches[-1].group(1).strip()
                dir_path = os.path.dirname(last_file)
                if os.path.exists(dir_path): return dir_path
            
            # Backup Match: Trial Directory
            matches_trial = list(re.finditer(r"Trial:\s*(.+)", text))
            if matches_trial:
                trial_path = matches_trial[-1].group(1).strip()
                if os.path.exists(trial_path): return trial_path

        elif stage == "Review":
            # Match: Saved BEST Metrics to: ...
            m_best = re.search(r"Saved BEST Metrics to:\s*(.+)", text)
            if m_best:
                file_path = m_best.group(1).strip()
                dir_path = os.path.dirname(file_path)
                if os.path.exists(dir_path): return dir_path
            
            # Backup Match: Results saved to
            matches = list(re.finditer(r"\[Result\] Results saved to\s*(.+)", text))
            if matches:
                dir_path = matches[-1].group(1).strip()
                if os.path.exists(dir_path): return dir_path

    # --- Strategy 3: Filesystem Fallback ---
    if stage == "Experiment" and base_dir:
        # Try finding in generate_execution/prompt/prompt_run_*
        prompt_root = os.path.join(base_dir, "prompt")
        found = find_recent_output_dir(prompt_root, "prompt_run_", t_start) or find_recent_output_dir(prompt_root, "workspace_", t_start)
        if found: return found
        
    elif stage == "Review" and base_dir:
        # Try finding in review_feedback/review_run_*
        found = find_recent_output_dir(base_dir, "review_run_", t_start)
        if found: return found

    return None


# =============================================================================
# Tee logger
# =============================================================================


class TeeStream:
    """Write-through stream to multiple underlying streams."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]
        self.encoding = getattr(self.streams[0], "encoding", "utf-8") if self.streams else "utf-8"

    def write(self, data: str):
        if data is None:
            return 0
        n = 0
        for s in self.streams:
            try:
                n = s.write(data)
            except Exception:
                pass
        return n

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self.streams:
            try:
                if hasattr(s, "isatty") and s.isatty():
                    return True
            except Exception:
                continue
        return False

    def fileno(self):
        for s in self.streams:
            if hasattr(s, "fileno"):
                try:
                    return s.fileno()
                except Exception:
                    continue
        raise OSError("No underlying fileno")


def setup_logging(results_root: str) -> Tuple[str, str, Any]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(results_root, f"logs_{ts}")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"pipeline_{ts}.log")

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    log_fp = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_fp)  # type: ignore
    sys.stderr = TeeStream(sys.__stderr__, log_fp)  # type: ignore
    print(f"📝 Logging console output to: {log_path}")
    return logs_dir, log_path, log_fp


def append_phase_header(
    phase_fp, dataset: str, phase_name: str, cmd: List[str], cwd: str
) -> None:
    if not phase_fp:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase_fp.write("\n" + "=" * 88 + "\n")
    phase_fp.write(f"[{ts}] dataset={dataset} | {phase_name}\n")
    phase_fp.write(f"cwd={cwd}\n")
    phase_fp.write(f"cmd={' '.join(cmd)}\n")
    phase_fp.write("=" * 88 + "\n")
    phase_fp.flush()


def run_cmd_streamed(
    cmd: List[str],
    *,
    cwd: str,
    phase_fp=None,
    detail_fp=None,
    console_filter_fp=None,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    """Run a subprocess with streamed output to multiple destinations.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        phase_fp: File handle for phase-specific log (experiment.log/review.log)
        detail_fp: File handle for complete detail log (execution_detail.log)
        console_filter_fp: File handle for console output backup (console_output.log)
        extra_env: Additional environment variables
    
    Prefix-based Output Routing:
        Subprocess modules use special prefixes to control output routing:
        
        - [CELL_CONSOLE] prefix → Remove prefix and send to:
          * Console (stdout)
          * console_filter_fp (console_output.log)
          * detail_fp (execution_detail.log)
          * phase_fp (experiment.log/review.log)
        
        - [DETAIL] prefix → Remove prefix and send to:
          * detail_fp (execution_detail.log) only
          * phase_fp (experiment.log/review.log)
          * NOT to console (silent)
        
        - No prefix (raw output from third-party libraries, Python errors, etc.) → Send to:
          * detail_fp (execution_detail.log) only
          * phase_fp (experiment.log/review.log)
          * NOT to console (silent)
        
        Backward Compatibility:
        - When detail_fp is None: All output goes to console (old behavior)
    """
    env = os.environ.copy()
    if extra_env:
        for k, v in extra_env.items():
            if v is None:
                continue
            env[str(k)] = str(v)

    # Prefix constants
    CONSOLE_PREFIX = "[CELL_CONSOLE] "
    DETAIL_PREFIX = "[DETAIL] "

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        # Determine routing based on prefix
        show_console = False
        processed_line = line
        
        if line.startswith(CONSOLE_PREFIX):
            # [CELL_CONSOLE] → console + console_output.log + all logs
            show_console = True
            processed_line = line[len(CONSOLE_PREFIX):]  # Remove prefix
        elif line.startswith(DETAIL_PREFIX):
            # [DETAIL] → detail logs only (no console)
            show_console = False
            processed_line = line[len(DETAIL_PREFIX):]  # Remove prefix
        else:
            # No prefix → detail logs only (no console)
            show_console = False
            # Keep line as-is for detail log
        
        # Always write original line to detail log (complete capture with prefixes)
        if detail_fp:
            try:
                detail_fp.write(line)
                detail_fp.flush()
            except Exception:
                pass
        
        # Write to phase log (experiment.log/review.log) - original line with prefix
        if phase_fp:
            try:
                phase_fp.write(line)
                phase_fp.flush()
            except Exception:
                pass
        
        # Console output and console_output.log
        if show_console:
            # Write processed line (without prefix) to console
            try:
                sys.stdout.write(processed_line)
                sys.stdout.flush()
            except Exception:
                pass
            
            # Write processed line to console_output.log
            if console_filter_fp:
                try:
                    console_filter_fp.write(processed_line)
                    console_filter_fp.flush()
                except Exception:
                    pass
        elif detail_fp is None:
            # Backward compatibility: when no detail_fp, show everything
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


# =============================================================================
# Backward-compatible alias (runner imports)
# =============================================================================


def run_subprocess_streamed(
    cmd: List[str],
    *,
    cwd: str,
    phase_fp=None,
    detail_fp=None,
    console_filter_fp=None,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    """Compatibility wrapper.

    The original runners code used the name ``run_subprocess_streamed``.
    During refactor, the function was renamed to ``run_cmd_streamed``.
    We keep the old name to avoid breaking the runner entrypoint.
    """

    return run_cmd_streamed(
        cmd, 
        cwd=cwd, 
        phase_fp=phase_fp, 
        detail_fp=detail_fp,
        console_filter_fp=console_filter_fp,
        extra_env=extra_env
    )


# =============================================================================
# Misc filesystem helpers
# =============================================================================


def results_root_for_dataset(dataset_name: str) -> str:
    return os.path.abspath(os.path.join(project_root(), "results", dataset_name))


def safe_copy(src: str, dst_dir: str, dst_name: Optional[str] = None) -> Optional[str]:
    try:
        if not src or not os.path.exists(src):
            return None
        os.makedirs(dst_dir, exist_ok=True)
        name = dst_name or os.path.basename(src)
        dst = os.path.join(dst_dir, name)
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def export_notebook_as_py(nb_path: str, out_py_path: str) -> bool:
    """Extract code cells into a .py for convenience."""
    try:
        import nbformat

        nb = nbformat.read(nb_path, as_version=4)
        parts: List[str] = []
        for cell in nb.cells:
            if cell.get("cell_type") == "code":
                src = cell.get("source") or ""
                if src.strip():
                    parts.append(src.rstrip() + "\n")
        code = "\n\n# ---- cell ----\n\n".join(parts)
        with open(out_py_path, "w", encoding="utf-8") as f:
            f.write(code)
        return True
    except Exception:
        return False