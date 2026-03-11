# -*- coding: utf-8 -*-
"""Notebook auto-fix utilities (unified).

This module merges the duplicated "auto-fix" logic previously living in:
- Design & Execution: prompt_executor.py (heuristics + LLM triple-quote dict output)
- Review & Optimization: executor_engine.py (heuristics + LLM JSON edits, with immutable cell handling)

Constraints:
- Preserve both behaviors and interfaces.
- Do not simplify; keep the original strategy branches.
"""

from __future__ import annotations

import os

from typing import Any, Dict, List, Optional, Tuple, Union
import ast
import hashlib
import json
import re
from copy import deepcopy

import nbformat

from .llm_client import chat_json, chat_text

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
# 0. Robust JSON/Python-Dict Extraction (Design-style)
# =============================================================================

def extract_edits_spec(text: str) -> Dict[str, Any]:
    """Surgical extraction of JSON/Dict from LLM output.

    This is the Design-style "Nuclear" parser v2.0:
    - Prefers ast.literal_eval so the LLM can return Python dicts with triple quotes.
    - Falls back to json.loads.

    Returns a dict (possibly with an "edits" list).
    """
    if not text:
        raise ValueError("LLM returned empty response.")

    t = text.strip()
    candidates: List[str] = []

    # Strategy 1: Markdown code fences (```json, ```python, or just ```)
    matches = re.findall(r"```(?:json|python)?\s*(.*?)\s*```", t, re.DOTALL)
    if matches:
        candidates.extend(sorted(matches, key=len, reverse=True))

    # Strategy 2: Python variable assignment (e.g., edits = {...})
    match_assign = re.search(r"\b(?:edits|result|data)\s*=\s*(\{.*\}|\[.*\])", t, re.DOTALL)
    if match_assign:
        candidates.append(match_assign.group(1))

    # Strategy 3: Outermost braces
    match_braces = re.search(r"(\{.*\})", t, re.DOTALL)
    if match_braces:
        candidates.append(match_braces.group(1))

    # Strategy 4: Raw text
    candidates.append(t)

    errors: List[str] = []
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue

        # A) Python literal (best for multi-line code with triple quotes)
        try:
            val = ast.literal_eval(cand)
            if isinstance(val, (dict, list)):
                return val if isinstance(val, dict) else {"edits": val}
        except Exception as e:
            errors.append(f"AST Error: {e}")

        # B) JSON
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                return {"edits": obj}
        except Exception as e:
            errors.append(f"JSON Error: {e}")

    preview = t[:500].replace("\n", " ") + "..."
    raise ValueError(
        f"CRITICAL PARSE FAILURE. Tried {len(candidates)} candidates. Errors: {errors}. Raw Preview: {preview}"
    )


# =============================================================================
# 1. Heuristic Patch Library (Merged)
# =============================================================================

def _get_heuristic_patch(evalue: str) -> Optional[str]:
    """Return a hardcoded patch snippet for common dumb errors."""
    msg = (evalue or "").lower()

    # --- Design heuristics ---
    if "input contains nan" in msg or "input contains infinity" in msg:
        return (
            "# [AUTO-FIX] Heuristic: Impute NaNs and Clip Infinity\n"
            "import numpy as np, pandas as pd\n"
            "from sklearn.impute import SimpleImputer\n"
            "print('[AUTO-FIX] Applying heuristic data cleaning...')\n"
            "def _clean_data(X):\n"
            "    if hasattr(X, 'fillna'): X = X.fillna(0)\n"
            "    X = np.nan_to_num(X, nan=0.0, posinf=None, neginf=None)\n"
            "    return X\n"
            "for _nm in ['X', 'X_train', 'X_test', 'y', 'y_train', 'y_test']:\n"
            "    if _nm in globals():\n"
            "        globals()[_nm] = _clean_data(globals()[_nm])\n"
        )

    if "could not convert string to float" in msg:
        return (
            "# [AUTO-FIX] Heuristic: Force Numeric Conversion\n"
            "import pandas as pd, numpy as np\n"
            "print('[AUTO-FIX] Filtering non-numeric columns...')\n"
            "def _force_numeric(df):\n"
            "    if isinstance(df, pd.DataFrame):\n"
            "        return df.select_dtypes(include=[np.number])\n"
            "    return df\n"
            "for _nm in ['X', 'X_train', 'X_test']:\n"
            "    if _nm in globals(): globals()[_nm] = _force_numeric(globals()[_nm])\n"
        )

    # --- Review heuristics (kept verbatim) ---
    if "could not convert string to float" in msg:
        return (
            "\n# [AUTO-FIX] Force numeric types\n"
            "import numpy as np\n"
            "for k in list(globals().keys()):\n"
            " if hasattr(globals()[k], 'select_dtypes'): globals()[k] = globals()[k].select_dtypes(include=[np.number])"
        )

    if "input x contains nan" in msg:
        return (
            "\n# [AUTO-FIX] Impute NaNs\n"
            "from sklearn.impute import SimpleImputer\n"
            "imp=SimpleImputer(strategy='median')\n"
            "for k in ['X','X_train']: \n"
            " if k in globals(): globals()[k] = imp.fit_transform(globals()[k])"
        )

    return None


def apply_heuristics(nb: nbformat.NotebookNode, errors: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
    """Apply heuristic patches.

    Returns (changed_count, patched_cell_indices).
    """
    changed_count = 0
    patched_indices: List[int] = []

    for e in errors:
        try:
            idx = int(e.get("cell_index"))
        except Exception:
            continue
        patch = _get_heuristic_patch(e.get("evalue", ""))
        if not patch:
            continue
        if 0 <= idx < len(nb.cells):
            cell = nb.cells[idx]
            if cell.get("cell_type") == "code":
                src = cell.get("source") or ""
                if patch not in src:
                    cell["source"] = src + "\n\n" + patch
                    changed_count += 1
                    patched_indices.append(idx)

    return changed_count, patched_indices


# =============================================================================
# 2. Edit Application (Hash Verified, Design-style)
# =============================================================================

def _compute_cell_hash(source: str) -> str:
    return hashlib.md5((source or "").strip().encode("utf-8")).hexdigest()


def apply_llm_edits(nb: nbformat.NotebookNode, edits: Union[List[Dict[str, Any]], Dict[str, Any]]) -> int:
    """Apply edits and return the number of effective changes."""
    effective_changes = 0

    # Normalize input: edits might be a list or a dict containing a list
    if isinstance(edits, dict) and "edits" in edits:
        edits = edits["edits"]

    if not isinstance(edits, list):
        return 0

    for ed in edits:
        try:
            idx = int(ed.get("cell_index"))
            new_src = ed.get("source")
            if idx >= 0 and idx < len(nb.cells) and new_src:
                old_src = nb.cells[idx].get("source") or ""
                if _compute_cell_hash(old_src) != _compute_cell_hash(new_src):
                    nb.cells[idx]["source"] = new_src
                    effective_changes += 1
        except Exception:
            pass

    return effective_changes


# =============================================================================
# 3. LLM Fix - Design Style (chat_text + Python Dict w/ Triple Quotes)
# =============================================================================

def llm_autofix_once_design(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    llm_cfg: Dict[str, Any],
    autofix_system_prompt: str,
) -> Tuple[nbformat.NotebookNode, bool]:
    """One-shot LLM fix using the Design pipeline format."""

    error_list = []
    for e in errors:
        error_list.append(
            {
                "cell_index": e.get("cell_index"),
                "error": f"{e.get('ename')}: {e.get('evalue')}",
                "traceback": (e.get("traceback") or "")[-3000:],
            }
        )

    indices = {e.get("cell_index") for e in errors}
    failing_cells = []
    for i in indices:
        try:
            ii = int(i)
        except Exception:
            continue
        if 0 <= ii < len(nb.cells):
            src = nb.cells[ii].get("source") or ""
            if len(src) > 15000:
                src = src[:15000] + "\n# ... [TRUNCATED]"
            failing_cells.append({"cell_index": ii, "source": src})

    user_payload = {
        "task": "Fix specific notebook cells. Maintain logic, fix runtime errors.",
        "errors": error_list,
        "failing_code": failing_cells,
        "instruction": (
            "Return a Python Dictionary with a key 'edits'. "
            "Use Python Triple Quotes (\"\"\") for the source code to handle newlines/quotes safely. "
            "Do NOT use JSON formatting for the code block if it contains many escapes."
        ),
    }

    enhanced_system_prompt = (
        f"{autofix_system_prompt}\n\n"
        "IMPORTANT: You are fixing a Jupyter Notebook cell.\n"
        "OUTPUT FORMAT: Return a **valid Python Dictionary** (parsable by ast.literal_eval).\n"
        "RULE: Use triple quotes (`\"\"\"`) for the code content to avoid JSON escaping errors.\n\n"
        "Example Response:\n"
        "```python\n"
        "{\n"
        "  'edits': [\n"
        "    {\n"
        "      'cell_index': 5,\n"
        "      'source': \"\"\"import torch\n"
        "def my_func():\n"
        "    return 'Success'\"\"\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )

    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]

    try:
        raw_text = chat_text(messages, llm_config=llm_cfg, temperature=0.0, timeout=600)
        spec = extract_edits_spec(raw_text)
    except Exception as e:
        _log(f"[FIX] LLM Call/Parse Failed: {e}", console=False)
        return nb, False

    edits = spec.get("edits") or []
    if not edits:
        _log("[FIX] LLM returned response but no 'edits' key found.", console=False)
        return nb, False

    changes = apply_llm_edits(nb, edits)
    return nb, (changes > 0)


# =============================================================================
# 4. LLM Fix - Review Style (chat_json, JSON edits)
# =============================================================================

def llm_autofix_request_review(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[nbformat.NotebookNode, bool]:
    """LLM fix request using the Review pipeline's JSON response contract."""

    err_context = [
        {"index": e.get("cell_index"), "msg": f"{e.get('ename')}: {e.get('evalue')}"}
        for e in errors
    ]

    failing_src = []
    for i in {e.get("cell_index") for e in errors}:
        try:
            ii = int(i)
        except Exception:
            continue
        if 0 <= ii < len(nb.cells):
            failing_src.append({"index": ii, "code": (nb.cells[ii].get("source") or "")[-2000:]})

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python Debugger. Fix the provided code errors.\n"
                "Return ONLY a JSON object: {\"edits\": [{\"cell_index\": <int>, \"source\": \"<full_new_code>\"}]}"
            ),
        },
        {"role": "user", "content": json.dumps({"errors": err_context, "code": failing_src})},
    ]

    try:
        resp = chat_json(messages, cfg, temperature=0.1)
    except Exception as e:
        _log(f"[FIX] LLM Error: {e}", console=False)
        return nb, False

    edits = resp.get("edits", [])
    if not isinstance(edits, list):
        edits = []

    changed = False
    for ed in edits:
        try:
            idx = int(ed.get("cell_index"))
            new_src = ed.get("source")
            if new_src is None:
                continue
            if idx < len(nb.cells):
                old_hash = hashlib.md5((nb.cells[idx].get("source") or "").encode("utf-8")).hexdigest()
                new_hash = hashlib.md5((new_src or "").encode("utf-8")).hexdigest()
                if old_hash != new_hash:
                    nb.cells[idx]["source"] = new_src
                    changed = True
        except Exception:
            pass

    return nb, changed


# =============================================================================
# 5. Public Review Interface (Preserved)
# =============================================================================

def attempt_fix_notebook(
    nb: nbformat.NotebookNode,
    errors: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    mutable_indices: Optional[List[int]] = None,
) -> Tuple[nbformat.NotebookNode, bool, str]:
    """Preserved Review interface.

    Returns (nb_next, changed, method).
    """

    nb_next = deepcopy(nb)
    target_errors = errors

    if mutable_indices is not None:
        target_errors = []
        for e in errors:
            try:
                idx = int(e.get("cell_index"))
            except Exception:
                continue
            if idx in mutable_indices:
                target_errors.append(e)
            else:
                _log(f"[FIX] 🚫 Skipping fix for Immutable Cell {idx} (Error: {e.get('ename')})", console=False)

        if not target_errors:
            return nb, False, "Aborted:ImmutableErrorsOnly"

    # Heuristics first
    h_changes, _patched = apply_heuristics(nb_next, target_errors)
    if h_changes > 0:
        return nb_next, True, "Heuristics"

    # Then Review-style JSON fix request
    nb_llm, l_changes = llm_autofix_request_review(nb_next, target_errors, cfg)
    if l_changes:
        return nb_llm, True, "LLM"

    return nb, False, "None"
