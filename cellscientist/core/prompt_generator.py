# design_execution/prompt_generator.py
import os
import json
import re
import ast
import nbformat
from typing import Dict, Any, Tuple, Union, List, Optional
from pathlib import Path

# Import centralized LLM tools
from .llm_client import chat_text
from .external_knowledge_mirothink import retrieve_external_knowledge, knowledge_pack_to_markdown

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
# Robust JSON extraction / repair utilities
# =============================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)

def _strip_think(text: str) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()

def _escape_control_chars_in_strings(s: str) -> str:
    """
    Best-effort sanitizer: escapes raw control chars inside JSON string values.
    This specifically addresses the common failure where the model outputs real newlines
    inside a quoted JSON string (invalid JSON).
    """
    if not s:
        return s

    out: List[str] = []
    in_str = False
    esc = False

    for ch in s:
        o = ord(ch)

        if in_str:
            if esc:
                out.append(ch)
                esc = False
                continue

            if ch == "\\":  # begin escape
                out.append(ch)
                esc = True
                continue

            if ch == '"':
                out.append(ch)
                in_str = False
                continue

            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if o < 32:
                out.append(f"\\u{o:04x}")
                continue

            out.append(ch)
            continue

        # not in string
        if ch == '"':
            out.append(ch)
            in_str = True
            esc = False
            continue

        # drop raw control chars outside strings (except whitespace)
        if o < 32 and ch not in ("\n", "\r", "\t"):
            continue

        out.append(ch)

    return "".join(out)

def _extract_balanced_json_object(text: str) -> str:
    """
    Extract the first balanced {...} JSON object from text.
    Ignores braces within quoted strings.
    """
    if not text:
        return ""

    s = text
    start = s.find("{")
    if start < 0:
        return ""

    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":  # escape
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            esc = False
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return ""

def _extract_balanced_json_list(text: str) -> str:
    """
    Extract the first balanced [...] JSON list from text.
    Useful if the model returns a list of cells directly.
    """
    if not text:
        return ""
    s = text
    start = s.find("[")
    if start < 0:
        return ""

    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":  # escape
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            esc = False
            continue

        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return ""

def _candidates(text: str) -> List[str]:
    t = text.strip() if text else ""
    if not t:
        return []

    out: List[str] = []

    # 1) fenced json
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        out.append(m.group(1).strip())

    # 2) balanced object
    obj = _extract_balanced_json_object(t)
    if obj:
        out.append(obj)

    # 3) balanced list
    lst = _extract_balanced_json_list(t)
    if lst:
        out.append(lst)

    # 4) raw
    out.append(t)
    return out

def extract_json_from_text(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Robust extraction of JSON (dict or list) from LLM output.
    Handles: markdown fences, <think> blocks, raw JSON, and Python dict literals.
    """
    if not text:
        raise ValueError("LLM returned empty response.")

    cleaned = _strip_think(text)

    last_err: Exception = ValueError("unknown")
    for cand in _candidates(cleaned):
        cand2 = _escape_control_chars_in_strings(cand)

        # A) json.loads (dict or list)
        try:
            return json.loads(cand2)
        except Exception as e:
            last_err = e

        # B) python literal (single quotes / True/False/None)
        try:
            py = cand
            py = re.sub(r"\btrue\b", "True", py, flags=re.IGNORECASE)
            py = re.sub(r"\bfalse\b", "False", py, flags=re.IGNORECASE)
            py = re.sub(r"\bnull\b", "None", py, flags=re.IGNORECASE)
            return ast.literal_eval(py)
        except Exception as e:
            last_err = e

    preview = (cleaned[:800] + (" ... " if len(cleaned) > 800 else ""))
    raise ValueError(f"Could not parse JSON (json/ast). Last error: {last_err}. Preview:\n{preview}")

def _repair_to_json_with_llm(cfg: Dict[str, Any], raw_text: str, strict_json_only: bool) -> str:
    """
    Ask the LLM to reformat the previous output into valid JSON only.
    This is much cheaper than regenerating the whole notebook when only formatting is wrong.
    """
    if not raw_text:
        return ""

    sys = "You are a strict JSON formatter."
    if strict_json_only:
        user = (
            "Convert the following content into a SINGLE valid JSON object or JSON array. "
            "Output MUST be JSON only (no markdown, no <think>, no explanation).\n\n"
            "CONTENT:\n"
            f"{raw_text}"
        )
    else:
        user = (
            "Extract the JSON object/array from the following content and output ONLY that JSON. "
            "You may ignore any <think> or explanation text.\n\n"
            "CONTENT:\n"
            f"{raw_text}"
        )

    return chat_text(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        llm_config=cfg.get("llm", {}),
        temperature=0.0,
        max_tokens=8000,
        timeout=600,
    )

# =============================================================================
# Existing helper: load ideas
# =============================================================================

def _load_ideas_if_available() -> str:
    idea_path = os.environ.get("STAGE1_IDEA_PATH")
    if not idea_path or not os.path.exists(idea_path):
        return ""
    try:
        with open(idea_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ideas = data.get("ideas", [])
        if not ideas:
            return ""
        text = ["**Candidate Ideas:**"]
        for i, idea in enumerate(ideas):
            text.append(f"{i+1}. {idea.get('title')}: {idea.get('description')}")
        return "\n".join(text)
    except Exception:
        return ""

def _synthesize_strategy(cfg: Dict[str, Any], raw_ideas: str, debug_dir: str) -> str:
    strategy_path = os.path.join(debug_dir, "research_strategy.md")
    prompts_map = cfg.get("prompts", {})
    idea_prompt_data = prompts_map.get("idea", {})
    sys_content = idea_prompt_data.get("system", "Act as Lead Architect.")
    user_prompt = f"Here are the available ideas:\n{raw_ideas}\n\nWrite the Strategy Summary."

    _log("\n[GEN] 🧠 Synthesizing Research Strategy...", console=True)
    try:
        strategy = chat_text(
            [{"role": "system", "content": sys_content}, {"role": "user", "content": user_prompt}],
            llm_config=cfg.get("llm", {}),
            temperature=0.7,
            max_tokens=10000,
            timeout=600,
        )
        if not strategy:
            _log("[GEN] ⚠️ Strategy synthesis returned empty string.", console=True)
            return ""

        with open(strategy_path, "w", encoding="utf-8") as f:
            f.write(strategy)
        _log(f"[GEN] ✅ Strategy saved to: {strategy_path}", console=True)
        return strategy
    except Exception as e:
        _log(f"[GEN][ERROR] Strategy Synthesis Failed: {e}", console=True)
        return ""

# =============================================================================
# Main generator
# =============================================================================

def generate_notebook_content(
    cfg: Dict[str, Any],
    spec_path: str,
    debug_dir: str
) -> Tuple[nbformat.NotebookNode, str, str]:

    os.makedirs(debug_dir, exist_ok=True)
    import yaml

    # Load technical specification
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    # Resolve CUDA device string from config
    exec_cfg = cfg.get("exec", {})
    cuda_id = exec_cfg.get("cuda_device_id", 0)
    cuda_device_str = f"cuda:{cuda_id}"

    def _inject_vars(text: str) -> str:
        if not text:
            return ""
        return text.replace("${cuda_device_str}", cuda_device_str)

    # Prepare System Prompt
    sys_txt_raw = spec.get("system", "You are an expert.")
    sys_txt = _inject_vars(sys_txt_raw)

    # Prepare User Content (Spec)
    spec_content = {k: v for k, v in spec.items() if k != "system"}
    spec_dump_raw = yaml.safe_dump(spec_content, sort_keys=False)
    spec_dump = _inject_vars(spec_dump_raw)

    # [NEW] Optional external knowledge retrieval (MiroThink-derived).
    # Controlled by cfg['literature']['enabled']. Uses Serper for search and (optionally) Jina Reader for scraping.
    external_knowledge_md = ""
    external_knowledge_section = ""
    pack = None
    try:
        pack = retrieve_external_knowledge(
            cfg,
            context_text=spec_dump,
            stage="design",
            workspace_dir=debug_dir,
            tag="design",
        )
        external_knowledge_md = knowledge_pack_to_markdown(
            pack,
            max_chars=int(((cfg.get("literature") or {}) if isinstance(cfg.get("literature"), dict) else {}).get("prompt_max_chars", 6000) or 6000),
        )
        
        # Add knowledge retrieval summary to console
        if pack and pack.items:
            # Count by source type
            biokb_count = sum(1 for item in pack.items if item.source == "biokb")
            web_count = len(pack.items) - biokb_count
            
            # Format counts for display
            parts = []
            if web_count > 0:
                parts.append(f"{web_count} papers")
            if biokb_count > 0:
                parts.append(f"{biokb_count} pathways")
            
            if parts:
                summary = " and ".join(parts) if len(parts) == 2 else parts[0]
                _log(f"├─ BioKB/Web knowledge: {summary}", console=True)
        
        # [DIAGNOSTIC] Explicitly warn if no items found
        if not pack.items:
            external_knowledge_section = "\n> [WARN] External knowledge search executed but returned NO results. Check logs/API key.\n"
        elif not external_knowledge_md.strip():
            external_knowledge_section = "\n> [WARN] External knowledge search executed but Markdown generation failed.\n"
        else:
            external_knowledge_section = f"""
================================================================================
# EXTERNAL KNOWLEDGE (retrieved standards / papers / notes)
================================================================================
{external_knowledge_md}
"""
    except Exception as e:
        _log(f"[GEN][LIT][WARN] External knowledge retrieval failed: {e}", console=False)
        external_knowledge_section = f"\n> [ERROR] External knowledge retrieval failed: {e}\n"


    raw_ideas = _load_ideas_if_available()
    strategy_md = ""

    # 1. Strategy Branching (Idea vs Freestyle)
    if raw_ideas:
        strategy_md = _synthesize_strategy(cfg, raw_ideas, debug_dir)
        if strategy_md:
            # Calculate and display strategy size
            strategy_size_kb = len(strategy_md.encode('utf-8')) / 1024
            _log(f"├─ Strategy synthesized: {strategy_size_kb:.1f}KB", console=True)
            
            full_user_content = f"""
================================================================================
# PART 1: RESEARCH STRATEGY (MANDATORY IMPLEMENTATION)
================================================================================
**INSTRUCTION**: You are the Lead Engineer. Implement the model described below.

{strategy_md}

================================================================================
# PART 2: TECHNICAL SPECIFICATION
================================================================================
{spec_dump}
{external_knowledge_section}
"""
            _log("├─ Codegen mode: strategy-driven", console=True)
        else:
            strategy_md = "**Strategy**: Fallback to Freestyle Design (Synthesis Failed)."
            full_user_content = f"# TECHNICAL SPECIFICATION\n{spec_dump}\n\n{external_knowledge_section}"
            _log("├─ Codegen mode: fallback freestyle", console=True)
    else:
        strategy_md = (
            "## 🧠 Research Strategy (Freestyle)\n\n"
            "**Mode**: Expert Autonomous Design\n\n"
            "This model architecture was designed autonomously by the AI Architect based on the "
            "provided Technical Specifications, without external idea constraints. "
            "The focus is on robust baseline performance and standard best practices."
        )
        full_user_content = f"""
================================================================================
# TECHNICAL SPECIFICATION
================================================================================
**INSTRUCTION**: Design a high-performance model based on your expert judgement.
Follow these rules strictly:

{spec_dump}
{external_knowledge_section}
"""
        _log("├─ Codegen mode: freestyle/no-idea", console=True)
    messages = [{"role": "system", "content": sys_txt}, {"role": "user", "content": full_user_content}]


    # [NEW] Persist the exact prompts sent to the LLM for transparency/debugging.
    # This is especially useful to verify that the EXTERNAL KNOWLEDGE section was injected.
    try:
        with open(os.path.join(debug_dir, "prompt_system.txt"), "w", encoding="utf-8") as f:
            f.write(sys_txt or "")
        with open(os.path.join(debug_dir, "prompt_user.txt"), "w", encoding="utf-8") as f:
            f.write(full_user_content or "")
        if external_knowledge_md:
            with open(os.path.join(debug_dir, "external_knowledge_design.md"), "w", encoding="utf-8") as f:
                f.write(external_knowledge_md)
        _log(f"[GEN] 🧾 Debug prompts saved under: {debug_dir}", console=False)
    except Exception as e:
        _log(f"[GEN][WARN] Failed to save debug prompts: {e}", console=False)

    pb = cfg.get("prompt_branch", {}) or {}

    strict_json_only = bool(pb.get("strict_json_only", True))
    json_retries = int(pb.get("json_retries", 2) or 2)

    # Call LLM to generate notebook spec (text)
    raw_text = chat_text(
        messages,
        llm_config=cfg.get("llm", {}),
        temperature=float(pb.get("temperature", 0.4)),
        max_tokens=int(pb.get("max_tokens", 40000)),
        timeout=int(pb.get("timeout", 1200) or 1200),
    )

    attempts_raw: List[str] = [raw_text or ""]

    nb_json: Union[Dict[str, Any], List[Any], None] = None
    last_exc: Optional[Exception] = None

    for attempt in range(json_retries + 1):
        try:
            nb_json = extract_json_from_text(attempts_raw[-1])
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            if attempt >= json_retries:
                break

            _log(f"[GEN] ⚠️ JSON parse failed (attempt {attempt+1}/{json_retries+1}). Trying to repair output...", console=False)
            repaired = _repair_to_json_with_llm(cfg, attempts_raw[-1], strict_json_only=strict_json_only)
            attempts_raw.append(repaired or "")

    if nb_json is None or last_exc is not None:
        # Save raw outputs for debugging
        debug_path = os.path.join(debug_dir, "LLM_GEN_FAILURE.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(attempts_raw, 1):
                f.write(f"\n===== RAW ATTEMPT {i} =====\n")
                f.write(t or "EMPTY RESPONSE")
                f.write("\n")
        _log(f"[GEN] ❌ Debug info saved to {debug_path}", console=True)
        raise RuntimeError(f"Notebook Generation Failed (JSON Parse): {last_exc}")

    # Build notebook object
    nb = nbformat.v4.new_notebook()
    cells_data: List[Dict[str, Any]] = []
    hypergraph_data: Dict[str, Any] = {}

    if isinstance(nb_json, dict):
        cells_data = nb_json.get("cells") or (nb_json.get("notebook", {}) or {}).get("cells", []) or []
        hypergraph_data = nb_json.get("hypergraph", {}) or {}
    elif isinstance(nb_json, list):
        cells_data = nb_json

    # Calculate total code size
    total_code_size = sum(len((c.get("code", "") or "").encode('utf-8')) for c in cells_data if isinstance(c, dict))
    code_size_kb = total_code_size / 1024
    _log(f"└─ Notebook generated: {code_size_kb:.1f}KB | {len(cells_data)} cells", console=True)

    if hypergraph_data:
        nb.metadata["execution"] = {"hypergraph": hypergraph_data}

    for c in cells_data:
        if not isinstance(c, dict):
            continue
        cell = nbformat.v4.new_code_cell(c.get("code", "") or "")
        subtask_meta = {
            "id": c.get("id"),
            "name": c.get("name"),
            "purpose": c.get("purpose"),
        }
        cell.metadata["subtask"] = subtask_meta
        nb.cells.append(cell)

    # Strategy markdown at index 0
    if strategy_md:
        final_md = strategy_md
        if not strategy_md.strip().startswith("#"):
            final_md = f"# 🧠 Research Strategy\n\n{strategy_md}"
        md_cell = nbformat.v4.new_markdown_cell(final_md)
        nb.cells.insert(0, md_cell)

    return nb, full_user_content, strategy_md
