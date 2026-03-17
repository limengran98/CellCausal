# -*- coding: utf-8 -*-
"""External knowledge retrieval (MiroThink-derived).

What this module does
---------------------
Provide a low-coupling, optional "literature / external KB" module for CellScientist.

This file adapts the core ideas from MiroThinker's miroflow-tools MCP servers:
- serper_mcp_server.py (Serper Google Search API wrapper)
- searching_sogou_mcp_server.py (Jina Reader-based page scraping)

We intentionally *do not* require MCP runtime here. We reuse the same API calls
directly, so CellScientist can call this module as a plain library.

Configuration
-------------
Reads from cfg["literature"] and/or environment variables.

Required for web search:
- SERPER_API_KEY (or cfg["literature"]["serper_api_key"])
Optional:
- SERPER_BASE_URL (default: https://google.serper.dev)

Required for scraping (optional but recommended):
- JINA_API_KEY (or cfg["literature"]["jina_api_key"])
Optional:
- JINA_BASE_URL (default: https://r.jina.ai)

Logging & artifacts (new)
-------------------------
To make the retrieval visible in logs/workspace, this module can:
- print concise "[LIT]" logs (enabled by default when literature.enabled == True)
- write artifacts to:
  - cfg["paths"]["literature_dir"] (if set)
  - workspace_dir (if passed in)

Artifacts include:
- external_knowledge_<stage>.json / .md  (latest)
- runs/<timestamp>_<stage>_<hash>.json / .md (per-run snapshot)

Optionally, it can append to a rolling domain knowledge file:
- cfg["paths"]["literature_knowledge_json"] (if set)

Public API
----------
- retrieve_external_knowledge(cfg, context_text, stage, query_hint=None, workspace_dir=None, log_fn=None, tag=None) -> KnowledgePack
- knowledge_pack_to_markdown(pack, max_chars=6000) -> str
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

# Optional: reuse CellScientist LLM to summarize long pages
try:
    from .llm_client import chat_text
except Exception:  # pragma: no cover
    chat_text = None  # type: ignore

# Optional: BioKB module for semantic enrichment
try:
    from .bio_kb import (
        generate_biokb_semantic_table,
        persist_biokb_semantic_table,
        biokb_table_to_evidence_items,
    )
except Exception:  # pragma: no cover
    generate_biokb_semantic_table = None  # type: ignore
    persist_biokb_semantic_table = None  # type: ignore
    biokb_table_to_evidence_items = None  # type: ignore


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str = ""
    source: str = ""  # domain or provider
    published: str = ""  # best-effort
    scraped_excerpt: str = ""  # optional short excerpt / summary
    eid: str = ""  # Evidence ID (e.g., "B1", "L3")


@dataclass
class KnowledgePack:
    query: str
    stage: str
    generated_at: str
    items: List[EvidenceItem]
    provider: str = "mirothink_web"


# -----------------------------
# Utilities
# -----------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc
    except Exception:
        return ""


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else (s[: max(0, n - 3)] + "...")


def _lit_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    lit = cfg.get("literature") if isinstance(cfg, dict) else {}
    return lit if isinstance(lit, dict) else {}


def _logger(cfg: Dict[str, Any], log_fn: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    if log_fn is not None:
        return log_fn
    lit = _lit_cfg(cfg)
    # default: if literature.enabled, print logs unless explicitly disabled
    if bool(lit.get("enabled", False)) and bool(lit.get("log_to_console", True)):
        return lambda msg: print(msg, flush=True)
    return lambda _msg: None


# -----------------------------
# Serper Search (from MiroThinker serper_mcp_server.py)
# -----------------------------


def _serper_google_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: Optional[str] = None,
    num: int = 10,
    tbs: Optional[str] = None,
    page: Optional[int] = None,
    autocorrect: Optional[bool] = False,
    *,
    api_key: str,
    base_url: str = "https://google.serper.dev",
    timeout: int = 30,
) -> Dict[str, Any]:
    if not q or not q.strip():
        return {"success": False, "error": "q is empty", "organic": []}

    payload: Dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl, "num": num}
    if location:
        payload["location"] = location
    if tbs:
        payload["tbs"] = tbs
    if page is not None:
        payload["page"] = page
    if autocorrect is not None:
        payload["autocorrect"] = autocorrect

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    r = requests.post(f"{base_url.rstrip('/')}/search", json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    data["success"] = True
    return data


def _serper_error_payload(data: Any) -> Optional[str]:
    """Detect common 'HTTP 200 but error in JSON payload' patterns."""
    if not isinstance(data, dict):
        return None
    # Common gateway formats
    if isinstance(data.get("error"), (str, dict)):
        return str(data.get("error"))
    if "errors" in data:
        return str(data.get("errors"))
    if "message" in data and ("statusCode" in data or "status" in data):
        return f"{data.get('message')} (status={data.get('statusCode') or data.get('status')})"
    return None


def _fallback_queries(original_q: str, cfg: Dict[str, Any], stage: str) -> List[str]:
    """Generate a small set of safer retry queries if the first query yields 0 results."""
    lit = _lit_cfg(cfg)
    base_kw = (lit.get("task_keywords") or "").strip()
    stage_kw = "design draft" if stage == "design" else "review feedback"
    academic_bias = ' (paper OR arxiv OR "journal" OR doi OR pubmed OR "technical report" OR "specification")'

    generic = "perturbation response prediction drug gene expression"
    q2 = " ".join([p for p in [base_kw, generic, stage_kw] if p]).strip() + academic_bias
    q3 = " ".join([p for p in [base_kw, generic] if p]).strip() + academic_bias

    out: List[str] = []
    for q in [original_q, q2, q3]:
        q = (q or "").strip()
        if not q:
            continue
        if q not in out:
            out.append(q)
    return out


# -----------------------------
# Jina Reader Scrape (from MiroThinker searching_sogou_mcp_server.py)
# -----------------------------


def _strip_markdown_links(md: str) -> str:
    # Very small helper: remove [text](url) to keep content compact.
    import re

    return re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", md)


def _jina_scrape_url(
    url: str,
    *,
    api_key: str,
    base_url: str = "https://r.jina.ai",
    timeout: int = 60,
    max_chars: int = 12000,
    fallback_snippet: str = "",
) -> str:
    """Scrape a URL via Jina Reader.

    Args:
        url: Target URL to scrape.
        api_key: Jina API key.
        base_url: Jina Reader base URL.
        timeout: Request timeout in seconds.
        max_chars: Maximum characters to return.
        fallback_snippet: Raw search snippet used as fallback when Jina is
            unavailable (402 Payment Required or timeout).

    Returns:
        Scraped page content, or a fallback snippet string on 402/timeout.
    """
    if not url or not url.startswith(("http://", "https://")):
        return f"Invalid URL: {url}"

    # Avoid duplicate prefix
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]

    # HF datasets/spaces: usually too large / not intended for scraping summaries
    if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
        return "Skipping HuggingFace dataset/space page (often too large for scraping)."

    if not api_key:
        return "JINA_API_KEY not set; scraping disabled."

    jina_url = f"{base_url.rstrip('/')}/{url}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get(jina_url, headers=headers, timeout=timeout)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        print(
            "[LIT] ⚠️ Jina scrape failed (402/timeout), using raw search snippets as fallback",
            flush=True,
        )
        snippet = fallback_snippet or "(no snippet available)"
        return f"[Jina unavailable] Raw snippet: {snippet}"
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 402:
            print(
                "[LIT] ⚠️ Jina scrape failed (402/timeout), using raw search snippets as fallback",
                flush=True,
            )
            snippet = fallback_snippet or "(no snippet available)"
            return f"[Jina unavailable] Raw snippet: {snippet}"
        raise
    content = _strip_markdown_links(r.text.strip())
    if max_chars and len(content) > max_chars:
        content = content[:max_chars] + "\n... (truncated)"
    return content


# -----------------------------
# Query builder
# -----------------------------


def _build_query(context_text: str, stage: str, cfg: Dict[str, Any], query_hint: Optional[str]) -> str:
    lit = _lit_cfg(cfg)
    base_kw = (lit.get("task_keywords") or "").strip()
    stage_kw = "design draft" if stage == "design" else "review feedback"

    # Users can override with cfg["literature"]["query"].
    forced = (lit.get("query") or "").strip()
    if forced:
        return forced

    # A light heuristic: use hint if supplied; else take keywords from context.
    # Robustness: aggressively filter placeholder/config tokens (e.g., dataset_name, STAGE1_H5_PATH).
    term_limit = int(lit.get("query_term_limit", 12) or 12)
    stop = {
        # common placeholders
        "dataset_name",
        "dataset",
        "resources",
        "resource",
        "h5_file",
        "h5",
        "path",
        "paths",
        "stage1_h5_path",
        "stage2_h5_path",
        "stage3_h5_path",
        "idea_file",
        "idea_json",
        "prompt",
        "prompts",
        "yaml",
        "json",
        "config",
        "configs",
        "workspace",
        "workdir",
        "output",
        "outputs",
        "intermediate",
        "debug",
        # env-like
        "stage1",
        "stage2",
        "stage3",
        "staging",
    }

    def _is_noise_token(t: str) -> bool:
        tl = t.lower()
        if tl in stop:
            return True
        # env vars / placeholders (ALL_CAPS_WITH_UNDERSCORES)
        if t.isupper() and "_" in t:
            return True
        if tl.endswith("_path") or tl.endswith("_file") or tl.endswith("_dir"):
            return True
        # too generic / likely template variable
        if "${" in t or "}" in t or "{" in t:
            return True
        return False

    def _sanitize_terms(raw: str) -> str:
        """Remove notebook/code scaffold noise from hint/context text.

        Review-stage prompts may leak notebook headers and import snippets
        (e.g., "CELL INDEX READ-ONLY CONTEXT import sys ..."). We apply the
        same sanitizer to both explicit query hints and context-derived terms.
        """
        import re

        toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", raw or "")
        deny = {
            "cell", "index", "read", "only", "context", "target", "optimize",
            "import", "from", "code", "markdown", "unnamed", "setup", "data", "loading",
            "read-only",
            # Common module names frequently leaked from notebook source
            "sys", "os", "h5py", "numpy", "pandas", "torch", "utils", "functional",
            # Common prompt/template leakage
            "legacy", "contract", "guardrails", "locked", "section", "sections", "agent_mode_virtual_cell_context",
        }
        keep: List[str] = []
        seen = set()
        for t in toks:
            tl = t.lower()
            if tl in deny or _is_noise_token(t):
                continue
            # Strongly prefer domain-ish words over code-ish tokens.
            if tl.startswith(("nn", "plt", "df", "np", "pd")):
                continue
            if tl in seen:
                continue
            seen.add(tl)
            keep.append(t)
            if len(keep) >= term_limit:
                break
        return " ".join(keep)

    if query_hint and query_hint.strip():
        hint = _sanitize_terms(query_hint.strip())
    else:
        # Apply the same sanitizer for context-derived queries to avoid
        # leaking notebook scaffold tokens into search terms.
        hint = _sanitize_terms(context_text or "")

    if not hint:
        # Last-resort safe fallback to avoid empty or code-polluted queries.
        hint = "single-cell perturbation response modeling"

    # Prefer academic-ish results by adding common anchors.
    academic_bias = ' (paper OR arxiv OR "journal" OR doi OR pubmed OR "technical report" OR "specification")'
    parts = [p for p in [base_kw, hint, stage_kw] if p]
    return " ".join(parts).strip() + academic_bias


# -----------------------------
# Summarization (optional)
# -----------------------------


def _summarize_if_needed(cfg: Dict[str, Any], text: str, title: str, url: str, max_chars: int, circuit_open: bool = False) -> str:
    """
    Summarize text using LLM if possible, otherwise truncate.
    Robustness Fix: If circuit_open is True, or LLM fails/returns empty, immediately fall back to truncation.
    """
    truncated_fallback = (text[:max_chars] + "\n... (truncated)") if len(text) > max_chars else text

    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    
    # [ROBUSTNESS] If circuit breaker is open, skip LLM entirely
    if circuit_open:
        return truncated_fallback

    # If CellScientist LLM is available and enabled, summarize.
    lit = _lit_cfg(cfg)
    if lit.get("use_llm_summarizer", True) and chat_text is not None:
        prompt = (
            "Summarize the following source for scientific RAG use.\n"
            "- Focus on methods, claims, assumptions, and limitations.\n"
            "- Output <= 12 bullet points.\n\n"
            f"Title: {title}\nURL: {url}\n\n"
            "CONTENT:\n"
        )
        try:
            summary = chat_text(
                [
                    {"role": "system", "content": "You are a precise scientific literature summarizer."},
                    {"role": "user", "content": prompt + text[:30000]},
                ],
                cfg=cfg,
                temperature=0.2,
                max_tokens=1200,
                timeout=120,
            )
            summary = (summary or "").strip()
            
            # [ROBUSTNESS] Crucial fix: check for empty return
            if summary:
                return summary
            else:
                # Log happens in caller, just return fallback here implies failure
                return truncated_fallback
                
        except Exception:
            return truncated_fallback

    # Fallback: truncate
    return truncated_fallback


# -----------------------------
# Persistence helpers (new)
# -----------------------------


def _json_safe_pack(pack: KnowledgePack, max_excerpt_chars: int = 6000) -> Dict[str, Any]:
    """Make a JSON-safe dict, truncating very long excerpts to keep files readable."""
    d = asdict(pack)
    for it in d.get("items") or []:
        ex = (it.get("scraped_excerpt") or "")
        if ex and len(ex) > max_excerpt_chars:
            it["scraped_excerpt"] = ex[:max_excerpt_chars] + "\n... (truncated)"
    return d


def _persist_pack_artifacts(
    pack: KnowledgePack,
    cfg: Dict[str, Any],
    stage: str,
    base_dir: str,
    *,
    tag: Optional[str] = None,
    md_max_chars: int = 12000,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Persist artifacts under base_dir; returns created paths."""
    if not base_dir:
        return {}
    log = log or (lambda _m: None)
    out: Dict[str, str] = {}

    safe_stage = "".join(ch for ch in (stage or "stage") if ch.isalnum() or ch in ("-", "_")) or "stage"
    safe_tag = ""
    if tag:
        safe_tag = "".join(ch for ch in str(tag) if ch.isalnum() or ch in ("-", "_"))[:48]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = _sha1(f"{pack.query}|{pack.generated_at}|{safe_stage}")[:10]

    runs_dir = os.path.join(base_dir, "runs")
    _ensure_dir(runs_dir)

    run_prefix = f"{stamp}_{safe_stage}_{run_hash}"
    if safe_tag:
        run_prefix = f"{run_prefix}_{safe_tag}"

    json_path = os.path.join(runs_dir, f"{run_prefix}.json")
    md_path = os.path.join(runs_dir, f"{run_prefix}.md")
    latest_json = os.path.join(base_dir, f"external_knowledge_{safe_stage}.json")
    latest_md = os.path.join(base_dir, f"external_knowledge_{safe_stage}.md")

    try:
        _write_json(json_path, _json_safe_pack(pack))
        _write_json(latest_json, _json_safe_pack(pack))
        out["json"] = json_path
        out["latest_json"] = latest_json
    except Exception as e:
        log(f"[LIT][WARN] Failed to write JSON artifacts: {e}")

    try:
        md = knowledge_pack_to_markdown(pack, max_chars=md_max_chars)
        _write_text(md_path, md)
        _write_text(latest_md, md)
        out["md"] = md_path
        out["latest_md"] = latest_md
    except Exception as e:
        log(f"[LIT][WARN] Failed to write Markdown artifacts: {e}")

    return out


def _append_domain_knowledge(cfg: Dict[str, Any], pack: KnowledgePack, log: Callable[[str], None]) -> None:
    """Append pack to cfg['paths']['literature_knowledge_json'] (rolling list)."""
    paths = cfg.get("paths") if isinstance(cfg, dict) else {}
    paths = paths if isinstance(paths, dict) else {}
    out_path = (paths.get("literature_knowledge_json") or "").strip()
    if not out_path:
        return

    lit = _lit_cfg(cfg)
    if not bool(lit.get("append_domain_knowledge", True)):
        return

    out_path = os.path.abspath(out_path)

    max_entries = int(lit.get("domain_knowledge_max_entries", 80) or 80)
    max_excerpt = int(lit.get("domain_excerpt_max_chars", 3000) or 3000)

    try:
        if os.path.exists(out_path):
            data = _read_json(out_path)
            if not isinstance(data, list):
                data = []
        else:
            data = []

        data.append(_json_safe_pack(pack, max_excerpt_chars=max_excerpt))
        if len(data) > max_entries:
            data = data[-max_entries:]
        _write_json(out_path, data)
    except Exception as e:
        log(f"[LIT][WARN] Failed to append domain knowledge json: {e}")


# -----------------------------
# Public function
# -----------------------------


def retrieve_external_knowledge(
    cfg: Dict[str, Any],
    context_text: str,
    stage: str,
    query_hint: Optional[str] = None,
    *,
    workspace_dir: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    tag: Optional[str] = None,
) -> KnowledgePack:
    """Retrieve and (optionally) scrape external knowledge for the given stage.

    stage: "design" | "review" | any string for tagging

    workspace_dir:
      If provided, artifacts will also be written under:
      <workspace_dir>/external_knowledge/
    """
    lit = _lit_cfg(cfg)
    log = _logger(cfg, log_fn)

    enabled = bool(lit.get("enabled", False))
    if not enabled:
        return KnowledgePack(query="", stage=stage, generated_at=_now_iso(), items=[], provider="disabled")

    # Determine artifact directories
    paths = cfg.get("paths") if isinstance(cfg, dict) else {}
    paths = paths if isinstance(paths, dict) else {}
    literature_dir = (paths.get("literature_dir") or lit.get("literature_dir") or "").strip()
    literature_dir = os.path.abspath(literature_dir) if literature_dir else ""
    workspace_out = os.path.join(os.path.abspath(workspace_dir), "external_knowledge") if workspace_dir else ""

    # ============================================================
    # PHASE 1: Build BioKB Pack (if enabled)
    # ============================================================
    biokb_items: List[EvidenceItem] = []
    bio_kb_cfg = lit.get("bio_kb", {}) if isinstance(lit, dict) else {}
    biokb_enabled = bool(bio_kb_cfg.get("enabled", False))  # Default disabled for backward compatibility
    
    if biokb_enabled and generate_biokb_semantic_table:
        try:
            log("[LIT] 🧬 Building BioKB pack...")
            semantic_table = generate_biokb_semantic_table(cfg, stage, log)
            
            # Persist semantic table
            if workspace_out:
                persist_biokb_semantic_table(semantic_table, workspace_dir, log)
            if literature_dir:
                persist_biokb_semantic_table(semantic_table, literature_dir, log)
            
            # Convert to evidence items
            biokb_items_raw = biokb_table_to_evidence_items(semantic_table)
            for item_dict in biokb_items_raw:
                biokb_items.append(EvidenceItem(
                    title=item_dict["title"],
                    url=item_dict["url"],
                    snippet=item_dict["snippet"],
                    source=item_dict["source"],
                    published=item_dict["published"],
                    scraped_excerpt=item_dict["scraped_excerpt"],
                    eid=item_dict["eid"]
                ))
            
            log(f"[LIT] ✅ BioKB pack: {len(biokb_items)} evidence items")
        except Exception as e:
            log(f"[LIT][WARN] BioKB generation failed: {e}")
            # Add placeholder to make failure visible
            biokb_items.append(EvidenceItem(
                title="BioKB Generation Failed",
                url="",
                snippet=f"BioKB module encountered an error: {str(e)[:200]}",
                source="biokb",
                eid="B0"
            ))

    # ============================================================
    # PHASE 2: Build Web Literature Pack
    # ============================================================
    
    # Resolve keys (do NOT log secrets)
    serper_key = (lit.get("serper_api_key") or os.environ.get("SERPER_API_KEY") or "").strip()
    serper_base = (lit.get("serper_base_url") or os.environ.get("SERPER_BASE_URL") or "https://google.serper.dev").strip()
    jina_key = (lit.get("jina_api_key") or os.environ.get("JINA_API_KEY") or "").strip()
    jina_base = (lit.get("jina_base_url") or os.environ.get("JINA_BASE_URL") or "https://r.jina.ai").strip()

    web_items: List[EvidenceItem] = []
    
    if not serper_key:
        log("[LIT] ⚠️ SERPER_API_KEY missing. Web search disabled (BioKB only).")
        web_items.append(EvidenceItem(
            title="SERPER_API_KEY missing",
            url="",
            snippet="Set SERPER_API_KEY or cfg['literature']['serper_api_key'].",
            eid="L0"
        ))
    else:
        # Build query
        q = _build_query(context_text=context_text, stage=stage, cfg=cfg, query_hint=query_hint)

        # Cache handling + Hierarchical limits
        cache_days = int(lit.get("cache_days", 30) or 30)
        # Hierarchical limits (with backward compatibility)
        max_papers = max(1, int(lit.get("max_papers", 12) or 12))  # Broad search limit
        deepread_max = int(lit.get("deepread_max_items") or min(max_papers, 8))
        summarize_max = int(lit.get("summarize_max_items") or min(deepread_max, 6))
        inject_max = int(lit.get("inject_max_items", 5) or 5)
        
        max_abstract_chars = int(lit.get("max_abstract_chars", 1200) or 1200)
        md_max_chars = int(lit.get("artifact_md_max_chars", 12000) or 12000)

        cache_dir = os.path.join(literature_dir, "cache") if literature_dir else ""
        cache_key = _sha1(json.dumps({"q": q, "stage": stage, "max": max_papers}, ensure_ascii=False))
        cache_path = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else ""

        log(f"[LIT] 🔎 stage={stage} | query={_truncate(q, 160)}")
        log(f"[LIT] 📊 Hierarchical limits: broad={max_papers}, deepread={deepread_max}, summarize={summarize_max}, inject={inject_max}")

        def _persist_everywhere(pack: KnowledgePack) -> None:
            if literature_dir:
                _persist_pack_artifacts(pack, cfg, stage, literature_dir, tag=tag, md_max_chars=md_max_chars, log=log)
            if workspace_out:
                _persist_pack_artifacts(pack, cfg, stage, workspace_out, tag=tag, md_max_chars=md_max_chars, log=log)
            _append_domain_knowledge(cfg, pack, log)

        # Respect "use_existing_only"
        if cache_path and lit.get("use_existing_only", False):
            if os.path.exists(cache_path):
                cached = _read_json(cache_path)
                cached_pack = _pack_from_json(cached)
                log(f"[LIT] ♻️ Using cached results (use_existing_only=true): {cache_path}")
                log(f"[LIT] ✅ Search results: {len(cached_pack.items)} items (cached)")
                web_items = cached_pack.items
            else:
                log(f"[LIT] ⚠️ Cache miss while use_existing_only=true: {cache_path}")
                web_items.append(EvidenceItem(
                    title="Cache miss",
                    url="",
                    snippet="use_existing_only=true but cache not found",
                    eid="L0"
                ))
        else:
            # Fresh-enough cache
            use_cached = False
            if cache_path and os.path.exists(cache_path):
                try:
                    cached = _read_json(cache_path)
                    ts = cached.get("generated_at") or ""
                    if ts:
                        gen_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if datetime.now(gen_time.tzinfo) - gen_time <= timedelta(days=cache_days):
                            cached_pack = _pack_from_json(cached)
                            log(f"[LIT] ♻️ Using cached results: {cache_path}")
                            log(f"[LIT] ✅ Search results: {len(cached_pack.items)} items (cached)")
                            if len(cached_pack.items) > 0:
                                web_items = cached_pack.items
                                use_cached = True
                            else:
                                log("[LIT][WARN] Cache is fresh but contains 0 items. Refreshing search...")
                except Exception:
                    pass

            if not use_cached:
                # Search (with small fallback set)
                gl = (lit.get("search_gl") or "us").strip()
                hl = (lit.get("search_hl") or "en").strip()
                location = lit.get("search_location")
                tbs = lit.get("search_tbs")

                data: Dict[str, Any] = {}
                used_q = q

                for attempt_q in _fallback_queries(q, cfg, stage):
                    used_q = attempt_q
                    try:
                        data = _serper_google_search(
                            q=attempt_q,
                            gl=gl,
                            hl=hl,
                            location=location,
                            num=max(3, min(30, max_papers)),
                            tbs=tbs,
                            api_key=serper_key,
                            base_url=serper_base,
                        )
                    except Exception as e:
                        log(f"[LIT][WARN] Search failed for query (attempt): {e}")
                        continue

                    err = _serper_error_payload(data)
                    if err:
                        log(f"[LIT] ❌ Serper error payload detected: {err}")
                        web_items.append(EvidenceItem(
                            title="Search provider error",
                            url="",
                            snippet=err,
                            source="serper",
                            eid="L0"
                        ))
                        break

                    organic = data.get("organic") or []
                    for idx, r in enumerate(organic[:max_papers], 1):
                        title = str(r.get("title") or "")
                        url = str(r.get("link") or r.get("url") or "")
                        snippet = str(r.get("snippet") or "")
                        published = str(r.get("date") or "")
                        web_items.append(EvidenceItem(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source=_domain(url),
                            published=published,
                            eid=""  # Will assign L* IDs later
                        ))

                    if web_items:
                        break

                # Log results
                if used_q != q:
                    log(f"[LIT] 🔁 Fallback query used: {_truncate(used_q, 160)}")
                log(f"[LIT] ✅ Search results: {len(web_items)} items")

                # Robustness: emit stub if no results
                emit_stub = bool(lit.get("emit_noresult_stub", True))
                if not web_items and emit_stub:
                    web_items.append(EvidenceItem(
                        title="No search results",
                        url="",
                        snippet=f"Search returned 0 results. Query used: {used_q}",
                        source="serper",
                        eid="L0"
                    ))

                # Optional scrape+summarize (TIER 2: deepread_max)
                circuit_breaker_threshold = 2
                consecutive_failures = 0
                circuit_open = False

                if bool(lit.get("scrape", True)) and web_items:
                    if not jina_key:
                        log("[LIT] ⚠️ JINA_API_KEY missing. Skipping page scraping (snippets only).")
                    else:
                        # Only scrape up to deepread_max items
                        items_with_urls = [it for it in web_items if (it.url or "").startswith(("http://", "https://"))]
                        items_to_scrape = items_with_urls[:deepread_max]
                        log(f"[LIT] 📄 Scraping {len(items_to_scrape)}/{len(web_items)} items (deepread limit)")
                        
                        for idx, it in enumerate(items_to_scrape, start=1):
                            try:
                                raw = _jina_scrape_url(it.url, api_key=jina_key, base_url=jina_base, max_chars=30000, fallback_snippet=it.snippet)

                                # TIER 3: Only summarize up to summarize_max
                                if idx <= summarize_max:
                                    excerpt = _summarize_if_needed(cfg, raw, it.title, it.url, max_chars=max_abstract_chars, circuit_open=circuit_open)
                                else:
                                    # Truncate without summarization
                                    excerpt = (raw[:max_abstract_chars] + "\n... (truncated)") if len(raw) > max_abstract_chars else raw

                                it.scraped_excerpt = excerpt
                                log(f"[LIT] 📄 scraped {idx}/{len(items_to_scrape)}: {_truncate(it.title, 72)}")
                                consecutive_failures = 0

                            except Exception as e:
                                it.scraped_excerpt = f"(scrape failed) {e}"
                                log(f"[LIT][WARN] scrape failed {idx}/{len(items_to_scrape)}: {e}")
                                consecutive_failures += 1

                            if consecutive_failures >= circuit_breaker_threshold:
                                circuit_open = True

                # Cache if we have real results
                has_real_results = any((it.url or "").startswith(("http://", "https://")) for it in web_items)
                if cache_path and literature_dir and has_real_results:
                    try:
                        # Create temporary pack for caching
                        temp_pack = KnowledgePack(
                            query=used_q,
                            stage=stage,
                            generated_at=_now_iso(),
                            items=web_items,
                            provider="mirothink_web"
                        )
                        _write_json(cache_path, _json_safe_pack(temp_pack))
                        log(f"[LIT] 💾 Cached: {cache_path}")
                    except Exception as e:
                        log(f"[LIT][WARN] Failed to write cache: {e}")

    # ============================================================
    # PHASE 3: Merge BioKB + Web Items
    # ============================================================
    
    # Assign L* IDs to web items (if not already assigned)
    for idx, item in enumerate(web_items, 1):
        if not item.eid:
            item.eid = f"L{idx}"
    
    # TIER 4: Only inject up to inject_max items total
    # Priority: BioKB first, then Web (sorted by relevance if needed)
    # TODO: Implement relevance-based sorting for better prioritization
    all_items = biokb_items + web_items
    inject_max = int(lit.get("inject_max_items", 5) or 5)
    
    # For now, simple truncation; see TODO above for future enhancement
    if len(all_items) > inject_max:
        log(f"[LIT] ✂️ Injecting {inject_max}/{len(all_items)} items (inject limit)")
        all_items = all_items[:inject_max]
    
    # Build final pack
    provider = "biokb+mirothink_web" if biokb_items else "mirothink_web"
    pack = KnowledgePack(
        query=q if web_items else "",
        stage=stage,
        generated_at=_now_iso(),
        items=all_items,
        provider=provider
    )
    
    # Persist final pack
    if literature_dir:
        _persist_pack_artifacts(pack, cfg, stage, literature_dir, tag=tag, md_max_chars=md_max_chars, log=log)
    if workspace_out:
        _persist_pack_artifacts(pack, cfg, stage, workspace_out, tag=tag, md_max_chars=md_max_chars, log=log)
    _append_domain_knowledge(cfg, pack, log)
    
    return pack


def _pack_from_json(d: Dict[str, Any]) -> KnowledgePack:
    """Parse a cached JSON pack with schema tolerance.

    Over time, different runs may store slightly different keys (e.g. Serper returns
    `link` vs `url`; older caches may use `name` instead of `title`). We therefore
    parse items defensively to avoid returning an empty pack due to minor schema drift.
    """
    raw_items = d.get("items") or []
    items: List[EvidenceItem] = []
    if isinstance(raw_items, list):
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or it.get("name") or it.get("heading") or "")
            url = str(it.get("url") or it.get("link") or it.get("href") or "")
            snippet = str(it.get("snippet") or it.get("summary") or it.get("description") or "")
            source = str(it.get("source") or it.get("domain") or _domain(url) or "")
            published = str(it.get("published") or it.get("date") or "")
            scraped_excerpt = str(it.get("scraped_excerpt") or it.get("excerpt") or it.get("notes") or "")
            eid = str(it.get("eid") or "")  # Evidence ID
            items.append(
                EvidenceItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=source,
                    published=published,
                    scraped_excerpt=scraped_excerpt,
                    eid=eid,
                )
            )

    return KnowledgePack(
        query=str(d.get("query") or ""),
        stage=str(d.get("stage") or ""),
        generated_at=str(d.get("generated_at") or _now_iso()),
        items=items,
        provider=str(d.get("provider") or "mirothink_web"),
    )


def knowledge_pack_to_markdown(pack: KnowledgePack, max_chars: int = 6000) -> str:
    if not pack.items:
        return ""
    lines = []
    lines.append(f"**External knowledge query**: {pack.query}")
    lines.append(f"**Stage**: {pack.stage} | **Generated**: {pack.generated_at} | **Provider**: {pack.provider}")
    lines.append("")
    for idx, it in enumerate(pack.items, start=1):
        # Display Evidence ID if available
        eid_prefix = f"[{it.eid}] " if it.eid else ""
        title = it.title or "(untitled)"
        url = it.url or ""
        if url:
            lines.append(f"{idx}. {eid_prefix}[{title}]({url})")
        else:
            lines.append(f"{idx}. {eid_prefix}{title}")
        meta = " | ".join([p for p in [it.source, it.published] if p])
        if meta:
            lines.append(f"   - {meta}")
        if it.snippet:
            lines.append(f"   - Snippet: {it.snippet}")
        if it.scraped_excerpt:
            excerpt = it.scraped_excerpt.strip().replace("\n", "\n     ")
            lines.append(f"   - Notes:\n     {excerpt}")
        lines.append("")
    md = "\n".join(lines).strip()
    if max_chars and len(md) > max_chars:
        md = md[:max_chars] + "\n... (truncated)"
    return md
