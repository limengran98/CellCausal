# -*- coding: utf-8 -*-
"""External knowledge retrieval (MiroThink-derived).

Goal
----
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

Public API
----------
- retrieve_external_knowledge(cfg, context_text, stage, query_hint=None) -> KnowledgePack
- knowledge_pack_to_markdown(pack, max_chars=6000) -> str
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

# Optional: reuse CellScientist LLM to summarize long pages
try:
    from .llm_client import chat_text
except Exception:  # pragma: no cover
    chat_text = None  # type: ignore


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
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except Exception:
        return ""


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
) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return f"Invalid URL: {url}"

    # Avoid duplicate prefix
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/"):]

    # HF datasets/spaces: usually too large / not intended for scraping summaries
    if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
        return "Skipping HuggingFace dataset/space page (often too large for scraping)."

    if not api_key:
        return "JINA_API_KEY not set; scraping disabled."

    jina_url = f"{base_url.rstrip('/')}/{url}"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(jina_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    content = _strip_markdown_links(r.text.strip())
    if max_chars and len(content) > max_chars:
        content = content[:max_chars] + "\n... (truncated)"
    return content


# -----------------------------
# Query builder
# -----------------------------

def _build_query(context_text: str, stage: str, cfg: Dict[str, Any], query_hint: Optional[str]) -> str:
    lit = (cfg.get("literature") or {}) if isinstance(cfg, dict) else {}
    base_kw = (lit.get("task_keywords") or "").strip()
    stage_kw = "design draft" if stage == "design" else "review feedback"
    # A light heuristic: use hint if supplied; else take first ~12 keywords from context.
    if query_hint and query_hint.strip():
        hint = query_hint.strip()
    else:
        # naive keyword extraction: keep alphanum tokens, length>=4, unique
        import re
        toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", context_text or "")
        uniq = []
        seen = set()
        for t in toks:
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            uniq.append(t)
            if len(uniq) >= 12:
                break
        hint = " ".join(uniq)

    # Prefer academic-ish results by adding common anchors.
    # Users can override with cfg["literature"]["query"].
    forced = (lit.get("query") or "").strip()
    if forced:
        return forced

    academic_bias = ' (paper OR arxiv OR "journal" OR doi OR pubmed OR "technical report" OR "specification")'
    parts = [p for p in [base_kw, hint, stage_kw] if p]
    return " ".join(parts) + academic_bias


# -----------------------------
# Summarization (optional)
# -----------------------------

def _summarize_if_needed(cfg: Dict[str, Any], text: str, title: str, url: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    # If CellScientist LLM is available and enabled, summarize.
    lit = (cfg.get("literature") or {}) if isinstance(cfg, dict) else {}
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
            return summary.strip()
        except Exception:
            pass

    # Fallback: truncate
    return text[:max_chars] + "\n... (truncated)"


# -----------------------------
# Public function
# -----------------------------

def retrieve_external_knowledge(
    cfg: Dict[str, Any],
    context_text: str,
    stage: str,
    query_hint: Optional[str] = None,
) -> KnowledgePack:
    """Retrieve and (optionally) scrape external knowledge for the given stage.

    stage: "design" | "review" | any string for tagging
    """
    lit = (cfg.get("literature") or {}) if isinstance(cfg, dict) else {}
    enabled = bool(lit.get("enabled", False))
    if not enabled:
        return KnowledgePack(query="", stage=stage, generated_at=_now_iso(), items=[], provider="disabled")

    # Resolve keys
    serper_key = (lit.get("serper_api_key") or os.environ.get("SERPER_API_KEY") or "").strip()
    serper_base = (lit.get("serper_base_url") or os.environ.get("SERPER_BASE_URL") or "https://google.serper.dev").strip()
    jina_key = (lit.get("jina_api_key") or os.environ.get("JINA_API_KEY") or "").strip()
    jina_base = (lit.get("jina_base_url") or os.environ.get("JINA_BASE_URL") or "https://r.jina.ai").strip()

    if not serper_key:
        # No key => return informative pack
        return KnowledgePack(
            query="",
            stage=stage,
            generated_at=_now_iso(),
            items=[EvidenceItem(title="SERPER_API_KEY missing", url="", snippet="Set SERPER_API_KEY or cfg['literature']['serper_api_key']")],
            provider="mirothink_web",
        )

    # Build query
    q = _build_query(context_text=context_text, stage=stage, cfg=cfg, query_hint=query_hint)

    # Cache handling
    cache_days = int(lit.get("cache_days", 30) or 30)
    max_papers = int(lit.get("max_papers", 12) or 12)
    max_abstract_chars = int(lit.get("max_abstract_chars", 1200) or 1200)
    prompt_max_chars = int(lit.get("prompt_max_chars", 6000) or 6000)
    literature_dir = ((cfg.get("paths") or {}) if isinstance(cfg.get("paths"), dict) else {}).get("literature_dir") or lit.get("literature_dir") or ""
    cache_dir = os.path.join(literature_dir, "cache") if literature_dir else ""

    cache_key = _sha1(json.dumps({"q": q, "stage": stage, "max": max_papers}, ensure_ascii=False))
    cache_path = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else ""

    if cache_path and lit.get("use_existing_only", False) and os.path.exists(cache_path):
        cached = _read_json(cache_path)
        return _pack_from_json(cached)

    if cache_path and os.path.exists(cache_path):
        try:
            cached = _read_json(cache_path)
            ts = cached.get("generated_at") or ""
            if ts:
                gen_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if datetime.now(gen_time.tzinfo) - gen_time <= timedelta(days=cache_days):
                    return _pack_from_json(cached)
        except Exception:
            pass

    # Search
    gl = (lit.get("search_gl") or "us").strip()
    hl = (lit.get("search_hl") or "en").strip()
    location = lit.get("search_location")
    tbs = lit.get("search_tbs")
    try:
        data = _serper_google_search(
            q=q,
            gl=gl,
            hl=hl,
            location=location,
            num=max(3, min(10, max_papers)),
            tbs=tbs,
            api_key=serper_key,
            base_url=serper_base,
        )
    except Exception as e:
        return KnowledgePack(query=q, stage=stage, generated_at=_now_iso(), items=[EvidenceItem(title="Search failed", url="", snippet=str(e))])

    organic = data.get("organic") or []
    items: List[EvidenceItem] = []
    for r in organic[:max_papers]:
        title = str(r.get("title") or "")
        url = str(r.get("link") or r.get("url") or "")
        snippet = str(r.get("snippet") or "")
        published = str(r.get("date") or "")
        items.append(EvidenceItem(title=title, url=url, snippet=snippet, source=_domain(url), published=published))

    # Optional scrape+summarize
    if lit.get("scrape", True) and items and jina_key:
        for it in items:
            try:
                raw = _jina_scrape_url(it.url, api_key=jina_key, base_url=jina_base, max_chars=30000)
                it.scraped_excerpt = _summarize_if_needed(cfg, raw, it.title, it.url, max_chars=max_abstract_chars)
            except Exception as e:
                it.scraped_excerpt = f"(scrape failed) {e}"

    pack = KnowledgePack(query=q, stage=stage, generated_at=_now_iso(), items=items, provider="mirothink_web")

    # Persist cache
    if cache_path and literature_dir:
        try:
            _write_json(cache_path, asdict(pack))
        except Exception:
            pass

    return pack


def _pack_from_json(d: Dict[str, Any]) -> KnowledgePack:
    try:
        items = [EvidenceItem(**it) for it in (d.get("items") or [])]
    except Exception:
        items = []
    return KnowledgePack(
        query=d.get("query") or "",
        stage=d.get("stage") or "",
        generated_at=d.get("generated_at") or _now_iso(),
        items=items,
        provider=d.get("provider") or "mirothink_web",
    )


def knowledge_pack_to_markdown(pack: KnowledgePack, max_chars: int = 6000) -> str:
    if not pack.items:
        return ""
    lines = []
    lines.append(f"**External knowledge query**: {pack.query}")
    lines.append(f"**Stage**: {pack.stage} | **Generated**: {pack.generated_at} | **Provider**: {pack.provider}")
    lines.append("")
    for idx, it in enumerate(pack.items, start=1):
        lines.append(f"{idx}. [{it.title}]({it.url})")
        meta = " | ".join([p for p in [it.source, it.published] if p])
        if meta:
            lines.append(f"   - {meta}")
        if it.snippet:
            lines.append(f"   - Snippet: {it.snippet}")
        if it.scraped_excerpt:
            # indent excerpt
            excerpt = it.scraped_excerpt.strip()
            excerpt = excerpt.replace("\n", "\n     ")
            lines.append(f"   - Notes:\n     {excerpt}")
        lines.append("")
    md = "\n".join(lines).strip()
    if max_chars and len(md) > max_chars:
        md = md[:max_chars] + "\n... (truncated)"
    return md
