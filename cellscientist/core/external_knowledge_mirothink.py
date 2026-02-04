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
        url = url[len("https://r.jina.ai/") :]

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
    lit = _lit_cfg(cfg)
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

    # Users can override with cfg["literature"]["query"].
    forced = (lit.get("query") or "").strip()
    if forced:
        return forced

    # Prefer academic-ish results by adding common anchors.
    academic_bias = ' (paper OR arxiv OR "journal" OR doi OR pubmed OR "technical report" OR "specification")'
    parts = [p for p in [base_kw, hint, stage_kw] if p]
    return " ".join(parts).strip() + academic_bias


# -----------------------------
# Summarization (optional)
# -----------------------------


def _summarize_if_needed(cfg: Dict[str, Any], text: str, title: str, url: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

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
            return summary.strip()
        except Exception:
            pass

    # Fallback: truncate
    return text[:max_chars] + "\n... (truncated)"


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

    # Resolve keys (do NOT log secrets)
    serper_key = (lit.get("serper_api_key") or os.environ.get("SERPER_API_KEY") or "").strip()
    serper_base = (lit.get("serper_base_url") or os.environ.get("SERPER_BASE_URL") or "https://google.serper.dev").strip()
    jina_key = (lit.get("jina_api_key") or os.environ.get("JINA_API_KEY") or "").strip()
    jina_base = (lit.get("jina_base_url") or os.environ.get("JINA_BASE_URL") or "https://r.jina.ai").strip()

    # Determine artifact directories
    paths = cfg.get("paths") if isinstance(cfg, dict) else {}
    paths = paths if isinstance(paths, dict) else {}
    literature_dir = (paths.get("literature_dir") or lit.get("literature_dir") or "").strip()
    literature_dir = os.path.abspath(literature_dir) if literature_dir else ""
    workspace_out = os.path.join(os.path.abspath(workspace_dir), "external_knowledge") if workspace_dir else ""

    if not serper_key:
        log("[LIT] ❌ SERPER_API_KEY missing. External search is disabled for this run.")
        pack = KnowledgePack(
            query="",
            stage=stage,
            generated_at=_now_iso(),
            items=[
                EvidenceItem(
                    title="SERPER_API_KEY missing",
                    url="",
                    snippet="Set SERPER_API_KEY or cfg['literature']['serper_api_key'].",
                )
            ],
            provider="mirothink_web",
        )
        if literature_dir:
            _persist_pack_artifacts(pack, cfg, stage, literature_dir, tag=tag, log=log)
        if workspace_out:
            _persist_pack_artifacts(pack, cfg, stage, workspace_out, tag=tag, log=log)
        return pack

    # Build query
    q = _build_query(context_text=context_text, stage=stage, cfg=cfg, query_hint=query_hint)

    # Cache handling
    cache_days = int(lit.get("cache_days", 30) or 30)
    max_papers = int(lit.get("max_papers", 12) or 12)
    max_abstract_chars = int(lit.get("max_abstract_chars", 1200) or 1200)
    md_max_chars = int(lit.get("artifact_md_max_chars", 12000) or 12000)

    cache_dir = os.path.join(literature_dir, "cache") if literature_dir else ""
    cache_key = _sha1(json.dumps({"q": q, "stage": stage, "max": max_papers}, ensure_ascii=False))
    cache_path = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else ""

    log(f"[LIT] 🔎 stage={stage} | query={_truncate(q, 160)} | cache_days={cache_days} | max_papers={max_papers}")

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
            pack = _pack_from_json(cached)
            log(f"[LIT] ♻️ Using cached results (use_existing_only=true): {cache_path}")
            _persist_everywhere(pack)
            return pack
        pack = KnowledgePack(
            query=q,
            stage=stage,
            generated_at=_now_iso(),
            items=[EvidenceItem(title="Cache miss", url="", snippet="use_existing_only=true but cache not found")],
        )
        log(f"[LIT] ⚠️ Cache miss while use_existing_only=true: {cache_path}")
        _persist_everywhere(pack)
        return pack

    # Fresh-enough cache
    if cache_path and os.path.exists(cache_path):
        try:
            cached = _read_json(cache_path)
            ts = cached.get("generated_at") or ""
            if ts:
                gen_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if datetime.now(gen_time.tzinfo) - gen_time <= timedelta(days=cache_days):
                    pack = _pack_from_json(cached)
                    log(f"[LIT] ♻️ Using cached results: {cache_path}")
                    _persist_everywhere(pack)
                    return pack
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
        pack = KnowledgePack(query=q, stage=stage, generated_at=_now_iso(), items=[EvidenceItem(title="Search failed", url="", snippet=str(e))])
        log(f"[LIT] ❌ Search failed: {e}")
        _persist_everywhere(pack)
        return pack

    organic = data.get("organic") or []
    items: List[EvidenceItem] = []
    for r in organic[:max_papers]:
        title = str(r.get("title") or "")
        url = str(r.get("link") or r.get("url") or "")
        snippet = str(r.get("snippet") or "")
        published = str(r.get("date") or "")
        items.append(EvidenceItem(title=title, url=url, snippet=snippet, source=_domain(url), published=published))

    log(f"[LIT] ✅ Search results: {len(items)} items")

    # Optional scrape+summarize
    if bool(lit.get("scrape", True)) and items:
        if not jina_key:
            log("[LIT] ⚠️ JINA_API_KEY missing. Skipping page scraping (snippets only).")
        else:
            for idx, it in enumerate(items, start=1):
                try:
                    raw = _jina_scrape_url(it.url, api_key=jina_key, base_url=jina_base, max_chars=30000)
                    it.scraped_excerpt = _summarize_if_needed(cfg, raw, it.title, it.url, max_chars=max_abstract_chars)
                    log(f"[LIT] 📄 scraped {idx}/{len(items)}: {_truncate(it.title, 72)}")
                except Exception as e:
                    it.scraped_excerpt = f"(scrape failed) {e}"
                    log(f"[LIT][WARN] scrape failed {idx}/{len(items)}: {e}")

    pack = KnowledgePack(query=q, stage=stage, generated_at=_now_iso(), items=items, provider="mirothink_web")

    # Persist cache
    if cache_path and literature_dir:
        try:
            _write_json(cache_path, _json_safe_pack(pack))
            log(f"[LIT] 💾 Cached: {cache_path}")
        except Exception as e:
            log(f"[LIT][WARN] Failed to write cache: {e}")

    _persist_everywhere(pack)
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
        title = it.title or "(untitled)"
        url = it.url or ""
        if url:
            lines.append(f"{idx}. [{title}]({url})")
        else:
            lines.append(f"{idx}. {title}")
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
