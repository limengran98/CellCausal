# -*- coding: utf-8 -*-
"""Unified LLM utilities.

This module merges the previously duplicated `llm_utils.py` implementations from:
- Design & Execution (Phase 2)
- Review & Optimization (Phase 3)

Strict constraints honored:
- No functional logic removed (Phase 1 excluded by user request).
- No "smart" simplifications; the combined behavior preserves both call styles.

Compatibility notes
-------------------
The legacy code uses two different conventions:

1) Phase-2 style:
   - resolve_llm_config(llm_config_dict)
   - chat_text(..., llm_config=...)
   - chat_json(..., llm_config=...)

2) Phase-3 style:
   - resolve_llm_config(full_cfg_with_providers)
   - chat_text(..., cfg)
   - chat_json(..., cfg)

This file supports BOTH without requiring call-site behavioral changes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Union

import requests

# =============================================================================
# Token Meter (Telemetry & Cost Tracking)
# =============================================================================


class TokenMeter:
    """Thread-safe singleton to track LLM usage (Tokens & Latency)."""

    _lock = threading.Lock()
    _stats: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_latency_sec": 0.0,
        "api_calls": 0,
        "model_breakdown": {},
    }

    @classmethod
    def record(cls, response: dict, latency: float, model: str):
        """Parse OpenAI-compatible usage format and record stats."""
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        p_tok = usage.get("prompt_tokens", 0) if usage else 0
        c_tok = usage.get("completion_tokens", 0) if usage else 0
        t_tok = usage.get("total_tokens", p_tok + c_tok) if usage else 0

        with cls._lock:
            cls._stats["prompt_tokens"] += p_tok
            cls._stats["completion_tokens"] += c_tok
            cls._stats["total_tokens"] += t_tok
            cls._stats["total_latency_sec"] += float(latency or 0.0)
            cls._stats["api_calls"] += 1

            if model not in cls._stats["model_breakdown"]:
                cls._stats["model_breakdown"][model] = {"prompt": 0, "completion": 0, "calls": 0}
            cls._stats["model_breakdown"][model]["prompt"] += p_tok
            cls._stats["model_breakdown"][model]["completion"] += c_tok
            cls._stats["model_breakdown"][model]["calls"] += 1

    @classmethod
    def get_and_reset(cls) -> dict:
        """Return current snapshot and reset counters (Call after each iteration)."""
        with cls._lock:
            snapshot = dict(cls._stats)
            snapshot["model_breakdown"] = json.loads(json.dumps(cls._stats["model_breakdown"]))

            cls._stats = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_latency_sec": 0.0,
                "api_calls": 0,
                "model_breakdown": {},
            }
        return snapshot


# =============================================================================
# 1) Config Resolution (Merged)
# =============================================================================


def _looks_like_phase3_full_cfg(obj: Any) -> bool:
    """Heuristically detect a "full" pipeline config (Phase-3 style).

    Many call sites pass the entire config dict (with a top-level "llm" block)
    even when no provider registry is present. We treat those as full cfg so
    base_url/api_key inside cfg["llm"] are honored.
    """
    if not isinstance(obj, dict):
        return False

    # Explicit Phase-3 provider registry
    if "providers" in obj or "default_provider" in obj:
        return True

    llm_block = obj.get("llm")
    if isinstance(llm_block, dict):
        # If it looks like a pipeline config (paths / dataset / phases), treat as full cfg.
        if any(k in obj for k in ("paths", "dataset_name", "experiment", "review", "prompt_branch", "exec")):
            return True

        # If the llm block contains Phase-3-ish keys, still treat as full cfg (even without providers).
        if any(k in llm_block for k in ("provider", "timeout", "max_tokens", "base_url", "api_key", "model")):
            return True

    return False


def _resolve_from_llm_block(llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Phase-2 style resolution: Config > Env Vars > Defaults."""
    llm = llm_cfg or {}

    api_key = llm.get("api_key") or os.environ.get("OPENAI_API_KEY")
    model = llm.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o")
    base_url = llm.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Clean markdown links if present in URL (copy/paste artifact)
    if base_url and base_url.startswith("[") and base_url.endswith(")"):
        base_url = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", base_url)

    max_tokens = int(llm.get("max_tokens") or 20000)
    temperature = float(llm.get("temperature", 0.5))
    timeout = int(llm.get("timeout") or 600)

    if not api_key:
        print("[LLM] ⚠️ WARNING: No API Key found in config or environment!", flush=True)

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }


def _resolve_from_full_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Phase-3 style resolution: llm + providers with fallbacks."""
    cfg = cfg or {}

    llm = cfg.get("llm") or {}
    providers = cfg.get("providers") or {}

    provider_name = llm.get("provider") or cfg.get("default_provider")
    if not provider_name and isinstance(providers, dict) and providers:
        provider_name = next(iter(providers.keys()))

    prof = providers.get(provider_name, {}) if provider_name else {}

    model = llm.get("model") or prof.get("model") or "gpt-4"
    base_url = llm.get("base_url") or prof.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    api_key = (
        llm.get("api_key")
        or prof.get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get(f"{(provider_name or 'OPENAI').upper()}_API_KEY")
    )

    timeout = int(llm.get("timeout", prof.get("timeout", 300)))
    temperature = float(llm.get("temperature", prof.get("temperature", 0.7)))
    max_tokens = int(llm.get("max_tokens", prof.get("max_tokens", 40000)))

    if not api_key:
        print("[LLM] ⚠️ WARNING: No API Key found in config/provider/env!", flush=True)

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": timeout,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def resolve_llm_config(cfg_or_llm: Dict[str, Any]) -> Dict[str, Any]:
    """Unified resolver (keeps the original function name)."""
    if _looks_like_phase3_full_cfg(cfg_or_llm):
        return _resolve_from_full_cfg(cfg_or_llm)

    # If a Phase-3 style cfg is passed but without providers, still accept it:
    # treat it as a llm block if it has model/base_url/api_key style keys.
    if isinstance(cfg_or_llm, dict) and ("model" in cfg_or_llm or "api_key" in cfg_or_llm or "base_url" in cfg_or_llm):
        return _resolve_from_llm_block(cfg_or_llm)

    # Otherwise assume Phase-2 llm block.
    return _resolve_from_llm_block(cfg_or_llm or {})


# =============================================================================
# 2) Robust Parsing Helpers (Merged)
# =============================================================================


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def _strip_think(text: str) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()


def _escape_control_chars_in_strings(s: str) -> str:
    """Best-effort sanitizer used by the Phase-2 implementation."""
    if not s:
        return s

    out = []
    in_str = False
    esc = False

    for ch in s:
        o = ord(ch)

        if in_str:
            if esc:
                out.append(ch)
                esc = False
                continue

            if ch == "\\":
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

        if ch == '"':
            out.append(ch)
            in_str = True
            esc = False
            continue

        if o < 32 and ch not in ("\n", "\r", "\t"):
            continue

        out.append(ch)

    return "".join(out)


def extract_json_from_text(text: str) -> Union[Dict[str, Any], List[Any]]:
    """Merged extractor.

    - Preserves Phase-3's behavior (strip wrappers; try regex; json; ast literal)
    - Preserves Phase-2's behavior (fenced blocks; think-strip; control-char fixes)

    Returns dict or list (legacy Phase-3 sometimes returned list).
    """
    if not text:
        raise ValueError("Empty response from LLM")

    raw = text.strip()
    raw = _strip_think(raw)

    # Phase-3 pre-processing: remove leading/trailing wrappers
    t = re.sub(r"^[^{\[]*", "", raw)
    t = re.sub(r"[^}\]]*$", "", t)
    t = t.strip() or raw

    candidates: List[str] = []

    # Phase-2/3: fenced blocks
    m = re.search(r"```(?:json|python)?\s*(\{.*?\}|\[.*?\])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidates.append(m.group(1))

    # braces/brackets
    m2 = re.search(r"(\{.*\}|\[.*\])", t, flags=re.DOTALL)
    if m2:
        candidates.append(m2.group(1))

    candidates.append(t)

    errors: List[str] = []
    for cand in candidates:
        c = cand.strip()
        if not c:
            continue

        # 1) strict json
        try:
            return json.loads(c)
        except Exception as e:
            errors.append(f"json: {e}")

        # 2) sanitize control chars then json
        try:
            fixed = _escape_control_chars_in_strings(c)
            return json.loads(fixed)
        except Exception as e:
            errors.append(f"json_sanitize: {e}")

        # 3) literal_eval (python dicts/single quotes/triple quotes)
        try:
            py = c
            py = re.sub(r"\btrue\b", "True", py, flags=re.IGNORECASE)
            py = re.sub(r"\bfalse\b", "False", py, flags=re.IGNORECASE)
            py = re.sub(r"\bnull\b", "None", py, flags=re.IGNORECASE)
            val = ast.literal_eval(py)
            if isinstance(val, (dict, list)):
                return val
        except Exception as e:
            errors.append(f"ast: {e}")

    preview = raw[:200] + " ... " + raw[-200:] if len(raw) > 450 else raw
    raise ValueError(f"Failed to parse JSON. Tried {len(candidates)} candidates. Errors: {errors}. Preview:\n{preview}")


# =============================================================================
# 3) HTTP Helper (Merged)
# =============================================================================


def _post_request(url: str, headers: dict, payload: dict, timeout: int = 600, retries: int = 3) -> dict:
    """Robust HTTP posting with retries (Phase-2 semantics)."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()

            print(f"[LLM] HTTP {r.status_code} (Attempt {attempt+1}/{retries}): {r.text[:200]}", flush=True)
            time.sleep(1 + attempt)
        except Exception as e:
            last_err = e
            print(f"[LLM] Connection Error (Attempt {attempt+1}/{retries}): {e}", flush=True)
            time.sleep(1 + attempt)

    raise RuntimeError(f"LLM Request failed after {retries} retries. URL: {url}. Last error: {last_err}")


def _extract_text_from_response(data: dict) -> str:
    """Best-effort extraction of assistant text.

    Some providers claim OpenAI-compatibility but return different shapes (e.g.
    Gemini-style 'candidates', or OpenAI completion 'text'). We keep this helper
    conservative: if we can't confidently find text, return ''.
    """
    try:
        # OpenAI chat
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0] or {}
            msg = c0.get("message") or {}
            content = msg.get("content")

            # content can be a string OR a list of parts
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        t = p.get("text") or p.get("content")
                        if isinstance(t, str):
                            parts.append(t)
                return "\n".join([p for p in parts if p]).strip()

            # Some servers return completion-style text
            t = c0.get("text")
            if isinstance(t, str):
                return t.strip()

            # Streaming delta fallback (shouldn't happen with stream=False but safe)
            delta = c0.get("delta") or {}
            dcontent = delta.get("content")
            if isinstance(dcontent, str):
                return dcontent.strip()

        # Gemini-style
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            c0 = candidates[0] or {}
            content = (c0.get("content") or {})
            parts = content.get("parts")
            if isinstance(parts, list):
                out = []
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        out.append(p["text"])
                return "\n".join(out).strip()

        # Generic fallbacks
        if isinstance(data.get("output_text"), str):
            return str(data.get("output_text")).strip()
    except Exception:
        return ""

    return ""


def _dump_llm_response_if_needed(data: dict, kind: str = "empty") -> None:
    """Dump raw response JSON for debugging when env var is set.

    Set LLM_DUMP_DIR to a directory to enable.
    """
    try:
        dump_dir = os.environ.get("LLM_DUMP_DIR", "").strip()
        if not dump_dir:
            return
        os.makedirs(dump_dir, exist_ok=True)
        fn = f"llm_{kind}_{int(time.time())}.json"
        with open(os.path.join(dump_dir, fn), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =============================================================================
# 4) Chat Functions (Merged)
# =============================================================================


def _resolve_any(cfg: Optional[Dict[str, Any]] = None, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if llm_config is not None:
        return resolve_llm_config(llm_config)
    if cfg is not None:
        # If caller passes full cfg, resolve from that; if they pass cfg.get("llm"), it still works.
        return resolve_llm_config(cfg)
    return resolve_llm_config({})


def chat_text(
    messages: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    llm_config: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    debug_dir: Optional[str] = None,
    **kwargs,  # swallow legacy extras (e.g., meta)
) -> str:
    """Chat completion returning raw text.

    - Phase-2 callers pass llm_config=...
    - Phase-3 callers pass cfg (full) positionally
    """

    resolved = _resolve_any(cfg=cfg, llm_config=llm_config)

    temp = temperature if temperature is not None else float(resolved.get("temperature", 0.7))
    toks = max_tokens if max_tokens is not None else int(resolved.get("max_tokens", 40000))
    to = int(timeout if timeout is not None else resolved.get("timeout", 600))

    url = (resolved.get("base_url") or "").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {resolved.get('api_key')}", "Content-Type": "application/json"}

    payload = {
        "model": resolved.get("model"),
        "messages": messages,
        "temperature": temp,
        "max_tokens": toks,
        "stream": False,
    }

    # Preserve Phase-2 logging verbosity.
    print(f"[LLM] Text Gen -> Model: {resolved.get('model')} | URL: {resolved.get('base_url')}", flush=True)

    # Phase-2 empty-content retry loop (does not duplicate HTTP retry logic).
    for attempt in range(3):
        t0 = time.time()
        data = _post_request(url, headers, payload, timeout=to, retries=3)
        duration = time.time() - t0
        TokenMeter.record(data, duration, str(resolved.get("model")))

        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"llm_resp_{int(time.time())}.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

        # Some servers return different response shapes; extract robustly.
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # Many OpenAI-ish gateways return HTTP 200 with an embedded error.
            err = data.get("error") or {}
            print(f"[LLM] ⚠️ API error field: {str(err)[:200]}", flush=True)

        content = _extract_text_from_response(data)
        if content:
            return content

        _dump_llm_response_if_needed(data, kind="empty_text")

        print(f"[LLM] ⚠️ Warning: Received empty content from API (Attempt {attempt+1}/3).", flush=True)
        time.sleep(0.8 + 0.8 * attempt)

    return ""


def chat_json(
    messages: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    llm_config: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: Optional[int] = None,
    max_retries: int = 3,
    **kwargs,  # swallow legacy extras
) -> Dict[str, Any]:
    """Chat completion returning parsed JSON/dict.

    Behavior preservation:
    - Phase-2: uses response_format=json_object and a stricter retry nudge.
    - Phase-3: uses plain chat + robust extraction (no response_format requirement).

    Returns dict for downstream compatibility (if list is produced, it will be wrapped).
    """

    resolved = _resolve_any(cfg=cfg, llm_config=llm_config)
    to = int(timeout if timeout is not None else resolved.get("timeout", 600))

    url = (resolved.get("base_url") or "").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {resolved.get('api_key')}", "Content-Type": "application/json"}

    base_payload: Dict[str, Any] = {
        "model": resolved.get("model"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Phase-2 requested response_format; preserve for llm_config style calls.
    if llm_config is not None:
        base_payload["response_format"] = {"type": "json_object"}

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        payload = dict(base_payload)

        # Phase-2 retry nudge: add a strict system reminder.
        if attempt > 0 and llm_config is not None:
            payload["messages"] = messages + [
                {"role": "system", "content": "Return ONLY a valid JSON object. No markdown, no prose, no <think>."}
            ]

        # Phase-3 retry behavior: slightly increase temperature for variety.
        if attempt > 0 and llm_config is None:
            payload["temperature"] = float(temperature) + (0.1 * attempt)

        try:
            t0 = time.time()
            data = _post_request(url, headers, payload, timeout=to, retries=3)
            duration = time.time() - t0
            TokenMeter.record(data, duration, str(resolved.get("model")))

            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                err = data.get("error") or {}
                print(f"[LLM] ⚠️ API error field: {str(err)[:200]}", flush=True)

            content = _extract_text_from_response(data)
            if not content:
                print(f"[LLM] ⚠️ Empty JSON content (Attempt {attempt+1}/{max_retries}).", flush=True)
                _dump_llm_response_if_needed(data, kind="empty_json")
                time.sleep(0.8 + 0.8 * attempt)
                continue

            parsed = extract_json_from_text(content)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"edits": parsed}

            # unexpected
            return {}

        except Exception as e:
            last_error = e
            print(f"[LLM] JSON Parse/Net Error (Attempt {attempt+1}/{max_retries}): {e}", flush=True)
            time.sleep(0.8 + 0.8 * attempt)

    print(f"[LLM] Failed to parse JSON after retries. Last error: {last_error}", flush=True)
    return {}
