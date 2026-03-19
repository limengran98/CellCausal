from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from ..core.llm_client import resolve_llm_config


def _clean_base_url(base_url: str) -> str:
    url = str(base_url or "").strip()
    if url.startswith("[") and url.endswith(")"):
        url = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", url)
    return url.rstrip("/")


def normalize_base_url(base_url: str) -> tuple[str, List[str]]:
    warnings: List[str] = []
    cleaned = _clean_base_url(base_url)
    if not cleaned:
        warnings.append("base_url is empty")
        return cleaned, warnings

    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]
        warnings.append("base_url included /chat/completions; stripped to provider root")

    if not re.match(r"^https?://", cleaned):
        warnings.append("base_url is missing http/https scheme")
        return cleaned, warnings

    parsed = urlparse(cleaned)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
        warnings.append("base_url had no API version path; normalized to /v1")
    elif path == "/v1":
        pass
    elif re.fullmatch(r"/v\d+", path):
        warnings.append(f"base_url uses a nonstandard version path: {path}")
    else:
        warnings.append(f"base_url path looks nonstandard for an OpenAI-compatible endpoint: {path}")

    normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return normalized, warnings


def _host_of(url: str) -> str:
    cleaned = str(url or "").strip()
    if not cleaned or not re.match(r"^https?://", cleaned):
        return ""
    return urlparse(cleaned).netloc.lower()


def _is_unresolved_placeholder(value: Any) -> bool:
    return bool(re.search(r"\$\{[^}]+\}", str(value or "")))


def _resolve_api_key(
    raw_provider: Dict[str, Any],
    *,
    default_api_key: Optional[str] = None,
    default_api_key_source: str = "primary_inherited",
) -> tuple[Optional[str], str]:
    if raw_provider.get("api_key"):
        return str(raw_provider["api_key"]), "config"

    api_key_env = str(raw_provider.get("api_key_env") or "").strip()
    if api_key_env and os.environ.get(api_key_env):
        return os.environ.get(api_key_env), f"env:{api_key_env}"

    if default_api_key:
        return default_api_key, default_api_key_source

    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_API_KEY"), "env:OPENAI_API_KEY"

    return None, "missing"


def _detect_primary_config_source(
    cfg: Dict[str, Any],
    bridge_override: Optional[Dict[str, Any]],
) -> str:
    if bridge_override and isinstance(bridge_override.get("llm"), dict):
        return "bridge_override"

    llm_cfg = (cfg.get("llm") or {}) if isinstance(cfg.get("llm"), dict) else {}
    if any(llm_cfg.get(key) is not None for key in ("model", "base_url", "api_key", "api_key_env")):
        missing_fields = [key for key in ("model", "base_url", "api_key") if not llm_cfg.get(key)]
        if missing_fields and os.environ.get("OPENAI_API_KEY"):
            return "env_mixed"
        return "pipeline_config.llm"

    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return "env"
    return "default"


def _detect_primary_api_key_source(
    cfg: Dict[str, Any],
    bridge_override: Optional[Dict[str, Any]],
) -> str:
    if bridge_override and isinstance(bridge_override.get("llm"), dict):
        override_llm = bridge_override["llm"]
        if override_llm.get("api_key"):
            return "bridge_override"
        api_key_env = str(override_llm.get("api_key_env") or "").strip()
        if api_key_env and os.environ.get(api_key_env):
            return f"env:{api_key_env}"

    llm_cfg = (cfg.get("llm") or {}) if isinstance(cfg.get("llm"), dict) else {}
    if llm_cfg.get("api_key"):
        return "config"
    api_key_env = str(llm_cfg.get("api_key_env") or "").strip()
    if api_key_env and os.environ.get(api_key_env):
        return f"env:{api_key_env}"
    if os.environ.get("OPENAI_API_KEY"):
        return "env:OPENAI_API_KEY"
    return "missing"


def _build_provider_record(
    *,
    name: str,
    llm_defaults: Dict[str, Any],
    raw_provider: Dict[str, Any],
    config_source: str,
    default_api_key: Optional[str],
    default_api_key_source: str,
) -> Dict[str, Any]:
    runtime_llm = deepcopy(llm_defaults)
    for key in ("model", "base_url", "max_tokens", "temperature", "timeout"):
        if raw_provider.get(key) is not None:
            runtime_llm[key] = raw_provider.get(key)

    api_key, api_key_source = _resolve_api_key(
        raw_provider,
        default_api_key=default_api_key,
        default_api_key_source=default_api_key_source,
    )
    runtime_llm["api_key"] = api_key
    normalized_base_url, warnings = normalize_base_url(str(runtime_llm.get("base_url") or ""))
    runtime_llm["base_url"] = normalized_base_url
    skip_reasons: List[str] = []

    if _is_unresolved_placeholder(runtime_llm.get("model")):
        warnings.append("model is an unresolved template placeholder")
        skip_reasons.append("model unresolved")
    elif not runtime_llm.get("model"):
        warnings.append("model is empty")
        skip_reasons.append("model missing")

    raw_base_url = str(raw_provider.get("base_url") or llm_defaults.get("base_url") or "")
    if _is_unresolved_placeholder(raw_base_url):
        warnings.append("base_url is an unresolved template placeholder")
        skip_reasons.append("base_url unresolved")
    elif not normalized_base_url:
        skip_reasons.append("base_url missing")
    elif not re.match(r"^https?://", normalized_base_url):
        skip_reasons.append("base_url is not a valid http/https endpoint")

    if not runtime_llm.get("api_key"):
        warnings.append("api_key could not be resolved")
        skip_reasons.append("api_key missing")

    enabled = not skip_reasons

    diagnostic = {
        "provider_name": name,
        "name": name,
        "model": runtime_llm.get("model"),
        "base_url": raw_provider.get("base_url") or llm_defaults.get("base_url"),
        "normalized_base_url": normalized_base_url,
        "api_key_source": api_key_source,
        "api_key_available": bool(runtime_llm.get("api_key")),
        "source": config_source,
        "config_source": config_source,
        "enabled": enabled,
        "skip_reason": "; ".join(skip_reasons) if skip_reasons else None,
        "warnings": warnings,
    }
    return {
        "name": name,
        "runtime_llm": runtime_llm,
        "diagnostic": diagnostic,
    }


def _parse_env_fallbacks() -> List[Dict[str, Any]]:
    raw = os.environ.get("CELLSCIENTIST_LLM_FALLBACKS", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def resolve_bridge_llm_providers(
    cfg: Dict[str, Any],
    *,
    bridge_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_primary_provider = deepcopy((cfg.get("llm") or {}) if isinstance(cfg.get("llm"), dict) else {})
    if bridge_override and isinstance(bridge_override.get("llm"), dict):
        raw_primary_provider.update(bridge_override["llm"])

    resolved_primary = resolve_llm_config(cfg)
    for key in ("model", "base_url", "max_tokens", "temperature", "timeout"):
        if raw_primary_provider.get(key) is not None:
            resolved_primary[key] = raw_primary_provider.get(key)
    primary_source = _detect_primary_config_source(cfg, bridge_override)
    primary_api_key_source = _detect_primary_api_key_source(cfg, bridge_override)

    providers: List[Dict[str, Any]] = [
        _build_provider_record(
            name="primary",
            llm_defaults=resolved_primary,
            raw_provider=raw_primary_provider,
            config_source=primary_source,
            default_api_key=resolved_primary.get("api_key"),
            default_api_key_source=primary_api_key_source,
        )
    ]

    fallback_specs: List[tuple[str, Dict[str, Any]]] = []
    if bridge_override and isinstance(bridge_override.get("llm_fallbacks"), list):
        fallback_specs.extend(
            ("bridge_override", item)
            for item in bridge_override["llm_fallbacks"]
            if isinstance(item, dict)
        )

    if isinstance(cfg.get("llm_fallbacks"), list):
        fallback_specs.extend(
            ("pipeline_config.llm_fallbacks", item)
            for item in cfg["llm_fallbacks"]
            if isinstance(item, dict)
        )

    if isinstance(cfg.get("llm_report"), dict):
        fallback_specs.append(("pipeline_config.llm_report_compat", cfg["llm_report"]))

    fallback_specs.extend(("env:CELLSCIENTIST_LLM_FALLBACKS", item) for item in _parse_env_fallbacks())

    seen_signatures = {
        (
            str(providers[0]["diagnostic"].get("model") or ""),
            str(providers[0]["diagnostic"].get("normalized_base_url") or ""),
        )
    }
    primary_host = _host_of(str(providers[0]["diagnostic"].get("normalized_base_url") or ""))
    for idx, (config_source, raw_provider) in enumerate(fallback_specs, start=1):
        name = str(raw_provider.get("name") or f"fallback_{idx}")
        provider = _build_provider_record(
            name=name,
            llm_defaults=resolved_primary,
            raw_provider=raw_provider,
            config_source=config_source,
            default_api_key=resolved_primary.get("api_key"),
            default_api_key_source="primary_inherited",
        )
        signature = (
            str(provider["diagnostic"].get("model") or ""),
            str(provider["diagnostic"].get("normalized_base_url") or ""),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        fallback_host = _host_of(str(provider["diagnostic"].get("normalized_base_url") or ""))
        if primary_host and fallback_host and primary_host == fallback_host:
            provider["diagnostic"]["warnings"] = list(provider["diagnostic"].get("warnings") or [])
            provider["diagnostic"]["warnings"].append(
                "fallback shares the same host as primary; host-level outages may still affect both providers"
            )
        providers.append(provider)

    return {
        "providers": providers,
        "llm_resolution": {
            "config_source": "bridge_provider_resolution",
            "primary_provider": providers[0]["diagnostic"],
            "fallback_providers": [provider["diagnostic"] for provider in providers[1:]],
            "fallback_provider_count": max(len(providers) - 1, 0),
            "enabled_provider_count": sum(
                1 for provider in providers if provider["diagnostic"].get("enabled")
            ),
            "provider_count": len(providers),
            "primary": providers[0]["diagnostic"],
            "fallbacks": [provider["diagnostic"] for provider in providers[1:]],
        },
    }


def is_retryable_provider_error(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    retryable_markers = (
        "http 404",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "404",
        "429",
        "500",
        "502",
        "503",
        "504",
        "bad gateway",
        "invalid url",
        "connection error",
        "failed to establish a new connection",
        "name or service not known",
        "temporary failure in name resolution",
        "timed out",
        "timeout",
        "proxyerror",
        "max retries exceeded",
    )
    non_retryable_markers = (
        "failed to parse json",
        "json parse",
        "could not parse json",
        "notebook generation failed (json parse)",
        "yaml",
        "schema",
        "keyerror",
        "typeerror",
        "syntaxerror",
    )
    if any(marker in text for marker in non_retryable_markers):
        return False
    return any(marker in text for marker in retryable_markers)


def summarize_provider_error(exc: BaseException, *, max_chars: int = 240) -> str:
    text = str(exc or "").strip() or exc.__class__.__name__
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
