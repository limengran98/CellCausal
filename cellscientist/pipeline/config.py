#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline-config loading + per-phase config materialization.

This module is a refactor of the logic previously embedded in run_cellscientist.py.
It keeps behavior compatible:

- Optional pipeline_config.json (or env CELL_SCI_PIPELINE_CONFIG) can override
  dataset_name / common env / llm fields / paths / per-phase overrides.
- A common "all-null" llm block behaves as a true no-op.
- If common GPU settings are provided, phase configs will have cuda_device_id
  wiped so children inherit CUDA_VISIBLE_DEVICES from the parent process.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .utils import load_json, project_root


def get_config_path(phase_info: Dict[str, Any]) -> str:
    """Return the resolved config path for a phase."""
    cfg = phase_info.get("config")
    if not cfg:
        return os.path.join(phase_info["folder"], cfg)
    if os.path.isabs(cfg) or os.path.exists(cfg):
        return cfg
    return os.path.join(phase_info["folder"], cfg)


def get_nested(data: Dict[str, Any], keys: List[str], default="N/A"):
    val: Any = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val


def deep_merge(dst: Any, src: Any) -> Any:
    """Recursively merge src into dst and return merged (does not mutate inputs)."""
    if isinstance(dst, dict) and isinstance(src, dict):
        out = dict(dst)
        for k, v in src.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return src if src is not None else dst


def drop_none(obj: Any) -> Any:
    """Recursively drop None values from dict/list structures."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            vv = drop_none(v)
            if vv == {} or vv == []:
                continue
            out[k] = vv
        return out
    if isinstance(obj, list):
        out_list = []
        for v in obj:
            if v is None:
                continue
            vv = drop_none(v)
            if vv == {} or vv == []:
                continue
            out_list.append(vv)
        return out_list
    return obj


def set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_pipeline_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load pipeline-level overrides.

    Priority order:
    1) Explicit ``path`` argument (if provided and exists)
    2) Env var ``CELL_SCI_PIPELINE_CONFIG`` (if exists)
    3) ``configs/pipeline_config.json`` under repo root (new unified layout)
    4) Legacy ``pipeline_config.json`` under repo root

    Returns an empty dict if no config is found.
    """

    # 1) explicit argument
    if path and os.path.exists(path):
        try:
            return load_json(path)
        except Exception:
            return {}

    # 2) env var
    env_path = os.environ.get("CELL_SCI_PIPELINE_CONFIG")
    if env_path and os.path.exists(env_path):
        try:
            return load_json(env_path)
        except Exception:
            return {}

    # 3) unified default location
    default_path = os.path.join(project_root(), "configs", "pipeline_config.json")
    if os.path.exists(default_path):
        try:
            return load_json(default_path)
        except Exception:
            return {}

    # 4) legacy
    legacy_path = os.path.join(project_root(), "pipeline_config.json")
    if os.path.exists(legacy_path):
        try:
            return load_json(legacy_path)
        except Exception:
            return {}

    return {}


def pipeline_extra_env(pipe_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Env vars to pass to all phase subprocesses."""
    env_out: Dict[str, str] = {}
    common = pipe_cfg.get("common") if isinstance(pipe_cfg.get("common"), dict) else {}
    if common.get("cuda_visible_devices") is not None:
        env_out["CUDA_VISIBLE_DEVICES"] = str(common["cuda_visible_devices"])
    elif common.get("cuda_device_id") is not None:
        env_out["CUDA_VISIBLE_DEVICES"] = str(common["cuda_device_id"])

    env_cfg = pipe_cfg.get("env") if isinstance(pipe_cfg.get("env"), dict) else {}
    for k, v in env_cfg.items():
        if v is None:
            continue
        env_out[str(k)] = str(v)
    return env_out


def apply_pipeline_overrides(
    stage_name: str,
    stage_cfg: Dict[str, Any],
    pipe_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply pipeline-level overrides to a single stage config.

    This is the unified refactor of the original Phase-based logic. Stage names are:
    - "Experiment"  (Design & Execution)
    - "Review"       (Review & Optimization)

    Phase 1 logic is intentionally removed per user requirement.
    """
    cfg = dict(stage_cfg)

    # 1) dataset_name
    if isinstance(pipe_cfg.get("dataset_name"), str) and pipe_cfg["dataset_name"].strip():
        cfg["dataset_name"] = pipe_cfg["dataset_name"].strip()

    common = pipe_cfg.get("common") if isinstance(pipe_cfg.get("common"), dict) else {}

    # 2) GPU selection: wipe cuda_device_id in child configs if parent sets CUDA_VISIBLE_DEVICES
    cuda_visible = common.get("cuda_visible_devices")
    cuda_id = common.get("cuda_device_id")
    has_gpu_setting = (cuda_visible is not None) or (cuda_id is not None)
    if has_gpu_setting:
        # Both stages share the same exec.cuda_device_id convention
        set_nested(cfg, ["exec", "cuda_device_id"], None)

    # 3) LLM defaults:
    #    pipeline_config.json -> llm can override stage llm blocks.
    llm_common_raw = pipe_cfg.get("llm") if isinstance(pipe_cfg.get("llm"), dict) else None
    llm_common = drop_none(llm_common_raw) if isinstance(llm_common_raw, dict) else None

    # Stage configs may have either:
    #   - top-level llm: {...}
    #   - provider style: {"llm": {...}, "providers": {...}, "default_provider": ...}
    if isinstance(llm_common, dict) and llm_common:
        if isinstance(cfg.get("llm"), dict):
            cfg["llm"] = deep_merge(cfg.get("llm") or {}, llm_common)
        else:
            # If the stage uses provider-style, merge into its nested llm block.
            nested = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
            if isinstance(cfg.get("providers"), dict) or isinstance(cfg.get("default_provider"), str):
                cfg["llm"] = deep_merge(nested, llm_common)
            else:
                cfg["llm"] = deep_merge({}, llm_common)

    # 4) Paths (common)
    paths_common = pipe_cfg.get("paths") if isinstance(pipe_cfg.get("paths"), dict) else None
    if isinstance(paths_common, dict) and paths_common:
        cur = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
        cfg["paths"] = deep_merge(cur, paths_common)

    # 5) Stage-specific overrides
    stage_overrides = pipe_cfg.get("stage_overrides") if isinstance(pipe_cfg.get("stage_overrides"), dict) else {}
    so = stage_overrides.get(stage_name) if isinstance(stage_overrides.get(stage_name), dict) else None
    if isinstance(so, dict) and so:
        cfg = deep_merge(cfg, so)

    return cfg


def materialize_merged_configs(stage_map: Dict[str, Dict[str, Any]], pipe_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Write merged per-phase configs under each phase folder and update stage_map in-place."""
    merged_paths: Dict[str, str] = {}
    dataset_tag = "default"
    if pipe_cfg.get("dataset_name"):
        dataset_tag = "".join(c for c in str(pipe_cfg["dataset_name"]) if c.isalnum() or c in ("-", "_"))

    for phase_name, info in stage_map.items():
        base_cfg_path = get_config_path(info)
        base_cfg = load_json(base_cfg_path)
        # Stage override keys are semantic (not phase-numbered)
        stage_key = phase_name
        if phase_name == "Phase 2":
            stage_key = "Experiment"
        elif phase_name == "Phase 3":
            stage_key = "Review"

        merged_cfg = apply_pipeline_overrides(stage_key, base_cfg, pipe_cfg)

        cache_dir = os.path.join(info["folder"], "_pipeline_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.basename(info["config"])
        if base_name.endswith(".json"):
            out_name = base_name[:-5] + f".{dataset_tag}.merged.json"
        else:
            out_name = base_name + f".{dataset_tag}.merged.json"

        out_path = os.path.abspath(os.path.join(cache_dir, out_name))
        with open(out_path, "w", encoding="utf-8") as f:
            import json

            json.dump(merged_cfg, f, ensure_ascii=False, indent=2)

        info["config"] = out_path
        info["_loaded_cfg"] = merged_cfg
        merged_paths[phase_name] = out_path

    return merged_paths


def validate_configs(stage_map: Dict[str, Dict[str, Any]]) -> str:
    """Ensure dataset_name matches across all phase configs and store *expanded* configs.

    NOTE: Phase workflows load configs via `cellscientist.core.config_loader.load_full_config`,
    which expands ${VAR} placeholders (e.g. ${dataset_name}). The unified pipeline runner
    also needs the expanded values when it computes artifact paths and reports metrics.
    """
    # Local import to avoid any potential import-order issues.
    try:
        from cellscientist.core.config_loader import load_full_config  # type: ignore
    except Exception:
        load_full_config = None  # type: ignore

    dataset_names: Dict[str, str] = {}
    for name, info in stage_map.items():
        path = get_config_path(info)
        path = os.path.abspath(path) if path and not os.path.isabs(path) else path

        cfg: Dict[str, Any]
        if load_full_config is not None:
            try:
                cfg = load_full_config(path)
            except Exception:
                cfg = load_json(path)
        else:
            cfg = load_json(path)

        ds = cfg.get("dataset_name", "MISSING")
        dataset_names[name] = ds
        info["_loaded_cfg"] = cfg

    unique = set(dataset_names.values())
    if len(unique) > 1:
        raise RuntimeError(f"CRITICAL ERROR: 'dataset_name' mismatch across phases: {dataset_names}")
    return list(unique)[0]
