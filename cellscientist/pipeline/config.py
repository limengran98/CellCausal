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
    
    The merging follows a 3-tier inheritance pattern:
    1. Start with pipeline_config.json as base (Tier 1)
    2. Stage config overrides pipeline config (Tier 2/3)
    3. Stage-specific overrides from pipeline_config.json (if any)
    """
    # Start with pipeline config as base, then overlay stage config on top
    # This implements the 3-tier inheritance: pipeline < stage < stage_overrides
    
    # Filter out internal fields from pipeline config that shouldn't be inherited
    pipe_base = {k: v for k, v in pipe_cfg.items() 
                 if k not in ("_comment", "phase_overrides", "stage_overrides", "common", "env")}
    
    # Start by merging pipeline base with stage config
    # Stage config values override pipeline config values
    cfg = deep_merge(pipe_base, stage_cfg)
    
    # Now apply special handling for specific fields (backward compatibility)
    
    # 1) dataset_name - ensure it's set from pipeline if available
    if isinstance(pipe_cfg.get("dataset_name"), str) and pipe_cfg["dataset_name"].strip():
        cfg["dataset_name"] = pipe_cfg["dataset_name"].strip()

    common = pipe_cfg.get("common") if isinstance(pipe_cfg.get("common"), dict) else {}

    # 2) GPU selection: wipe cuda_device_id in child configs if parent sets CUDA_VISIBLE_DEVICES
    #    BUT only if the stage config didn't explicitly set its own cuda_device_id
    cuda_visible = common.get("cuda_visible_devices")
    cuda_id = common.get("cuda_device_id")
    has_gpu_setting = (cuda_visible is not None) or (cuda_id is not None)
    
    # Check if stage config has explicit cuda_device_id
    stage_has_cuda_id = (isinstance(stage_cfg.get("exec"), dict) and 
                         "cuda_device_id" in stage_cfg.get("exec", {}))
    
    if has_gpu_setting and not stage_has_cuda_id:
        # Both stages share the same exec.cuda_device_id convention
        # Only wipe if stage didn't explicitly set it
        if "exec" in cfg and isinstance(cfg["exec"], dict):
            set_nested(cfg, ["exec", "cuda_device_id"], None)

    # 3) LLM defaults - already handled by deep_merge above, but ensure proper merging order
    #    pipeline_config.json -> llm can provide base, stage overrides it
    llm_common_raw = pipe_cfg.get("llm") if isinstance(pipe_cfg.get("llm"), dict) else None
    llm_common = drop_none(llm_common_raw) if isinstance(llm_common_raw, dict) else None

    # The deep_merge above already handled this, but we need to ensure stage values win
    if isinstance(llm_common, dict) and llm_common:
        if isinstance(cfg.get("llm"), dict):
            # Re-merge to ensure stage config values override pipeline values
            cfg["llm"] = deep_merge(llm_common, cfg.get("llm") or {})

    # 4) Paths - already handled by deep_merge above
    
    # 5) Stage-specific overrides (from pipeline config's stage_overrides section)
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
