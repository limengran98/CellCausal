#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Idea file generation.

Some workflows enable "idea mode" (--use-idea) and expect an `idea.json` file.
Historically this was produced by a Phase-1 step, but Phase-1 is excluded.

This module generates a minimal, well-formed `idea.json` from the current
technical spec so Phase-2 can still run in idea mode.

Expected schema
---------------
{
  "ideas": [
    {"title": "...", "description": "..."},
    ...
  ],
  "meta": {"generated_at": "...", "spec_path": "..."}
}

"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:
    yaml = None

from cellscientist.core.llm_client import chat_json


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _read_spec_text(spec_path: str) -> str:
    if not spec_path or not os.path.exists(spec_path):
        return ""
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _read_spec_obj(spec_path: str) -> Dict[str, Any]:
    if not yaml:
        return {}
    if not spec_path or not os.path.exists(spec_path):
        return {}
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _fallback_ideas() -> Dict[str, Any]:
    return {
        "ideas": [
            {
                "title": "Baseline sanity + strong regularization",
                "description": "Start from a minimal baseline, add early-stopping, weight decay, dropout, and ensure data pipeline correctness. Use this as a stable anchor for later improvements.",
            },
            {
                "title": "Loss shaping and target transform",
                "description": "Try target normalization/transform and a composite loss (e.g., MSE + rank-aware term) to stabilize optimization and preserve biological variance.",
            },
            {
                "title": "Feature fusion ablations",
                "description": "Systematically ablate and improve fusion (concat vs gated vs attention). Track which modality contributes under which conditions to guide architecture choices.",
            },
        ],
        "meta": {"generated_at": _now_iso(), "generator": "fallback"},
    }


def generate_idea_json(
    cfg: Dict[str, Any],
    spec_path: str,
    out_path: str,
    *,
    max_ideas: int = 6,
) -> Optional[str]:
    """Generate an idea.json from the spec.

    Returns the written path on success.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    spec_text = _read_spec_text(spec_path)
    spec_obj = _read_spec_obj(spec_path)

    # Build a compact prompt for idea generation.
    sys_msg = (
        "You are a research lead who proposes concrete candidate approaches for a ML/biology pipeline. "
        "You MUST return valid JSON only."
    )

    # Provide some structure so outputs are comparable across runs.
    user_msg = (
        "Given the following technical specification for the current task, propose "
        f"{max_ideas} candidate research / modeling ideas that could improve performance.\n\n"
        "Rules:\n"
        "- Each idea must have a short title and a 2-4 sentence description.\n"
        "- Ideas must be actionable within this codebase (model/loss/fusion/training/eval).\n"
        "- Avoid vague advice; mention what to change and why.\n\n"
        "Return JSON ONLY with this schema:\n"
        "{\n  \"ideas\": [{\"title\": str, \"description\": str}, ...],\n  \"meta\": {\"generated_at\": str, \"spec_path\": str}\n}\n\n"
        "TECH SPEC:\n"
        f"{spec_text[:12000]}\n"
    )

    data: Dict[str, Any] = {}
    try:
        data = chat_json([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ], cfg=cfg, temperature=0.2, max_tokens=2500, timeout=600)
    except Exception:
        data = {}

    # Validate minimal schema.
    ideas = []
    if isinstance(data, dict):
        raw = data.get("ideas")
        if isinstance(raw, list):
            for it in raw:
                if not isinstance(it, dict):
                    continue
                title = (it.get("title") or "").strip()
                desc = (it.get("description") or "").strip()
                if title and desc:
                    ideas.append({"title": title, "description": desc})

    if not ideas:
        data = _fallback_ideas()
        ideas = data.get("ideas", [])

    # Trim and write.
    ideas = ideas[:max_ideas]
    out = {
        "ideas": ideas,
        "meta": {
            "generated_at": _now_iso(),
            "spec_path": os.path.abspath(spec_path) if spec_path else "",
            "generator": "llm" if ideas and data and data is not _fallback_ideas() else "fallback",
            "spec_summary": {
                "dataset": (spec_obj.get("dataset_name") if spec_obj else None),
                "task": (spec_obj.get("task") if spec_obj else None),
            },
        },
    }

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return out_path
    except Exception:
        return None
