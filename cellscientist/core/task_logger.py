#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structured task trace logging.

A lightweight, dependency-free implementation inspired by MiroThinker's
TaskLog/StepLog pattern.

Design goals
------------
- One JSON trace per trial/workspace.
- Append-only step logs with stable UTC timestamps.
- Record artifacts by path (and optional hash) for causal traceability.

This module is intentionally small and safe to import anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_file(path: str, max_bytes: int = 2_000_000) -> Optional[str]:
    """Compute sha256 for small/medium artifacts. Avoid hashing huge files."""
    try:
        if not path or not os.path.exists(path):
            return None
        size = os.path.getsize(path)
        if size <= 0:
            return None
        if size > max_bytes:
            # large file: hash head+tail for cheap fingerprint
            h = hashlib.sha256()
            with open(path, "rb") as f:
                h.update(f.read(max_bytes // 2))
                if size > max_bytes // 2:
                    f.seek(max(0, size - (max_bytes // 2)))
                    h.update(f.read(max_bytes // 2))
            return "sha256_headtail:" + h.hexdigest()

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 256), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _safe_write_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


@dataclass
class TaskLogger:
    """Append-only task trace logger."""

    trace_path: str
    state: Dict[str, Any]

    @classmethod
    def create_or_load(
        cls,
        root_dir: str,
        *,
        task_id: Optional[str] = None,
        trace_relpath: str = os.path.join("intermediate", "trace", "task_trace.json"),
        meta: Optional[Dict[str, Any]] = None,
    ) -> "TaskLogger":
        root_dir = os.path.abspath(root_dir)
        trace_path = os.path.join(root_dir, trace_relpath)

        if os.path.exists(trace_path):
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    return cls(trace_path=trace_path, state=state)
            except Exception:
                pass

        tid = task_id or os.path.basename(root_dir.rstrip(os.sep)) or f"task_{int(time.time())}"
        state = {
            "version": 1,
            "task_id": tid,
            "root_dir": root_dir,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "meta": meta or {},
            "steps": [],
            "artifacts": [],
        }
        logger = cls(trace_path=trace_path, state=state)
        logger.flush()
        return logger

    def flush(self) -> None:
        self.state["updated_at"] = _now_iso()
        _safe_write_json(self.state, self.trace_path)

    def log_step(self, name: str, message: str = "", level: str = "INFO", **metadata: Any) -> None:
        rec = {
            "ts": _now_iso(),
            "name": str(name),
            "level": str(level),
            "message": str(message or ""),
            "metadata": metadata or {},
        }
        self.state.setdefault("steps", []).append(rec)
        self.flush()

    def log_artifact(self, kind: str, path: str, description: str = "", *, compute_hash: bool = True, **metadata: Any) -> None:
        abs_path = os.path.abspath(path) if path else ""
        rec = {
            "ts": _now_iso(),
            "kind": str(kind),
            "path": abs_path,
            "description": str(description or ""),
            "hash": _sha256_file(abs_path) if compute_hash else None,
            "metadata": metadata or {},
        }
        self.state.setdefault("artifacts", []).append(rec)
        self.flush()


def get_task_logger(root_dir: str) -> TaskLogger:
    """Convenience loader/creator."""
    return TaskLogger.create_or_load(root_dir)
