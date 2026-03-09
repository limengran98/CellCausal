# -*- coding: utf-8 -*-
"""Workspace management for per-task file I/O and auditability.

Creates and manages a directory hierarchy under ``runs/{task_id}/`` for
research notes, generated code, execution logs, and final reports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Canonical subdirectory names.
_CATEGORIES = ("research", "modeling", "execution", "final")


class WorkspaceManager:
    """Manages the per-task directory hierarchy for a pipeline run.

    The layout created under *base_dir* is::

        runs/{task_id}/
        ├── research/     # BioKB & search results
        ├── modeling/     # Generated GNN/PyTorch code
        ├── execution/    # Logs and checkpoints
        ├── final/        # Final Auditable Virtual Cell report
        └── metadata.json # Task metadata

    Example::

        wm = WorkspaceManager("task-abc123")
        wm.ensure_dirs()
        path = wm.save_artifact("execution", "stdout.txt", "Hello world")

    Attributes:
        task_id: Unique identifier for the pipeline run.
        base_dir: Root directory that holds all ``runs/`` sub-trees.
        task_dir: ``base_dir / "runs" / task_id``.
    """

    def __init__(
        self,
        task_id: str,
        base_dir: Optional[Path] = None,
    ) -> None:
        """Initialise the workspace manager.

        Args:
            task_id: Unique identifier for the pipeline run.
            base_dir: Optional root directory.  Defaults to
                ``<project_root>/runs``.
        """
        self.task_id: str = task_id

        if base_dir is None:
            # Resolve relative to the cellscientist package root.
            base_dir = Path(__file__).resolve().parent.parent.parent / "runs"

        self.base_dir: Path = Path(base_dir)
        self.task_dir: Path = self.base_dir / task_id

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create all required subdirectories (idempotent).

        Creates:
        - ``<task_dir>/research/``
        - ``<task_dir>/modeling/``
        - ``<task_dir>/execution/``
        - ``<task_dir>/final/``
        """
        self.task_dir.mkdir(parents=True, exist_ok=True)
        for category in _CATEGORIES:
            (self.task_dir / category).mkdir(exist_ok=True)
        logger.debug("[WorkspaceManager] Ensured directories for task '%s'.", self.task_id)

    def get_task_dir(self) -> Path:
        """Return the root task directory path.

        Returns:
            :class:`~pathlib.Path` pointing to ``<base_dir>/runs/<task_id>/``.
        """
        return self.task_dir

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def save_artifact(
        self,
        category: str,
        filename: str,
        content: Union[str, dict, bytes],
    ) -> Path:
        """Save *content* as a file in the appropriate subdirectory.

        The category directory is created automatically if it does not exist.

        Args:
            category: Subdirectory name (e.g. ``"execution"``).
            filename: File name within that subdirectory.
            content: Content to write.  A :class:`dict` is serialised as
                JSON; :class:`bytes` are written in binary mode; everything
                else is coerced to ``str`` and written as UTF-8 text.

        Returns:
            :class:`~pathlib.Path` of the saved file.
        """
        dest_dir = self.task_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        if isinstance(content, bytes):
            dest.write_bytes(content)
        elif isinstance(content, dict):
            dest.write_text(
                json.dumps(content, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            dest.write_text(str(content), encoding="utf-8")

        logger.debug(
            "[WorkspaceManager] Saved artifact '%s/%s'.", category, filename
        )
        return dest

    def load_artifact(
        self,
        category: str,
        filename: str,
    ) -> Optional[Union[str, dict]]:
        """Load a previously saved artifact.

        Args:
            category: Subdirectory name.
            filename: File name within that subdirectory.

        Returns:
            Parsed :class:`dict` if the file is valid JSON, raw ``str``
            otherwise.  Returns ``None`` if the file does not exist.
        """
        path = self.task_dir / category / filename
        if not path.exists():
            logger.debug(
                "[WorkspaceManager] Artifact '%s/%s' not found.", category, filename
            )
            return None

        raw = path.read_text(encoding="utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def list_artifacts(self, category: str) -> List[Path]:
        """Return all files in *category* subdirectory.

        Args:
            category: Subdirectory name.

        Returns:
            Sorted list of :class:`~pathlib.Path` objects.  Empty list if the
            directory does not exist.
        """
        cat_dir = self.task_dir / category
        if not cat_dir.exists():
            return []
        return sorted(p for p in cat_dir.iterdir() if p.is_file())

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def save_metadata(self, extra: Dict[str, Any]) -> Path:
        """Update (merge) task metadata in ``metadata.json``.

        The file is created on first call.  Subsequent calls merge *extra*
        into the existing metadata dict (new keys added, existing keys
        overwritten).

        Args:
            extra: Additional fields to write/update.

        Returns:
            :class:`~pathlib.Path` of the metadata file.
        """
        meta_path = self.task_dir / "metadata.json"
        self.task_dir.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        # Inject standard fields if not already present.
        existing.setdefault("task_id", self.task_id)
        existing.setdefault(
            "created_at",
            datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        existing.setdefault("status", "running")

        existing.update(extra)

        meta_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("[WorkspaceManager] Saved metadata for task '%s'.", self.task_id)
        return meta_path
