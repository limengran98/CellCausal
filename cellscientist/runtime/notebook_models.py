from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NotebookArtifact:
    """Structured notebook artifact metadata for the V2 notebook skill family."""

    name: str
    path: Optional[str]
    trial_dir: Optional[str]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotebookRunResult:
    """Structured notebook execution result for the V2 notebook skill family."""

    notebook_path: Optional[str]
    trial_dir: Optional[str]
    status: str
    error_log_path: Optional[str]
    run_log_path: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
