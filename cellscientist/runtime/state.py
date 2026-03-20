from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .notebook_models import NotebookArtifact, NotebookRunResult


TaskType = Literal["drug_info", "legacy_notebook", "unknown"]


@dataclass
class ResearchIntent:
    """Minimal normalized intent extracted from a user query."""

    raw_query: str
    task_type: TaskType = "unknown"
    requested_actions: List[str] = field(default_factory=list)
    secondary_task_hints: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    mode: str = "default"


@dataclass
class Artifact:
    """A lightweight output container produced during orchestration."""

    type: str
    name: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """Mutable state passed through the new runtime skeleton."""

    session_id: str
    user_query: str
    intent: Optional[ResearchIntent] = None
    evidence_ids: List[str] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    skill_trace: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    last_notebook_artifact: Optional[NotebookArtifact] = None
    last_notebook_run_result: Optional[NotebookRunResult] = None
