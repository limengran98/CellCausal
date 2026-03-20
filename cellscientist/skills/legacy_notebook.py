from __future__ import annotations

from ..runtime.state import ResearchIntent, SessionState
from .base import BaseSkill
from .notebook_workflow import NotebookWorkflowSkill


class LegacyNotebookSkill(BaseSkill):
    """Compatibility shim that delegates legacy notebook tasks to the workflow router."""

    name = "legacy-notebook"
    description = "Compatibility shim for legacy notebook requests."
    aliases = ["legacy-notebook"]
    triggers = ["legacy notebook"]
    visible_in_suggestions = False
    supported_task_types = ["legacy_notebook"]

    def match(self, intent: ResearchIntent) -> float:
        if intent.task_type in self.supported_task_types:
            return 0.5
        return 0.0

    def run(self, state: SessionState) -> dict[str, str]:
        return NotebookWorkflowSkill().run(state)
