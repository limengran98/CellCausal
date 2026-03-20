from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..runtime.state import ResearchIntent, SessionState


class BaseSkill(ABC):
    """Base class for V2 runtime skills.

    Tiny repo-native metadata keeps the current runtime simple while leaving a
    future path open for directory-style skill packages or SKILL.md loaders.
    """

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    aliases: ClassVar[list[str]] = []
    triggers: ClassVar[list[str]] = []
    visible_in_suggestions: ClassVar[bool] = True
    supported_task_types: ClassVar[list[str]] = []

    def match(self, intent: ResearchIntent) -> float:
        """Return a simple compatibility score for the given intent."""

        if intent.task_type in self.supported_task_types:
            return 1.0
        return 0.0

    def skill_metadata(self) -> dict[str, Any]:
        """Expose lightweight metadata for suggestions and future skill dirs."""

        return {
            "name": self.name,
            "description": self.description,
            "aliases": list(self.aliases),
            "triggers": list(self.triggers),
            "supported_task_types": list(self.supported_task_types),
            "visible_in_suggestions": self.visible_in_suggestions,
        }

    @abstractmethod
    def run(self, state: SessionState) -> Any:
        """Execute the skill against the current session state."""
