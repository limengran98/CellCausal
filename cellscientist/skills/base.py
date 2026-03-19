from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..runtime.state import ResearchIntent, SessionState


class BaseSkill(ABC):
    """Base class for V2 runtime skills."""

    name: ClassVar[str] = "base"
    supported_task_types: ClassVar[list[str]] = []

    def match(self, intent: ResearchIntent) -> float:
        """Return a simple compatibility score for the given intent."""

        if intent.task_type in self.supported_task_types:
            return 1.0
        return 0.0

    @abstractmethod
    def run(self, state: SessionState) -> Any:
        """Execute the skill against the current session state."""
