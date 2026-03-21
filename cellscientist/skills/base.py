from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ..pipeline.utils import project_root
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
    skill_directory_name: ClassVar[str | None] = None

    def match(self, intent: ResearchIntent) -> float:
        """Return a simple compatibility score for the given intent."""

        if intent.task_type in self.supported_task_types:
            return 1.0
        return 0.0

    def skill_package_metadata(self) -> dict[str, Any]:
        """Return the optional directory-level skill package mapping.

        The Python class remains the runtime source of truth. This only exposes
        a lightweight bridge to future `skills/<name>/SKILL.md` packages.
        """

        directory_name = self.skill_directory_name or self.name
        package_dir = Path(project_root()) / "skills" / directory_name
        skill_doc = package_dir / "SKILL.md"
        has_skill_package = package_dir.is_dir()
        has_skill_doc = skill_doc.is_file()

        return {
            "directory_name": directory_name,
            "directory_path": str(package_dir) if has_skill_package else None,
            "skill_doc_path": str(skill_doc) if has_skill_doc else None,
            "has_skill_package": has_skill_package,
            "has_skill_doc": has_skill_doc,
        }

    def skill_metadata(self) -> dict[str, Any]:
        """Expose lightweight metadata for suggestions and future skill dirs."""

        return {
            "name": self.name,
            "description": self.description,
            "aliases": list(self.aliases),
            "triggers": list(self.triggers),
            "supported_task_types": list(self.supported_task_types),
            "visible_in_suggestions": self.visible_in_suggestions,
            "skill_package": self.skill_package_metadata(),
        }

    @abstractmethod
    def run(self, state: SessionState) -> Any:
        """Execute the skill against the current session state."""
