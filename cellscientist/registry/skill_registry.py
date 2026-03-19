from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from ..runtime.state import ResearchIntent
from ..skills.base import BaseSkill
from ..skills.defaults import build_default_skills


class SkillRegistry:
    """Minimal score-based registry for V2 skills."""

    def __init__(self, skills: Optional[Iterable[BaseSkill]] = None) -> None:
        self._skills: List[BaseSkill] = list(skills or [])

    def register(self, skill: BaseSkill) -> None:
        self._skills.append(skill)

    def resolve(self, intent: ResearchIntent) -> BaseSkill:
        ranked: List[Tuple[float, BaseSkill]] = sorted(
            ((skill.match(intent), skill) for skill in self._skills),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] <= 0.0:
            registered = ", ".join(skill.name for skill in self._skills) or "<none>"
            raise LookupError(
                f"No skill matched intent task_type='{intent.task_type}'. "
                f"Registered skills: {registered}"
            )
        return ranked[0][1]


def build_minimal_skill_registry() -> SkillRegistry:
    """Create the default V2 skill registry."""

    return SkillRegistry(skills=build_default_skills())
