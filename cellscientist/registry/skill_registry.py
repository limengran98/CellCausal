from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

from ..runtime.state import ResearchIntent
from ..skills.base import BaseSkill
from ..skills.defaults import build_default_skills


class SkillRegistry:
    """Minimal score-based registry for V2 skills."""

    def __init__(self, skills: Optional[Iterable[BaseSkill]] = None) -> None:
        self._skills: List[BaseSkill] = list(skills or [])

    def register(self, skill: BaseSkill) -> None:
        self._skills.append(skill)

    def skill_catalog(self) -> List[dict[str, Any]]:
        """Expose repo-native skill metadata plus optional package mapping."""

        return [skill.skill_metadata() for skill in self._skills]

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

    def suggest_skills(self, intent: Optional[ResearchIntent] = None, *, limit: int = 3) -> List[dict[str, Any]]:
        """Return lightweight skill suggestions for clarification fallback."""

        query = (intent.raw_query.lower() if intent is not None else "").strip()
        suggestions: List[Tuple[float, dict[str, Any]]] = []
        for skill in self._skills:
            metadata = skill.skill_metadata()
            if not metadata.get("visible_in_suggestions", True):
                continue

            score = 0.0
            if intent is not None and intent.task_type in metadata.get("supported_task_types", []):
                score += 2.0
            if query:
                searchable_terms = [metadata.get("name", ""), metadata.get("description", "")]
                searchable_terms.extend(metadata.get("aliases", []) or [])
                searchable_terms.extend(metadata.get("triggers", []) or [])
                if any(term and str(term).lower() in query for term in searchable_terms):
                    score += 1.0

            suggestions.append((score, metadata))

        suggestions.sort(key=lambda item: (item[0], item[1].get("name", "")), reverse=True)
        top = [metadata for _, metadata in suggestions[:limit]]
        return [
            {
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "aliases": metadata.get("aliases", []),
                "triggers": metadata.get("triggers", []),
                "skill_package": metadata.get("skill_package", {}),
            }
            for metadata in top
        ]


def build_minimal_skill_registry() -> SkillRegistry:
    """Create the default V2 skill registry."""

    return SkillRegistry(skills=build_default_skills())
