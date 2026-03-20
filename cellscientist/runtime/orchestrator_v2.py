from __future__ import annotations

from typing import Any

from ..registry.skill_registry import SkillRegistry
from .planner import build_intent
from .session import create_session
from .state import Artifact, SessionState


class OrchestratorV2:
    """Minimal runtime orchestrator for the new architecture spine."""

    def __init__(self, skill_registry: SkillRegistry) -> None:
        self.skill_registry = skill_registry

    def run(self, query: str) -> tuple[SessionState, Any]:
        """Create session state, resolve a skill, and execute it."""

        state = create_session(query)
        state.intent = build_intent(query)

        try:
            skill = self.skill_registry.resolve(state.intent)
        except LookupError:
            state.notes.append("fallback:needs_clarification")
            return state, self._build_fallback_result(state)
        state.skill_trace.append(f"{state.intent.task_type}:{skill.name}")

        result = skill.run(state)
        self._record_artifacts(state, result)
        return state, result

    @staticmethod
    def _record_artifacts(state: SessionState, result: Any) -> None:
        """Persist Artifact outputs back into session state when present."""

        if isinstance(result, Artifact):
            state.artifacts.append(result)
            return

        if isinstance(result, (list, tuple)):
            artifacts = [item for item in result if isinstance(item, Artifact)]
            state.artifacts.extend(artifacts)

    def _build_fallback_result(self, state: SessionState) -> dict[str, Any]:
        """Return a structured clarification result instead of crashing on unknown."""

        suggestions = self.skill_registry.suggest_skills(state.intent)
        return {
            "task_type": state.intent.task_type if state.intent is not None else "unknown",
            "status": "needs_clarification",
            "message": (
                "I couldn't confidently map this request to one of the current skill families. "
                "Try asking about notebook design/review/execute/autofix, or ask for drug targets, indications, or adverse effects."
            ),
            "requested_actions": list(state.intent.requested_actions) if state.intent is not None else [],
            "suggested_skills": suggestions,
        }
