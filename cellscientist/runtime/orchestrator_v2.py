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

        skill = self.skill_registry.resolve(state.intent)
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
