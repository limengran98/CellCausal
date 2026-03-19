from __future__ import annotations

from typing import Optional

from ..runtime.notebook_models import NotebookArtifact
from ..runtime.state import SessionState
from .base import BaseSkill
from .notebook_execute import NotebookExecuteSkill
from .notebook_generate import NotebookGenerateSkill

_GENERATE_KEYWORDS = (
    "生成",
    "create",
    "generate",
    "设计",
    "design notebook",
    "实验设计",
)

_EXECUTE_KEYWORDS = (
    "执行",
    "run",
    "execute",
    "运行 notebook",
    "review",
    "修改",
    "modify",
    "修复",
    "fix",
    "autofix",
)


class NotebookWorkflowSkill(BaseSkill):
    """Route legacy notebook requests into the minimal notebook skill family."""

    name = "notebook-workflow"
    supported_task_types = ["legacy_notebook"]

    def __init__(self) -> None:
        self._generate_skill = NotebookGenerateSkill()
        self._execute_skill = NotebookExecuteSkill()

    def run(self, state: SessionState) -> dict[str, str]:
        self._ensure_trace(state, self.name)

        action = self._resolve_action(state.user_query)
        if action == "execute":
            latest_notebook = self._get_latest_notebook_artifact(state)
            if latest_notebook is not None:
                state.last_notebook_artifact = latest_notebook
                state.notes.append(
                    f"notebook_context:{latest_notebook.path or latest_notebook.name}"
                )
        delegated_skill = (
            self._execute_skill if action == "execute" else self._generate_skill
        )
        self._ensure_trace(state, delegated_skill.name)
        return delegated_skill.run(state)

    @staticmethod
    def _resolve_action(query: str) -> str:
        lowered = query.strip().lower()
        if any(keyword in lowered for keyword in _EXECUTE_KEYWORDS):
            return "execute"
        if any(keyword in lowered for keyword in _GENERATE_KEYWORDS):
            return "generate"
        return "generate"

    @staticmethod
    def _ensure_trace(state: SessionState, skill_name: str) -> None:
        task_type = state.intent.task_type if state.intent is not None else "legacy_notebook"
        trace_entry = f"{task_type}:{skill_name}"
        if trace_entry not in state.skill_trace:
            state.skill_trace.append(trace_entry)

    @staticmethod
    def _get_latest_notebook_artifact(state: SessionState) -> Optional[NotebookArtifact]:
        if state.last_notebook_artifact is not None:
            return state.last_notebook_artifact

        for artifact in reversed(state.artifacts):
            if artifact.type != "notebook":
                continue

            content = artifact.content if isinstance(artifact.content, dict) else {}
            metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
            return NotebookArtifact(
                name=str(content.get("name") or artifact.name),
                path=str(content.get("path") or metadata.get("path")) if (content.get("path") or metadata.get("path")) else None,
                trial_dir=str(content.get("trial_dir") or metadata.get("trial_dir")) if (content.get("trial_dir") or metadata.get("trial_dir")) else None,
                source=str(content.get("source") or metadata.get("source") or "unknown"),
                metadata=dict(content.get("metadata") or metadata),
            )

        return None
