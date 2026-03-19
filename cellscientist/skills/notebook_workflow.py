from __future__ import annotations

from typing import Optional

from ..runtime.notebook_models import NotebookArtifact, NotebookRunResult
from ..runtime.state import SessionState
from .base import BaseSkill
from .notebook_autofix import NotebookAutofixSkill
from .notebook_execute import NotebookExecuteSkill
from .notebook_generate import NotebookGenerateSkill
from .notebook_review import NotebookReviewSkill

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
)

_REVIEW_KEYWORDS = (
    "review",
    "审查",
    "检查 notebook",
    "优化 notebook",
    "看看 notebook 质量",
)

_AUTOFIX_KEYWORDS = (
    "autofix",
    "修复",
    "fix notebook",
    "修 notebook 报错",
    "修复这个 notebook",
    "修改",
)


class NotebookWorkflowSkill(BaseSkill):
    """Route legacy notebook requests into the minimal notebook skill family."""

    name = "notebook-workflow"
    supported_task_types = ["legacy_notebook"]

    def __init__(self) -> None:
        self._generate_skill = NotebookGenerateSkill()
        self._execute_skill = NotebookExecuteSkill()
        self._review_skill = NotebookReviewSkill()
        self._autofix_skill = NotebookAutofixSkill()

    def run(self, state: SessionState) -> dict[str, object]:
        self._ensure_trace(state, self.name)

        action = self._resolve_action(state.user_query)
        if action in {"execute", "review", "autofix"}:
            self._prime_notebook_context(state)

        if action == "execute":
            delegated_skill = self._execute_skill
        elif action == "review":
            delegated_skill = self._review_skill
        elif action == "autofix":
            delegated_skill = self._autofix_skill
        else:
            delegated_skill = self._generate_skill

        self._ensure_trace(state, delegated_skill.name)
        return delegated_skill.run(state)

    @staticmethod
    def _resolve_action(query: str) -> str:
        lowered = query.strip().lower()
        if any(keyword in lowered for keyword in _AUTOFIX_KEYWORDS):
            return "autofix"
        if any(keyword in lowered for keyword in _REVIEW_KEYWORDS):
            return "review"
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

    @staticmethod
    def _get_latest_notebook_run_result(state: SessionState) -> Optional[NotebookRunResult]:
        if state.last_notebook_run_result is not None:
            return state.last_notebook_run_result

        for artifact in reversed(state.artifacts):
            if artifact.type != "notebook_run":
                continue

            content = artifact.content if isinstance(artifact.content, dict) else {}
            metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
            return NotebookRunResult(
                notebook_path=str(content.get("notebook_path") or metadata.get("notebook_path"))
                if (content.get("notebook_path") or metadata.get("notebook_path"))
                else None,
                trial_dir=str(content.get("trial_dir") or metadata.get("trial_dir"))
                if (content.get("trial_dir") or metadata.get("trial_dir"))
                else None,
                status=str(content.get("status") or metadata.get("status") or "unknown"),
                error_log_path=str(content.get("error_log_path") or metadata.get("error_log_path"))
                if (content.get("error_log_path") or metadata.get("error_log_path"))
                else None,
                run_log_path=str(content.get("run_log_path") or metadata.get("run_log_path"))
                if (content.get("run_log_path") or metadata.get("run_log_path"))
                else None,
                metadata=dict(content.get("metadata") or metadata),
            )

        return None

    @classmethod
    def _prime_notebook_context(cls, state: SessionState) -> None:
        latest_notebook = cls._get_latest_notebook_artifact(state)
        latest_run = cls._get_latest_notebook_run_result(state)

        if latest_notebook is not None:
            state.last_notebook_artifact = latest_notebook
            state.notes.append(f"notebook_context:{latest_notebook.path or latest_notebook.name}")

        if latest_run is not None:
            state.last_notebook_run_result = latest_run
            state.notes.append(
                f"notebook_run_context:{latest_run.notebook_path or latest_run.trial_dir or latest_run.status}"
            )

        if state.last_notebook_artifact is None and latest_run is not None and latest_run.notebook_path:
            state.last_notebook_artifact = NotebookArtifact(
                name=(latest_run.notebook_path.rsplit("/", 1)[-1]),
                path=latest_run.notebook_path,
                trial_dir=latest_run.trial_dir,
                source="derived_from_run",
                metadata={"status": latest_run.status},
            )
