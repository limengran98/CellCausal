from __future__ import annotations

from typing import Optional

from ..runtime.notebook_models import NotebookArtifact, NotebookRunResult
from ..runtime.state import ResearchIntent
from ..runtime.state import SessionState
from .base import BaseSkill
from .notebook_autofix import NotebookAutofixSkill
from .notebook_execute import NotebookExecuteSkill
from .notebook_generate import NotebookGenerateSkill
from .notebook_retrieval_refresh import NotebookRetrievalRefreshSkill
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

_RETRIEVAL_REFRESH_KEYWORDS = (
    "retrieval refresh",
    "refresh evidence",
    "挖掘生物知识",
    "补充生物学证据",
    "补充证据",
    "检索更多信息",
    "检索更多药物",
    "检索更多通路",
    "检索更多靶点",
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
    """Route notebook-family requests across repo-native subskills.

    This treats notebook work as a small skill tree over shared artifacts and
    workspace state, rather than forcing every request back into one fixed
    pipeline stage.
    """

    name = "notebook-workflow"
    description = "Notebook family router for evidence refresh, generation, review, execution, and external autofix."
    aliases = ["notebook", "notebook-workflow", "experiment-design"]
    triggers = ["实验设计", "review notebook", "执行 notebook", "autofix notebook", "补充生物学证据"]
    supported_task_types = ["legacy_notebook"]

    def __init__(self) -> None:
        self._generate_skill = NotebookGenerateSkill()
        self._retrieval_refresh_skill = NotebookRetrievalRefreshSkill()
        self._execute_skill = NotebookExecuteSkill()
        self._review_skill = NotebookReviewSkill()
        self._autofix_skill = NotebookAutofixSkill()

    def run(self, state: SessionState) -> dict[str, object]:
        self._ensure_trace(state, self.name)
        requested_actions = self._resolve_requested_actions(state.intent, state.user_query)
        if len(requested_actions) > 1:
            return self._run_multi_step(state, requested_actions)

        action = requested_actions[0]
        if action in {"retrieval_refresh", "execute", "review", "autofix"}:
            self._prime_notebook_context(state)

        delegated_skill = self._skill_for_action(action)
        self._ensure_trace(state, delegated_skill.name)
        return delegated_skill.run(state)

    def _run_multi_step(self, state: SessionState, actions: list[str]) -> dict[str, object]:
        step_results: list[dict[str, object]] = []

        for action in actions:
            if action in {"retrieval_refresh", "execute", "review", "autofix"}:
                self._prime_notebook_context(state)

            delegated_skill = self._skill_for_action(action)
            self._ensure_trace(state, delegated_skill.name)
            try:
                step_result = delegated_skill.run(state)
            except Exception as exc:
                step_result = {
                    "action": action,
                    "status": f"{action}_failed",
                    "message": f"The '{action}' step failed inside notebook-workflow.",
                    "details": {"error": str(exc)},
                }
            step_results.append(step_result)

        return {
            "action": "multi_step",
            "requested_actions": actions,
            "status": self._summarize_multi_step_status(step_results),
            "message": self._build_multi_step_message(step_results),
            "step_results": step_results,
        }

    def _skill_for_action(self, action: str) -> BaseSkill:
        if action == "retrieval_refresh":
            return self._retrieval_refresh_skill
        if action == "execute":
            return self._execute_skill
        if action == "review":
            return self._review_skill
        if action == "autofix":
            return self._autofix_skill
        return self._generate_skill

    @staticmethod
    def _resolve_action(query: str) -> str:
        lowered = query.strip().lower()
        if any(keyword in lowered for keyword in _AUTOFIX_KEYWORDS):
            return "autofix"
        if any(keyword in lowered for keyword in _RETRIEVAL_REFRESH_KEYWORDS):
            return "retrieval_refresh"
        if any(keyword in lowered for keyword in _REVIEW_KEYWORDS):
            return "review"
        if any(keyword in lowered for keyword in _EXECUTE_KEYWORDS):
            return "execute"
        if any(keyword in lowered for keyword in _GENERATE_KEYWORDS):
            return "generate"
        return "generate"

    @classmethod
    def _resolve_requested_actions(
        cls,
        intent: Optional[ResearchIntent],
        query: str,
    ) -> list[str]:
        requested_actions = list(intent.requested_actions) if intent is not None else []
        valid_actions = [
            action
            for action in requested_actions
            if action in {"generate", "retrieval_refresh", "review", "execute", "autofix"}
        ]
        if valid_actions:
            return valid_actions
        return [cls._resolve_action(query)]

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

    @staticmethod
    def _summarize_multi_step_status(step_results: list[dict[str, object]]) -> str:
        statuses = [str(step.get("status") or "unknown") for step in step_results]
        if statuses and all(
            not any(token in status for token in ("failed", "missing", "needs", "blocked"))
            for status in statuses
        ):
            return "completed"
        if any(any(token in status for token in ("failed", "missing", "needs", "blocked")) for status in statuses):
            return "completed_with_partial_blocks"
        return "completed"

    @staticmethod
    def _build_multi_step_message(step_results: list[dict[str, object]]) -> str:
        blocked = [
            str(step.get("action") or "unknown")
            for step in step_results
            if any(token in str(step.get("status") or "") for token in ("failed", "missing", "needs", "blocked"))
        ]
        if blocked:
            return (
                "Notebook workflow executed multiple requested actions, but some steps were partially blocked: "
                + ", ".join(blocked)
            )
        return "Notebook workflow executed the requested actions in sequence."
