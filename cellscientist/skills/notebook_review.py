from __future__ import annotations

import os

from ..legacy.notebook_review_bridge import bridge_review_notebook
from ..runtime.state import Artifact, SessionState
from .base import BaseSkill


class NotebookReviewSkill(BaseSkill):
    """Thin post-processing review skill for the notebook family."""

    name = "notebook-review"

    @staticmethod
    def _get_latest_refreshed_evidence(state: SessionState) -> dict[str, object] | None:
        for artifact in reversed(state.artifacts):
            if artifact.type != "evidence_refresh":
                continue
            if isinstance(artifact.content, dict):
                return artifact.content
            if isinstance(artifact.metadata, dict):
                return {"details": dict(artifact.metadata)}
        return None

    def run(self, state: SessionState) -> dict[str, object]:
        latest_notebook = state.last_notebook_artifact
        latest_run = state.last_notebook_run_result
        refreshed_evidence_context = self._get_latest_refreshed_evidence(state)

        result = bridge_review_notebook(
            state.user_query,
            preferred_notebook_path=latest_notebook.path if latest_notebook else None,
            preferred_trial_dir=latest_notebook.trial_dir if latest_notebook else None,
            preferred_run_result=latest_run,
            source_artifact_metadata=latest_notebook.metadata if latest_notebook else None,
            refreshed_evidence_context=refreshed_evidence_context,
        )

        report_path = result.get("review_report_path")
        if report_path:
            state.artifacts.append(
                Artifact(
                    type="review_report",
                    name=os.path.basename(str(report_path)),
                    content=result,
                    metadata={
                        "skill": self.name,
                        "status": result.get("status"),
                        "report_path": report_path,
                        "target_notebook_path": result.get("target_notebook_path"),
                        "trial_dir": (result.get("details") or {}).get("target_trial_dir"),
                        "legacy_entry": result.get("legacy_entry"),
                        "source": "legacy",
                    },
                )
            )

        return result
