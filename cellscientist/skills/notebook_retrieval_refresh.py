from __future__ import annotations

import os

from ..legacy.notebook_retrieval_bridge import bridge_refresh_notebook_retrieval
from ..runtime.notebook_models import NotebookArtifact
from ..runtime.state import Artifact, SessionState
from .base import BaseSkill


class NotebookRetrievalRefreshSkill(BaseSkill):
    """Refresh biology evidence for the current notebook context."""

    name = "notebook-retrieval-refresh"

    def run(self, state: SessionState) -> dict[str, object]:
        latest_notebook = state.last_notebook_artifact
        latest_run = state.last_notebook_run_result

        result = bridge_refresh_notebook_retrieval(
            state.user_query,
            preferred_notebook_path=latest_notebook.path if latest_notebook else None,
            preferred_trial_dir=latest_notebook.trial_dir if latest_notebook else None,
            preferred_run_result=latest_run,
            source_artifact_metadata=latest_notebook.metadata if latest_notebook else None,
        )

        details = result.get("details") or {}
        evidence_ids = [str(eid) for eid in (details.get("evidence_ids") or []) if eid]
        for evidence_id in evidence_ids:
            if evidence_id not in state.evidence_ids:
                state.evidence_ids.append(evidence_id)

        target_notebook_path = result.get("target_notebook_path")
        target_trial_dir = details.get("target_trial_dir")
        if target_notebook_path or target_trial_dir:
            existing_artifact = state.last_notebook_artifact
            resolved_path = str(target_notebook_path) if target_notebook_path else (
                existing_artifact.path if existing_artifact is not None else None
            )
            resolved_trial_dir = str(target_trial_dir) if target_trial_dir else (
                existing_artifact.trial_dir if existing_artifact is not None else None
            )
            resolved_name = (
                os.path.basename(resolved_path)
                if resolved_path
                else (existing_artifact.name if existing_artifact is not None else "review_context_notebook")
            )
            state.last_notebook_artifact = NotebookArtifact(
                name=resolved_name,
                path=resolved_path,
                trial_dir=resolved_trial_dir,
                source="legacy_retrieval_context",
                metadata={
                    "status": result.get("status"),
                    "legacy_entry": result.get("legacy_entry"),
                    "evidence_count": result.get("evidence_count"),
                },
            )

        evidence_md_path = details.get("external_knowledge_md_path")
        artifact_name = (
            os.path.basename(str(evidence_md_path))
            if evidence_md_path
            else "external_knowledge_review.md"
        )
        state.artifacts.append(
            Artifact(
                type="evidence_refresh",
                name=artifact_name,
                content=result,
                metadata={
                    "skill": self.name,
                    "status": result.get("status"),
                    "target_notebook_path": result.get("target_notebook_path"),
                    "trial_dir": details.get("target_trial_dir"),
                    "evidence_count": result.get("evidence_count"),
                    "evidence_ids": evidence_ids,
                    "external_knowledge_json_path": details.get("external_knowledge_json_path"),
                    "external_knowledge_md_path": evidence_md_path,
                    "legacy_entry": result.get("legacy_entry"),
                    "source": "legacy_retrieval",
                },
            )
        )

        state.notes.append(
            f"retrieval_refresh:{result.get('status')}:{result.get('evidence_count', 0)}"
        )
        return result
