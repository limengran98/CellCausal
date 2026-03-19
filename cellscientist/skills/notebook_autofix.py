from __future__ import annotations

import os
from dataclasses import asdict

from ..legacy.notebook_repair_bridge import bridge_autofix_notebook
from ..runtime.notebook_models import NotebookArtifact
from ..runtime.state import Artifact, SessionState
from .base import BaseSkill


class NotebookAutofixSkill(BaseSkill):
    """External autofix wrapper for an existing failed notebook run."""

    name = "notebook-autofix"

    def run(self, state: SessionState) -> dict[str, object]:
        latest_notebook = state.last_notebook_artifact
        latest_run = state.last_notebook_run_result

        result = bridge_autofix_notebook(
            state.user_query,
            preferred_notebook_path=latest_notebook.path if latest_notebook else None,
            preferred_trial_dir=latest_notebook.trial_dir if latest_notebook else None,
            preferred_run_result=latest_run,
            source_artifact_metadata=latest_notebook.metadata if latest_notebook else None,
        )

        patched_notebook_path = result.get("patched_notebook_path")
        if patched_notebook_path:
            trial_dir = (result.get("details") or {}).get("target_trial_dir")
            patched_artifact = NotebookArtifact(
                name=os.path.basename(str(patched_notebook_path)),
                path=str(patched_notebook_path),
                trial_dir=str(trial_dir) if trial_dir else None,
                source="legacy_autofix",
                metadata={
                    "skill": self.name,
                    "status": result.get("status"),
                    "legacy_entry": result.get("legacy_entry"),
                    "target_notebook_path": result.get("target_notebook_path"),
                    "error_log_path": result.get("error_log_path"),
                },
            )
            state.last_notebook_artifact = patched_artifact
            state.artifacts.append(
                Artifact(
                    type="notebook",
                    name=patched_artifact.name,
                    content=asdict(patched_artifact),
                    metadata={
                        "skill": self.name,
                        "status": result.get("status"),
                        "path": patched_artifact.path,
                        "trial_dir": patched_artifact.trial_dir,
                        "source": patched_artifact.source,
                        "legacy_entry": result.get("legacy_entry"),
                        "target_notebook_path": result.get("target_notebook_path"),
                        "error_log_path": result.get("error_log_path"),
                    },
                )
            )

        return result
