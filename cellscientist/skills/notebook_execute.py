from __future__ import annotations

import os
from dataclasses import asdict

from ..legacy.notebook_bridge import bridge_execute_notebook
from ..runtime.notebook_models import NotebookArtifact, NotebookRunResult
from ..runtime.state import Artifact, SessionState
from .base import BaseSkill


class NotebookExecuteSkill(BaseSkill):
    """Thin bridge from notebook-execute into the legacy notebook execution path."""

    name = "notebook-execute"

    def run(self, state: SessionState) -> dict[str, object]:
        latest_notebook = state.last_notebook_artifact
        result = bridge_execute_notebook(
            state.user_query,
            preferred_notebook_path=latest_notebook.path if latest_notebook else None,
            preferred_trial_dir=latest_notebook.trial_dir if latest_notebook else None,
        )
        notebook_source = latest_notebook.source if latest_notebook is not None else "legacy"

        run_result = NotebookRunResult(
            notebook_path=str(result.get("notebook_path")) if result.get("notebook_path") else None,
            trial_dir=str(result.get("trial_dir")) if result.get("trial_dir") else None,
            status=str(result.get("status") or "unknown"),
            error_log_path=str(result.get("error_log_path")) if result.get("error_log_path") else None,
            run_log_path=str(result.get("run_log_path")) if result.get("run_log_path") else None,
            metadata={
                "skill": self.name,
                "legacy_entry": result.get("legacy_entry"),
                "query": state.user_query,
            },
        )
        state.last_notebook_run_result = run_result

        state.artifacts.append(
            Artifact(
                type="notebook_run",
                name=os.path.basename(run_result.notebook_path) if run_result.notebook_path else "notebook_run",
                content=asdict(run_result),
                metadata={
                    "skill": self.name,
                    "status": run_result.status,
                    "notebook_path": run_result.notebook_path,
                    "trial_dir": run_result.trial_dir,
                    "error_log_path": run_result.error_log_path,
                    "run_log_path": run_result.run_log_path,
                    "legacy_entry": result.get("legacy_entry"),
                    "source": notebook_source,
                },
            )
        )

        if run_result.notebook_path:
            state.last_notebook_artifact = NotebookArtifact(
                name=os.path.basename(run_result.notebook_path),
                path=run_result.notebook_path,
                trial_dir=run_result.trial_dir,
                source=notebook_source,
                metadata={
                    "status": run_result.status,
                    "legacy_entry": result.get("legacy_entry"),
                    "derived_from": "notebook-execute",
                },
            )

        return result
