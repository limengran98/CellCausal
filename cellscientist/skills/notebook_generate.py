from __future__ import annotations

import os
from dataclasses import asdict

from ..legacy.notebook_bridge import bridge_generate_notebook
from ..runtime.notebook_models import NotebookArtifact
from ..runtime.state import Artifact, SessionState
from .base import BaseSkill


class NotebookGenerateSkill(BaseSkill):
    """Thin bridge from notebook-generate into the legacy notebook generation path."""

    name = "notebook-generate"

    def run(self, state: SessionState) -> dict[str, object]:
        result = bridge_generate_notebook(state.user_query)
        notebook_path = result.get("notebook_path")
        trial_dir = result.get("trial_dir") or (result.get("details") or {}).get("trial_dir")
        final_provider = ((result.get("details") or {}).get("final_provider_used") or {})
        artifact_name = os.path.basename(str(notebook_path)) if notebook_path else "draft_notebook"
        notebook_artifact = NotebookArtifact(
            name=artifact_name,
            path=str(notebook_path) if notebook_path else None,
            trial_dir=str(trial_dir) if trial_dir else None,
            source="legacy",
            metadata={
                "skill": self.name,
                "status": result["status"],
                "legacy_entry": result.get("legacy_entry"),
                "provider_name": final_provider.get("provider_name") or final_provider.get("name"),
                "provider_source": final_provider.get("source") or final_provider.get("config_source"),
            },
        )
        state.last_notebook_artifact = notebook_artifact

        state.artifacts.append(
            Artifact(
                type="notebook",
                name=notebook_artifact.name,
                content=asdict(notebook_artifact),
                metadata={
                    "skill": self.name,
                    "status": result["status"],
                    "path": notebook_artifact.path,
                    "trial_dir": notebook_artifact.trial_dir,
                    "source": "legacy",
                    "legacy_entry": result.get("legacy_entry"),
                    "provider_name": final_provider.get("provider_name") or final_provider.get("name"),
                    "provider_source": final_provider.get("source") or final_provider.get("config_source"),
                },
            )
        )
        return result
