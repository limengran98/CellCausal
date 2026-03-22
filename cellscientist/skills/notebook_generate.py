from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from ..legacy.notebook_bridge import bridge_generate_notebook
from ..runtime.notebook_models import NotebookArtifact
from ..runtime.state import Artifact, SessionState
from .base import BaseSkill


class NotebookGenerateSkill(BaseSkill):
    """Thin bridge from notebook-generate into the legacy notebook generation path."""

    name = "notebook-generate"

    def run(self, state: SessionState) -> dict[str, object]:
        if state.intent is not None and state.intent.task_type == "data_analysis":
            result = self._generate_generic_notebook(state)
        else:
            result = bridge_generate_notebook(state.user_query)
        notebook_path = result.get("notebook_path")
        trial_dir = result.get("trial_dir") or (result.get("details") or {}).get("trial_dir")
        final_provider = ((result.get("details") or {}).get("final_provider_used") or {})
        notebook_source = "generic_data" if state.intent is not None and state.intent.task_type == "data_analysis" else "legacy"
        artifact_name = os.path.basename(str(notebook_path)) if notebook_path else "draft_notebook"
        notebook_artifact = NotebookArtifact(
            name=artifact_name,
            path=str(notebook_path) if notebook_path else None,
            trial_dir=str(trial_dir) if trial_dir else None,
            source=notebook_source,
            metadata={
                "skill": self.name,
                "status": result["status"],
                "legacy_entry": result.get("legacy_entry"),
                "provider_name": final_provider.get("provider_name") or final_provider.get("name"),
                "provider_source": final_provider.get("source") or final_provider.get("config_source"),
                "workflow_path": (result.get("details") or {}).get("workflow_path"),
                "input_path": (result.get("details") or {}).get("input_path"),
                "file_type": (result.get("details") or {}).get("file_type"),
                "shape": (result.get("details") or {}).get("shape"),
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
                    "source": notebook_source,
                    "legacy_entry": result.get("legacy_entry"),
                    "provider_name": final_provider.get("provider_name") or final_provider.get("name"),
                    "provider_source": final_provider.get("source") or final_provider.get("config_source"),
                    "workflow_path": (result.get("details") or {}).get("workflow_path"),
                    "input_path": (result.get("details") or {}).get("input_path"),
                    "file_type": (result.get("details") or {}).get("file_type"),
                    "shape": (result.get("details") or {}).get("shape"),
                },
            )
        )
        return result

    @staticmethod
    def _latest_artifact_content(state: SessionState, artifact_type: str) -> dict[str, object] | None:
        for artifact in reversed(state.artifacts):
            if artifact.type == artifact_type and isinstance(artifact.content, dict):
                return artifact.content
        return None

    def _generate_generic_notebook(self, state: SessionState) -> dict[str, object]:
        data_profile = self._latest_artifact_content(state, "data_profile")
        analysis_plan = self._latest_artifact_content(state, "analysis_plan")

        if data_profile is None:
            return {
                "action": "generate",
                "status": "generic_generation_missing_profile",
                "message": "Generic notebook generation needs a data profile step first.",
                "query": state.user_query,
                "notebook_path": None,
                "legacy_entry": None,
                "details": {"workflow_path": "generic_data"},
            }

        input_path = str(data_profile.get("input_path") or "")
        file_type = str(data_profile.get("file_type") or "unknown")
        shape = data_profile.get("shape")
        recommended_steps = list((analysis_plan or {}).get("recommended_steps") or [])
        trial_dir = Path("results") / "generic_data" / f"generic_data_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = trial_dir / "generic_analysis_notebook.ipynb"

        notebook_doc = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": (
                        "# Generic Data Analysis Notebook\n\n"
                        "This notebook was generated by the minimal generic data intake path. "
                        "It validates user-provided tabular data without assuming the BBBC036 legacy recipe."
                    ),
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": self._build_load_snippet(input_path, file_type),
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": (
                        "summary = {\n"
                        "    'shape': tuple(df.shape),\n"
                        "    'columns': df.columns.tolist(),\n"
                        "    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},\n"
                        "    'missingness': df.isna().sum().to_dict(),\n"
                        "}\n"
                        "summary"
                    ),
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": (
                        "## Suggested Next Steps\n\n"
                        + "\n".join(f"- {step}" for step in recommended_steps)
                        if recommended_steps
                        else "## Suggested Next Steps\n\n- Review the dataset summary and define the main analysis question."
                    ),
                },
            ],
        }

        with open(notebook_path, "w", encoding="utf-8") as handle:
            json.dump(notebook_doc, handle, ensure_ascii=False, indent=2)

        with open(trial_dir / "data_profile.json", "w", encoding="utf-8") as handle:
            json.dump(data_profile, handle, ensure_ascii=False, indent=2)
        if analysis_plan is not None:
            with open(trial_dir / "analysis_plan.json", "w", encoding="utf-8") as handle:
                json.dump(analysis_plan, handle, ensure_ascii=False, indent=2)

        return {
            "action": "generate",
            "status": "generated_generic_notebook",
            "message": "Generated a lightweight generic analysis notebook from the profiled user dataset.",
            "query": state.user_query,
            "notebook_path": str(notebook_path.resolve()),
            "trial_dir": str(trial_dir.resolve()),
            "legacy_entry": None,
            "details": {
                "workflow_path": "generic_data",
                "input_path": input_path,
                "file_type": file_type,
                "shape": shape,
                "suggested_analysis_modes": list(data_profile.get("suggested_analysis_modes") or []),
                "recommended_steps": recommended_steps,
            },
        }

    @staticmethod
    def _build_load_snippet(input_path: str, file_type: str) -> str:
        if file_type == "tsv":
            reader = f"df = pd.read_csv({input_path!r}, sep='\\t')"
        elif file_type == "parquet":
            reader = f"df = pd.read_parquet({input_path!r})"
        elif file_type in {"xlsx", "xls"}:
            reader = f"df = pd.read_excel({input_path!r})"
        else:
            reader = f"df = pd.read_csv({input_path!r})"

        return "import pandas as pd\n\n" + reader + "\n" + "df.head()"
