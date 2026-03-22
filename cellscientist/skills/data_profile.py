from __future__ import annotations

import os

from ..runtime.state import Artifact, SessionState
from ..tools.tabular_data import (
    extract_tabular_path_from_text,
    profile_tabular_file,
    resolve_tabular_path,
)
from .base import BaseSkill


class DataProfileSkill(BaseSkill):
    """Minimal tabular data intake and profiling skill for generic notebook flows."""

    name = "data-profile"
    description = "Profile a user-provided tabular dataset before planning or notebook generation."
    aliases = ["data-profile", "data profile", "tabular-profile"]
    triggers = ["csv", "tsv", "parquet", "自己的数据", "数据表", "探索分析"]
    visible_in_suggestions = False

    def run(self, state: SessionState) -> dict[str, object]:
        raw_path = extract_tabular_path_from_text(state.user_query)
        resolved_path = resolve_tabular_path(raw_path, base_dir=os.getcwd())

        if not raw_path or not resolved_path:
            return {
                "task": "data_profile",
                "status": "missing_input_path",
                "input_path": None,
                "file_type": "unknown",
                "shape": None,
                "columns": [],
                "column_types": {},
                "missingness_summary": {},
                "suggested_analysis_modes": [],
                "message": "No supported user-provided data path was found in the query.",
                "workflow_path": "generic_data",
            }

        profile = profile_tabular_file(resolved_path)
        result = {
            "task": "data_profile",
            "message": (
                "Profiled the user-provided dataset for the generic notebook workflow."
                if profile.get("status") == "profiled"
                else profile.get("message") or "The dataset could not be fully profiled."
            ),
            "workflow_path": "generic_data",
            **profile,
        }

        state.notes.append("workflow_path:generic_data")

        state.artifacts.append(
            Artifact(
                type="data_profile",
                name=os.path.basename(resolved_path) if resolved_path else "data_profile",
                content=result,
                metadata={
                    "skill": self.name,
                    "workflow_path": "generic_data",
                    "input_path": result.get("input_path"),
                    "file_type": result.get("file_type"),
                    "shape": result.get("shape"),
                    "status": result.get("status"),
                },
            )
        )
        return result
