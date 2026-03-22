from __future__ import annotations

from typing import Any

from ..runtime.state import Artifact, SessionState
from .base import BaseSkill


class AnalysisPlanSkill(BaseSkill):
    """Rule-based planning step between data profiling and notebook generation."""

    name = "analysis-plan"
    description = "Create a minimal analysis plan from a profiled user dataset."
    aliases = ["analysis-plan", "analysis plan", "plan analysis"]
    triggers = ["分析计划", "探索分析", "验证", "run notebook on my data"]
    visible_in_suggestions = False

    def run(self, state: SessionState) -> dict[str, object]:
        profile = self._latest_data_profile(state)
        if profile is None:
            return {
                "task": "analysis_plan",
                "status": "analysis_plan_missing_profile",
                "problem_type": "unknown",
                "recommended_steps": [],
                "recommended_artifacts": [],
                "notebook_needed": False,
                "notes": [
                    "A data profile artifact is required before a generic analysis plan can be formed."
                ],
                "workflow_path": "generic_data",
            }

        problem_type = self._infer_problem_type(state.user_query, profile)
        recommended_steps = self._recommended_steps(problem_type, profile)
        recommended_artifacts = [
            "data_profile",
            "analysis_plan",
            "analysis_notebook",
            "run_manifest",
        ]
        notebook_needed = self._query_requests_notebook(state.user_query)

        result = {
            "task": "analysis_plan",
            "status": "planned",
            "problem_type": problem_type,
            "recommended_steps": recommended_steps,
            "recommended_artifacts": recommended_artifacts,
            "notebook_needed": notebook_needed,
            "notes": self._build_notes(profile, problem_type),
            "workflow_path": "generic_data",
            "input_path": profile.get("input_path"),
            "file_type": profile.get("file_type"),
            "shape": profile.get("shape"),
        }

        state.notes.append("workflow_path:generic_data")
        state.artifacts.append(
            Artifact(
                type="analysis_plan",
                name="generic_analysis_plan",
                content=result,
                metadata={
                    "skill": self.name,
                    "workflow_path": "generic_data",
                    "problem_type": problem_type,
                    "notebook_needed": notebook_needed,
                    "input_path": profile.get("input_path"),
                    "shape": profile.get("shape"),
                    "status": "planned",
                },
            )
        )
        return result

    @staticmethod
    def _latest_data_profile(state: SessionState) -> dict[str, Any] | None:
        for artifact in reversed(state.artifacts):
            if artifact.type == "data_profile" and isinstance(artifact.content, dict):
                return artifact.content
        return None

    @staticmethod
    def _infer_problem_type(query: str, profile: dict[str, Any]) -> str:
        lowered = query.lower()
        column_types = profile.get("column_types") or {}
        columns = [str(column).lower() for column in (profile.get("columns") or [])]
        suggested_modes = set(profile.get("suggested_analysis_modes") or [])

        if any(keyword in lowered for keyword in ("classification", "分类")) or "baseline_classification" in suggested_modes:
            return "classification"
        if any(keyword in lowered for keyword in ("regression", "回归")) or "baseline_regression" in suggested_modes:
            return "regression"
        if any(keyword in lowered for keyword in ("time series", "时序")) or "time_series_screen" in suggested_modes:
            return "time_series_exploration"
        if any(keyword in columns for keyword in ("label", "class", "target", "response", "outcome")):
            target_column = next(
                (column for column in columns if column in {"label", "class", "target", "response", "outcome"}),
                None,
            )
            if target_column and "float" in str(column_types.get(target_column, "")).lower():
                return "regression"
            return "classification"
        return "tabular_exploration"

    @staticmethod
    def _recommended_steps(problem_type: str, profile: dict[str, Any]) -> list[str]:
        steps = [
            "Validate file loading and confirm row/column counts.",
            "Summarize column types, missingness, and a few representative rows.",
            "Check obvious data quality issues such as duplicated keys or empty columns.",
        ]
        if problem_type == "classification":
            steps.extend(
                [
                    "Inspect label balance and candidate feature columns.",
                    "Prepare a simple baseline classification notebook section.",
                ]
            )
        elif problem_type == "regression":
            steps.extend(
                [
                    "Inspect target distribution and feature scale ranges.",
                    "Prepare a simple baseline regression notebook section.",
                ]
            )
        else:
            steps.extend(
                [
                    "Run exploratory summaries, univariate plots, and grouped comparisons where applicable.",
                    "Reserve a notebook section for task-specific follow-up analysis once the user clarifies the main question.",
                ]
            )
        if profile.get("missingness_summary", {}).get("columns_with_missing"):
            steps.append("Add explicit missing-value handling notes to the notebook.")
        return steps

    @staticmethod
    def _build_notes(profile: dict[str, Any], problem_type: str) -> list[str]:
        notes = [
            f"Planned around file type '{profile.get('file_type')}' and shape {profile.get('shape')}.",
            f"Current rule-based inference labels the task as '{problem_type}'.",
        ]
        if profile.get("suggested_analysis_modes"):
            notes.append(
                "Suggested modes from profiling: "
                + ", ".join(str(mode) for mode in profile.get("suggested_analysis_modes") or [])
            )
        return notes

    @staticmethod
    def _query_requests_notebook(query: str) -> bool:
        lowered = query.lower()
        return any(
            keyword in lowered
            for keyword in (
                "notebook",
                "生成 notebook",
                "生成一个 notebook",
                "run a notebook",
                "跑一个 notebook",
                "执行 notebook",
            )
        )
