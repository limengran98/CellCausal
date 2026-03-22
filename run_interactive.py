from __future__ import annotations

import argparse
import json
from typing import Any

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.runtime.run_manifest import record_runtime_run
from cellscientist.skills.drug_analysis import DrugAnalysisSkill
from cellscientist.skills.drug_info import DrugInfoSkill
from cellscientist.skills.legacy_notebook import LegacyNotebookSkill
from cellscientist.skills.notebook_workflow import NotebookWorkflowSkill


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the minimal V2 interactive entrypoint."""

    parser = argparse.ArgumentParser(description="Run the minimal CellScientist V2 entrypoint.")
    parser.add_argument("--query", required=True, help="User query to send into the V2 runtime.")
    return parser


def build_payload(state: Any, result: Any) -> dict[str, Any]:
    """Convert runtime objects into a stable JSON payload."""

    task_type = state.intent.task_type if state.intent is not None else "unknown"
    requested_actions = state.intent.requested_actions if state.intent is not None else []
    secondary_task_hints = state.intent.secondary_task_hints if state.intent is not None else []
    artifacts = [
        {
            "type": artifact.type,
            "name": artifact.name,
            "metadata": artifact.metadata,
        }
        for artifact in state.artifacts
    ]
    return {
        "session_id": state.session_id,
        "task_type": task_type,
        "requested_actions": requested_actions,
        "secondary_task_hints": secondary_task_hints,
        "skill_trace": state.skill_trace,
        "artifacts": artifacts,
        "result": result,
    }


def main() -> None:
    """Run the V2 interactive entrypoint."""

    args = build_parser().parse_args()

    skill_registry = SkillRegistry()
    skill_registry.register(DrugAnalysisSkill())
    skill_registry.register(DrugInfoSkill())
    skill_registry.register(NotebookWorkflowSkill())
    skill_registry.register(LegacyNotebookSkill())

    orchestrator = OrchestratorV2(skill_registry)
    state, result = orchestrator.run(args.query)

    try:
        record_runtime_run(
            run_id=state.session_id,
            query=args.query,
            state=state,
            result=result,
            skill_catalog=skill_registry.skill_catalog(),
        )
    except Exception as exc:
        state.notes.append(f"run_manifest_failed:{type(exc).__name__}")

    payload = build_payload(state, result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
