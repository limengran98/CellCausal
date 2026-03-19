from __future__ import annotations

import argparse
import json
from typing import Any

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
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
        "skill_trace": state.skill_trace,
        "artifacts": artifacts,
        "result": result,
    }


def main() -> None:
    """Run the V2 interactive entrypoint."""

    args = build_parser().parse_args()

    skill_registry = SkillRegistry()
    skill_registry.register(DrugInfoSkill())
    skill_registry.register(NotebookWorkflowSkill())
    skill_registry.register(LegacyNotebookSkill())

    orchestrator = OrchestratorV2(skill_registry)
    state, result = orchestrator.run(args.query)

    payload = build_payload(state, result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
