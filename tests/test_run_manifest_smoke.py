from __future__ import annotations

import json
from pathlib import Path

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.runtime.run_manifest import build_run_manifest, write_manifest_json
from cellscientist.skills.drug_info import DrugInfoSkill


def test_run_manifest_can_be_built_for_runtime_and_eval_style_records(tmp_path: Path):
    registry = SkillRegistry([DrugInfoSkill()])
    state, result = OrchestratorV2(registry).run("metformin 的靶点和适应症")

    runtime_manifest = build_run_manifest(
        run_id="smoke-runtime",
        context="test_runtime",
        query=state.user_query,
        state=state,
        result=result,
        skill_catalog=registry.skill_catalog(),
    )
    runtime_path = write_manifest_json(runtime_manifest, tmp_path / "runtime_manifest.json")

    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime_payload["run_id"] == "smoke-runtime"
    assert runtime_payload["task_type"] == "drug_info"
    assert runtime_payload["final_status"] == "ok"
    assert runtime_payload["skill_trace"] == ["drug_info:drug-info"]
    assert runtime_payload["artifact_summary"]
    assert "numpy" in runtime_payload["key_package_versions"]

    eval_manifest = build_run_manifest(
        run_id="smoke-eval",
        context="test_eval",
        query=None,
        state=None,
        result={"status": "completed"},
        skill_catalog=registry.skill_catalog(),
        eval_case_files=["evals/drug_analysis_cases.json"],
        extra={"eval_summary": {"total_cases": 1, "passed_cases": 1, "failed_cases": 0}},
    )
    eval_path = write_manifest_json(eval_manifest, tmp_path / "eval_manifest.json")

    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    assert eval_payload["run_id"] == "smoke-eval"
    assert eval_payload["final_status"] == "completed"
    assert eval_payload["eval_case_files"] == ["evals/drug_analysis_cases.json"]
    assert eval_payload["extra"]["eval_summary"]["passed_cases"] == 1
