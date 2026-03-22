from __future__ import annotations

from pathlib import Path

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.skills.enzyme_mining import EnzymeMiningSkill


def _build_orchestrator() -> OrchestratorV2:
    return OrchestratorV2(SkillRegistry([EnzymeMiningSkill()]))


def test_enzyme_mining_returns_structured_candidate_panel():
    state, result = _build_orchestrator().run("挖一下和脂代谢相关的候选酶，并说明依据")

    assert state.intent is not None
    assert state.intent.task_type == "enzyme_mining"
    assert state.skill_trace == ["enzyme_mining:enzyme-mining"]
    assert result["task"] == "enzyme_mining"
    assert result["substrate_context"]["status"] == "no_substrate_context"
    assert result["substrate_smiles"] is None
    assert result["candidate_sources"]
    assert result["candidate_sequences_status"]["raw_sequence_count"] > 0
    assert result["candidate_sequences_status"]["unique_sequence_count"] > 0
    assert result["candidate_sequences_status"]["unique_sequence_count"] <= result["candidate_sequences_status"]["raw_sequence_count"]
    assert result["candidate_enzymes"]
    assert result["filtering_steps"]
    assert result["ranking_status"]
    assert result["ranking_ready"] is False
    assert result["ranking_model"]["name"] == "CataPro"
    assert isinstance(result["ranking_results"], list)
    assert result["why_not_runnable"]
    assert result["required_assets"]
    assert result["prepared_input_preview"]["status"] in {"preview_ready", "preview_blocked"}
    assert result["pathway_context"]
    assert result["evidence"]
    assert result["next_questions"]
    assert result["notebook_ready"] is False
    enzyme_artifact = next(artifact for artifact in state.artifacts if artifact.type == "enzyme_mining")
    assert enzyme_artifact.metadata["ranking_model_name"] == "CataPro"
    assert enzyme_artifact.metadata["raw_sequence_count"] >= enzyme_artifact.metadata["unique_sequence_count"]


def test_enzyme_mining_can_emit_notebook_ready_scaffold_without_auto_execution():
    state, result = _build_orchestrator().run(
        "挖一下和胆固醇代谢有关的候选酶，并生成一个可验证的 notebook 框架"
    )

    assert state.intent is not None
    assert state.intent.task_type == "enzyme_mining"
    assert result["candidate_sources"]
    assert result["filtering_steps"]
    assert result["ranking_status"]
    assert result["ranking_ready"] is False
    assert result["ranking_model"]["name"] == "CataPro"
    assert isinstance(result["ranking_results"], list)
    assert result["notebook_ready"] is True
    assert "experiment_scaffold" in result
    scaffold = result["experiment_scaffold"]
    assert scaffold["handoff"]["auto_execute"] is False
    assert scaffold["handoff"]["mode"] == "scaffold_only"
    assert "Sequence-ranking validation" in scaffold["recommended_notebook_sections"]
    assert "Substrate-SMILES consistency check" in scaffold["recommended_notebook_sections"]
    assert any("ranking bridge status" in note.lower() for note in scaffold["notes"])
    assert Path(result["artifact_export"]["files"]["experiment_scaffold_json"]).exists()
    assert any(artifact.type == "experiment_scaffold" for artifact in state.artifacts)


def test_enzyme_mining_recognizes_smiles_driven_substrate_ranking_queries():
    state, result = _build_orchestrator().run(
        "根据这个 SMILES 挖可作用的候选酶并排序: C(C(C(=O)O)N)S"
    )

    assert state.intent is not None
    assert state.intent.task_type == "enzyme_mining"
    assert result["substrate_context"]["status"] == "substrate_smiles_provided"
    assert result["substrate_smiles"] == "NC(CS)C(=O)O"
    assert result["ranking_status"] in {
        "awaiting_candidate_sequence_mapping",
        "blocked_missing_runtime_dependencies",
        "blocked_incomplete_model_assets",
        "ranking_input_ready",
        "ranking_completed",
        "ranking_run_failed",
    }
    assert result["ranking_model"]["name"] == "CataPro"
    assert "next_step_instructions" in result


def test_enzyme_mining_exports_standard_result_artifacts():
    state, result = _build_orchestrator().run(
        "根据这个 SMILES 挖可作用的候选酶并排序: C(C(C(=O)O)N)S"
    )

    export = result["artifact_export"]
    result_dir = Path(export["result_dir"])
    assert result_dir.exists()

    files = export["files"]
    required_files = [
        "candidate_sources_json",
        "candidate_sequences_status_json",
        "candidate_table_csv",
        "filtering_steps_json",
        "ranking_status_json",
        "ranking_input_preview_csv",
        "ranking_results_csv",
        "enzyme_mining_result_json",
    ]
    for key in required_files:
        assert key in files
        assert Path(files[key]).exists()

    ranking_status_payload = Path(files["ranking_status_json"]).read_text(encoding="utf-8")
    assert "current_status" in ranking_status_payload
    assert "whether_real_ranking_completed" in ranking_status_payload
    assert "why_not_runnable" in ranking_status_payload
    assert "required_assets" in ranking_status_payload
    artifact_types = {artifact.type for artifact in state.artifacts}
    assert "enzyme_candidate_table" in artifact_types
    assert "enzyme_filtering_summary" in artifact_types
    assert "enzyme_ranking_result" in artifact_types
