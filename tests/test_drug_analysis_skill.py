from __future__ import annotations

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.skills.drug_analysis import DrugAnalysisSkill
from cellscientist.skills.drug_info import DrugInfoSkill


def _build_orchestrator() -> OrchestratorV2:
    return OrchestratorV2(SkillRegistry([DrugAnalysisSkill(), DrugInfoSkill()]))


def test_drug_analysis_skill_returns_structured_sections_for_drug_name_query():
    import cellscientist.skills.drug_analysis as drug_analysis_module

    original = drug_analysis_module._run_biokb_analysis
    drug_analysis_module._run_biokb_analysis = lambda _smiles, workspace_dir: {
        "semantic_table": {"summary": {"total_targets": 1}},
        "evidence": [
            {
                "id": "B1",
                "source": "biokb",
                "claim": "Structure-native prior points to energy metabolism signaling.",
                "citation": "BioKB synthetic test record",
                "confidence": 0.7,
                "metadata": {},
            }
        ],
        "targets": ["AMPK signaling axis"],
        "pathways": ["AMPK activation"],
        "processes": ["energy homeostasis"],
        "workspace_dir": workspace_dir,
        "status": "biokb_ready",
    }
    try:
        state, result = _build_orchestrator().run("分析一下 metformin 的靶点、机制和安全性")
    finally:
        drug_analysis_module._run_biokb_analysis = original

    assert state.intent is not None
    assert state.intent.task_type == "drug_analysis"
    assert state.skill_trace == ["drug_analysis:drug-analysis"]
    assert result["task"] == "drug_analysis"
    assert result["input_type"] == "drug_name"
    assert result["normalized_entity"]["name"] == "metformin"
    assert result["mechanism"]
    assert result["targets"]
    assert result["safety"]
    assert result["evidence"]
    assert result["next_questions"]
    assert any(artifact.type == "drug_analysis" for artifact in state.artifacts)


def test_drug_analysis_skill_normalizes_smiles_query_without_crashing():
    import cellscientist.skills.drug_analysis as drug_analysis_module

    original = drug_analysis_module._run_biokb_analysis
    drug_analysis_module._run_biokb_analysis = lambda _smiles, workspace_dir: {
        "semantic_table": {"summary": {"total_targets": 0}},
        "evidence": [],
        "targets": [],
        "pathways": [],
        "processes": [],
        "workspace_dir": workspace_dir,
        "status": "biokb_ready",
    }
    try:
        state, result = _build_orchestrator().run(
            "根据这个 SMILES 做药物分析: CN(C)C(=N)NC(=N)N"
        )
    finally:
        drug_analysis_module._run_biokb_analysis = original

    assert state.intent is not None
    assert state.intent.task_type == "drug_analysis"
    assert result["task"] == "drug_analysis"
    assert result["input_type"] == "smiles"
    assert result["normalized_entity"]["canonical_smiles"] == "CN(C)C(=N)NC(=N)N"
    assert result["normalized_entity"]["normalization_status"] in {
        "matched_seeded_drug_from_smiles",
        "canonicalized_smiles_only",
    }
    assert "mechanism" in result
    assert "targets" in result
    assert "safety" in result


def test_drug_analysis_can_emit_notebook_ready_scaffold_without_routing_to_notebook():
    import cellscientist.skills.drug_analysis as drug_analysis_module

    original = drug_analysis_module._run_biokb_analysis
    drug_analysis_module._run_biokb_analysis = lambda _smiles, workspace_dir: {
        "semantic_table": {"summary": {"total_targets": 1}},
        "evidence": [],
        "targets": ["AMPK signaling axis"],
        "pathways": ["AMPK activation"],
        "processes": ["energy homeostasis"],
        "workspace_dir": workspace_dir,
        "status": "biokb_ready",
    }
    try:
        state, result = _build_orchestrator().run(
            "分析一下 metformin 的机制和安全性，并给我一个后续验证 notebook"
        )
    finally:
        drug_analysis_module._run_biokb_analysis = original

    assert state.intent is not None
    assert state.intent.task_type == "drug_analysis"
    assert state.skill_trace == ["drug_analysis:drug-analysis"]
    assert result["notebook_ready"] is True
    assert "experiment_scaffold" in result
    scaffold = result["experiment_scaffold"]
    assert scaffold["handoff"]["auto_execute"] is False
    assert scaffold["handoff"]["mode"] == "scaffold_only"
    assert any(artifact.type == "experiment_scaffold" for artifact in state.artifacts)
