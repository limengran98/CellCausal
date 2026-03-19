from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.skills.drug_info import DrugInfoSkill
from cellscientist.tools.drug_lookup import lookup_drug_profile


def test_lookup_drug_profile_returns_structured_evidence_for_metformin():
    profile = lookup_drug_profile("metformin")

    assert profile["drug_name"] == "metformin"
    assert profile["targets"]
    assert profile["indications"]
    assert profile["adverse_effects"]
    assert len(profile["evidence"]) >= 1
    assert {"id", "source", "claim", "citation", "confidence"} <= set(profile["evidence"][0])


def test_drug_info_skill_emits_structured_result_and_artifact():
    orchestrator = OrchestratorV2(SkillRegistry([DrugInfoSkill()]))

    state, result = orchestrator.run("metformin 的靶点和适应症")

    assert state.intent is not None
    assert state.intent.task_type == "drug_info"
    assert state.skill_trace == ["drug_info:drug-info"]
    assert result["task"] == "drug_info"
    assert result["drug_name"] == "metformin"
    assert "Placeholder" not in result["summary"]
    assert len(result["evidence"]) >= 1
    assert state.evidence_ids
    assert any(artifact.type == "drug_profile" for artifact in state.artifacts)
