from __future__ import annotations

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
    assert result["candidate_enzymes"]
    assert result["pathway_context"]
    assert result["evidence"]
    assert result["next_questions"]
    assert result["notebook_ready"] is False
    assert any(artifact.type == "enzyme_mining" for artifact in state.artifacts)


def test_enzyme_mining_can_emit_notebook_ready_scaffold_without_auto_execution():
    state, result = _build_orchestrator().run(
        "挖一下和胆固醇代谢有关的候选酶，并生成一个可验证的 notebook 框架"
    )

    assert state.intent is not None
    assert state.intent.task_type == "enzyme_mining"
    assert result["notebook_ready"] is True
    assert "experiment_scaffold" in result
    scaffold = result["experiment_scaffold"]
    assert scaffold["handoff"]["auto_execute"] is False
    assert scaffold["handoff"]["mode"] == "scaffold_only"
    assert any(artifact.type == "experiment_scaffold" for artifact in state.artifacts)
