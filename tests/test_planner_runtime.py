from __future__ import annotations

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.runtime.planner import build_intent
from cellscientist.skills.drug_info import DrugInfoSkill
from cellscientist.skills.legacy_notebook import LegacyNotebookSkill
from cellscientist.skills.notebook_workflow import NotebookWorkflowSkill


def _build_orchestrator() -> OrchestratorV2:
    registry = SkillRegistry()
    registry.register(DrugInfoSkill())
    registry.register(NotebookWorkflowSkill())
    registry.register(LegacyNotebookSkill())
    return OrchestratorV2(registry)


def test_planner_extracts_requested_actions_for_basic_notebook_queries():
    generate_intent = build_intent("帮我生成一个 notebook 实验设计")
    execute_intent = build_intent("执行这个 notebook")
    review_intent = build_intent("review 一下这个 notebook 的结构和科学性")
    autofix_intent = build_intent("这个 notebook 执行报错了，帮我 autofix")

    assert generate_intent.task_type == "legacy_notebook"
    assert generate_intent.requested_actions == ["generate"]
    assert execute_intent.requested_actions == ["execute"]
    assert review_intent.requested_actions == ["review"]
    assert autofix_intent.requested_actions == ["autofix"]


def test_planner_routes_drug_analysis_queries_without_collapsing_to_drug_info():
    metformin_intent = build_intent("分析一下 metformin 的靶点、机制和安全性")
    smiles_intent = build_intent("根据这个 SMILES 做药物分析: CN(C)C(=N)NC(=N)N")

    assert metformin_intent.task_type == "drug_analysis"
    assert smiles_intent.task_type == "drug_analysis"


def test_planner_parses_composite_notebook_followup_actions():
    intent = build_intent("生成的效果一般啊，挖掘生物知识重新审查，然后执行")

    assert intent.task_type == "legacy_notebook"
    assert intent.requested_actions == ["retrieval_refresh", "review", "execute"]


def test_planner_parses_retrieval_augmented_review_and_execute_followup():
    intent = build_intent("补充生物学证据重新 review，再执行")

    assert intent.task_type == "legacy_notebook"
    assert intent.requested_actions == ["retrieval_refresh", "review", "execute"]


def test_orchestrator_returns_clarification_fallback_for_unknown_queries():
    state, result = _build_orchestrator().run("这个系统是干什么的")

    assert state.intent is not None
    assert state.intent.task_type == "unknown"
    assert result["status"] == "needs_clarification"
    suggested_names = {item["name"] for item in result["suggested_skills"]}
    assert "drug-info" in suggested_names
    assert "notebook-workflow" in suggested_names


def test_notebook_workflow_executes_composite_actions_in_sequence():
    import cellscientist.skills.notebook_execute as notebook_execute_module
    import cellscientist.skills.notebook_retrieval_refresh as notebook_refresh_module
    import cellscientist.skills.notebook_review as notebook_review_module

    def _fake_refresh(
        _query,
        *,
        preferred_notebook_path=None,
        preferred_trial_dir=None,
        preferred_run_result=None,
        source_artifact_metadata=None,
    ):
        return {
            "action": "retrieval_refresh",
            "status": "retrieval_refreshed",
            "message": "ok",
            "target_notebook_path": preferred_notebook_path,
            "evidence_summary": "BioKB/PTGS1 evidence refreshed.",
            "evidence_count": 2,
            "legacy_entry": "legacy.retrieval",
            "details": {
                "target_trial_dir": preferred_trial_dir,
                "evidence_ids": ["B1", "L1"],
                "evidence_source_summary": "BioKB + literature refresh",
                "used_recent_run_result": bool(preferred_run_result is not None),
                "source_artifact_metadata": source_artifact_metadata or {},
            },
        }

    def _fake_review(
        _query,
        *,
        preferred_notebook_path=None,
        preferred_trial_dir=None,
        preferred_run_result=None,
        source_artifact_metadata=None,
        refreshed_evidence_context=None,
    ):
        return {
            "action": "review",
            "status": "review_summary_only",
            "message": "ok",
            "target_notebook_path": preferred_notebook_path,
            "review_report_path": "/tmp/review.md",
            "legacy_entry": "legacy.review",
            "details": {
                "target_trial_dir": preferred_trial_dir,
                "used_recent_run_result": bool(preferred_run_result is not None),
                "used_refreshed_evidence": bool(refreshed_evidence_context),
                "source_artifact_metadata": source_artifact_metadata or {},
            },
        }

    def _fake_execute(_query, *, preferred_notebook_path=None, preferred_trial_dir=None):
        return {
            "action": "execute",
            "status": "executed_via_legacy",
            "message": "ok",
            "notebook_path": preferred_notebook_path,
            "trial_dir": preferred_trial_dir,
            "error_log_path": None,
            "run_log_path": "/tmp/task_trace.json",
            "legacy_entry": "legacy.execute",
            "details": {"trial_dir": preferred_trial_dir},
        }

    original_refresh = notebook_refresh_module.bridge_refresh_notebook_retrieval
    original_review = notebook_review_module.bridge_review_notebook
    original_execute = notebook_execute_module.bridge_execute_notebook
    notebook_refresh_module.bridge_refresh_notebook_retrieval = _fake_refresh
    notebook_review_module.bridge_review_notebook = _fake_review
    notebook_execute_module.bridge_execute_notebook = _fake_execute
    try:
        state, result = _build_orchestrator().run("生成的效果一般啊，挖掘生物知识重新审查，然后执行")
    finally:
        notebook_refresh_module.bridge_refresh_notebook_retrieval = original_refresh
        notebook_review_module.bridge_review_notebook = original_review
        notebook_execute_module.bridge_execute_notebook = original_execute

    assert state.intent is not None
    assert state.intent.requested_actions == ["retrieval_refresh", "review", "execute"]
    assert result["action"] == "multi_step"
    assert result["requested_actions"] == ["retrieval_refresh", "review", "execute"]
    assert len(result["step_results"]) == 3
    assert result["step_results"][0]["action"] == "retrieval_refresh"
    assert result["step_results"][1]["action"] == "review"
    assert result["step_results"][1]["details"]["used_refreshed_evidence"] is True
    assert result["step_results"][2]["action"] == "execute"
    assert state.skill_trace == [
        "legacy_notebook:notebook-workflow",
        "legacy_notebook:notebook-retrieval-refresh",
        "legacy_notebook:notebook-review",
        "legacy_notebook:notebook-execute",
    ]
