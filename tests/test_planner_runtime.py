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


def test_planner_parses_composite_notebook_followup_actions():
    intent = build_intent("生成的效果一般啊，挖掘生物知识重新审查，然后执行")

    assert intent.task_type == "legacy_notebook"
    assert intent.requested_actions == ["review", "execute"]


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
    import cellscientist.skills.notebook_review as notebook_review_module

    def _fake_review(
        _query,
        *,
        preferred_notebook_path=None,
        preferred_trial_dir=None,
        preferred_run_result=None,
        source_artifact_metadata=None,
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

    original_review = notebook_review_module.bridge_review_notebook
    original_execute = notebook_execute_module.bridge_execute_notebook
    notebook_review_module.bridge_review_notebook = _fake_review
    notebook_execute_module.bridge_execute_notebook = _fake_execute
    try:
        state, result = _build_orchestrator().run("生成的效果一般啊，挖掘生物知识重新审查，然后执行")
    finally:
        notebook_review_module.bridge_review_notebook = original_review
        notebook_execute_module.bridge_execute_notebook = original_execute

    assert state.intent is not None
    assert state.intent.requested_actions == ["review", "execute"]
    assert result["action"] == "multi_step"
    assert result["requested_actions"] == ["review", "execute"]
    assert len(result["step_results"]) == 2
    assert result["step_results"][0]["action"] == "review"
    assert result["step_results"][1]["action"] == "execute"
    assert state.skill_trace == [
        "legacy_notebook:notebook-workflow",
        "legacy_notebook:notebook-review",
        "legacy_notebook:notebook-execute",
    ]
