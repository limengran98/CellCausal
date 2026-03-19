import os
import sys
import tempfile
import types

from cellscientist.legacy.llm_resilience import resolve_bridge_llm_providers
from cellscientist.legacy.notebook_bridge import _create_degraded_notebook
from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.notebook_models import NotebookArtifact
from cellscientist.runtime.session import create_session
from cellscientist.runtime.state import ResearchIntent
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.skills.notebook_autofix import NotebookAutofixSkill
from cellscientist.skills.notebook_execute import NotebookExecuteSkill
from cellscientist.skills.legacy_notebook import LegacyNotebookSkill
from cellscientist.skills.notebook_review import NotebookReviewSkill
from cellscientist.skills.notebook_workflow import NotebookWorkflowSkill


def _build_orchestrator() -> OrchestratorV2:
    return OrchestratorV2(
        SkillRegistry(
            [
                NotebookWorkflowSkill(),
                LegacyNotebookSkill(),
            ]
        )
    )


def test_notebook_workflow_routes_generate_requests_and_emits_notebook_artifact():
    import cellscientist.skills.notebook_generate as notebook_generate_module

    def _fake_generate(_query):
        return {
            "action": "generate",
            "status": "generated_via_legacy",
            "message": "ok",
            "notebook_path": "/tmp/example_notebook.ipynb",
            "legacy_entry": "legacy.generate",
            "details": {
                "trial_dir": "/tmp/example_trial",
                "final_provider_used": {
                    "provider_name": "primary",
                    "source": "pipeline_config.llm",
                },
            },
        }

    original = notebook_generate_module.bridge_generate_notebook
    notebook_generate_module.bridge_generate_notebook = _fake_generate
    try:
        state, result = _build_orchestrator().run("帮我生成一个 notebook 实验设计")
    finally:
        notebook_generate_module.bridge_generate_notebook = original

    assert state.intent is not None
    assert state.intent.task_type == "legacy_notebook"
    assert state.skill_trace == [
        "legacy_notebook:notebook-workflow",
        "legacy_notebook:notebook-generate",
    ]
    assert result["action"] == "generate"
    assert result["status"] == "generated_via_legacy"
    assert result["legacy_entry"] == "legacy.generate"
    assert any(artifact.type == "notebook" for artifact in state.artifacts)
    notebook_artifact = next(artifact for artifact in state.artifacts if artifact.type == "notebook")
    assert notebook_artifact.metadata["path"] == "/tmp/example_notebook.ipynb"
    assert notebook_artifact.metadata["trial_dir"] == "/tmp/example_trial"
    assert notebook_artifact.metadata["source"] == "legacy"
    assert notebook_artifact.metadata["provider_name"] == "primary"
    assert state.last_notebook_artifact is not None
    assert state.last_notebook_artifact.path == "/tmp/example_notebook.ipynb"


def test_notebook_workflow_routes_execute_requests():
    import cellscientist.skills.notebook_execute as notebook_execute_module

    def _fake_execute(_query, *, preferred_notebook_path=None, preferred_trial_dir=None):
        return {
            "action": "execute",
            "status": "executed_via_legacy",
            "message": "ok",
            "query": _query,
            "notebook_path": preferred_notebook_path,
            "trial_dir": preferred_trial_dir,
            "error_log_path": "/tmp/error.txt",
            "run_log_path": "/tmp/task_trace.json",
            "legacy_entry": "legacy.execute",
            "details": {"trial_dir": "/tmp/example_trial"},
        }

    original = notebook_execute_module.bridge_execute_notebook
    notebook_execute_module.bridge_execute_notebook = _fake_execute
    try:
        state, result = _build_orchestrator().run("执行这个 notebook")
    finally:
        notebook_execute_module.bridge_execute_notebook = original

    assert state.intent is not None
    assert state.intent.task_type == "legacy_notebook"
    assert state.skill_trace == [
        "legacy_notebook:notebook-workflow",
        "legacy_notebook:notebook-execute",
    ]
    assert result["action"] == "execute"
    assert result["status"] == "executed_via_legacy"
    assert result["run_log_path"] == "/tmp/task_trace.json"
    run_artifact = next(artifact for artifact in state.artifacts if artifact.type == "notebook_run")
    assert run_artifact.metadata["error_log_path"] == "/tmp/error.txt"


def test_notebook_workflow_prefers_latest_notebook_artifact_for_execute_requests():
    import cellscientist.skills.notebook_execute as notebook_execute_module

    captured = {}

    def _fake_execute(_query, *, preferred_notebook_path=None, preferred_trial_dir=None):
        captured["preferred_notebook_path"] = preferred_notebook_path
        captured["preferred_trial_dir"] = preferred_trial_dir
        return {
            "action": "execute",
            "status": "legacy_execute_missing_notebook",
            "message": "ok",
            "query": _query,
            "notebook_path": preferred_notebook_path,
            "trial_dir": preferred_trial_dir,
            "error_log_path": None,
            "run_log_path": None,
            "legacy_entry": "legacy.execute",
            "details": {},
        }

    original = notebook_execute_module.bridge_execute_notebook
    notebook_execute_module.bridge_execute_notebook = _fake_execute
    try:
        state = create_session("执行这个 notebook")
        state.intent = ResearchIntent(raw_query=state.user_query, task_type="legacy_notebook")
        state.last_notebook_artifact = NotebookArtifact(
            name="draft_notebook.ipynb",
            path="/tmp/draft_notebook.ipynb",
            trial_dir="/tmp/example_trial",
            source="legacy",
            metadata={},
        )
        result = NotebookWorkflowSkill().run(state)
    finally:
        notebook_execute_module.bridge_execute_notebook = original

    assert result["action"] == "execute"
    assert captured["preferred_notebook_path"] == "/tmp/draft_notebook.ipynb"
    assert captured["preferred_trial_dir"] == "/tmp/example_trial"


def test_notebook_workflow_routes_review_requests_and_emits_review_artifact():
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
            "query": _query,
            "target_notebook_path": preferred_notebook_path or "/tmp/review_target.ipynb",
            "review_report_path": "/tmp/notebook_review_summary.md",
            "legacy_entry": "legacy.review",
            "details": {
                "target_trial_dir": preferred_trial_dir or "/tmp/example_trial",
                "used_recent_run_result": bool(preferred_run_result is not None),
                "source_artifact_metadata": source_artifact_metadata or {},
            },
        }

    original = notebook_review_module.bridge_review_notebook
    notebook_review_module.bridge_review_notebook = _fake_review
    try:
        state = create_session("review 一下这个 notebook 的结构和科学性")
        state.intent = ResearchIntent(raw_query=state.user_query, task_type="legacy_notebook")
        state.last_notebook_artifact = NotebookArtifact(
            name="draft_notebook.ipynb",
            path="/tmp/draft_notebook.ipynb",
            trial_dir="/tmp/example_trial",
            source="legacy",
            metadata={"origin": "test"},
        )
        result = NotebookWorkflowSkill().run(state)
    finally:
        notebook_review_module.bridge_review_notebook = original

    assert result["action"] == "review"
    assert state.skill_trace == [
        "legacy_notebook:notebook-workflow",
        "legacy_notebook:notebook-review",
    ]
    review_artifact = next(artifact for artifact in state.artifacts if artifact.type == "review_report")
    assert review_artifact.metadata["report_path"] == "/tmp/notebook_review_summary.md"
    assert review_artifact.metadata["target_notebook_path"] == "/tmp/draft_notebook.ipynb"


def test_notebook_workflow_routes_autofix_requests_using_recent_failed_run_context():
    import cellscientist.skills.notebook_autofix as notebook_autofix_module
    from cellscientist.runtime.notebook_models import NotebookRunResult

    captured = {}

    def _fake_autofix(
        _query,
        *,
        preferred_notebook_path=None,
        preferred_trial_dir=None,
        preferred_run_result=None,
        source_artifact_metadata=None,
    ):
        captured["preferred_notebook_path"] = preferred_notebook_path
        captured["preferred_trial_dir"] = preferred_trial_dir
        captured["preferred_run_result"] = preferred_run_result
        captured["source_artifact_metadata"] = source_artifact_metadata
        return {
            "action": "autofix",
            "status": "autofix_no_change",
            "message": "ok",
            "query": _query,
            "target_notebook_path": preferred_notebook_path,
            "error_log_path": "/tmp/error_log_round_1.txt",
            "patched_notebook_path": None,
            "legacy_entry": "legacy.autofix",
            "details": {
                "target_trial_dir": preferred_trial_dir,
                "used_recent_run_result": bool(preferred_run_result is not None),
            },
        }

    original = notebook_autofix_module.bridge_autofix_notebook
    notebook_autofix_module.bridge_autofix_notebook = _fake_autofix
    try:
        state = create_session("这个 notebook 执行报错了，帮我 autofix")
        state.intent = ResearchIntent(raw_query=state.user_query, task_type="legacy_notebook")
        state.last_notebook_artifact = NotebookArtifact(
            name="draft_notebook.ipynb",
            path="/tmp/draft_notebook.ipynb",
            trial_dir="/tmp/example_trial",
            source="legacy",
            metadata={"origin": "test"},
        )
        state.last_notebook_run_result = NotebookRunResult(
            notebook_path="/tmp/draft_notebook_exec.ipynb",
            trial_dir="/tmp/example_trial",
            status="legacy_execution_failed",
            error_log_path="/tmp/error_log_round_1.txt",
            run_log_path="/tmp/task_trace.json",
            metadata={"origin": "test"},
        )
        result = NotebookWorkflowSkill().run(state)
    finally:
        notebook_autofix_module.bridge_autofix_notebook = original

    assert result["action"] == "autofix"
    assert state.skill_trace == [
        "legacy_notebook:notebook-workflow",
        "legacy_notebook:notebook-autofix",
    ]
    assert captured["preferred_notebook_path"] == "/tmp/draft_notebook.ipynb"
    assert captured["preferred_trial_dir"] == "/tmp/example_trial"
    assert captured["preferred_run_result"] is not None


def test_bridge_llm_resolution_normalizes_primary_and_adds_compat_fallback():
    original_primary_key = os.environ.get("PRIMARY_TEST_KEY")
    original_fallback_key = os.environ.get("FALLBACK_TEST_KEY")
    os.environ["PRIMARY_TEST_KEY"] = "sk-test-primary"
    os.environ["FALLBACK_TEST_KEY"] = "sk-test-fallback"
    cfg = {
        "llm": {
            "model": "gpt-5.4",
            "base_url": "https://vip.yi-zhan.top",
            "api_key_env": "PRIMARY_TEST_KEY",
        },
        "llm_fallbacks": [
            {
                "name": "fallback_1",
                "model": "gpt-5.4-mini",
                "base_url": "https://fallback.example",
                "api_key_env": "FALLBACK_TEST_KEY",
            }
        ],
        "llm_report": {
            "model": "gpt-5.4-mini",
            "base_url": "https://sub.jia4u.de/v11",
            "api_key": "sk-test-compat",
        },
    }

    try:
        provider_plan = resolve_bridge_llm_providers(cfg)
    finally:
        if original_primary_key is None:
            os.environ.pop("PRIMARY_TEST_KEY", None)
        else:
            os.environ["PRIMARY_TEST_KEY"] = original_primary_key
        if original_fallback_key is None:
            os.environ.pop("FALLBACK_TEST_KEY", None)
        else:
            os.environ["FALLBACK_TEST_KEY"] = original_fallback_key

    assert provider_plan["llm_resolution"]["primary_provider"]["normalized_base_url"] == "https://vip.yi-zhan.top/v1"
    assert provider_plan["llm_resolution"]["primary_provider"]["api_key_source"] == "env:PRIMARY_TEST_KEY"
    assert provider_plan["llm_resolution"]["fallback_provider_count"] == 2
    assert len(provider_plan["providers"]) == 3
    assert provider_plan["llm_resolution"]["fallback_providers"][0]["provider_name"] == "fallback_1"
    assert provider_plan["llm_resolution"]["fallback_providers"][0]["config_source"] == "pipeline_config.llm_fallbacks"
    assert provider_plan["llm_resolution"]["fallback_providers"][0]["normalized_base_url"] == "https://fallback.example/v1"
    assert provider_plan["llm_resolution"]["fallback_providers"][1]["config_source"] == "pipeline_config.llm_report_compat"
    assert any(
        "nonstandard version path" in warning
        for warning in provider_plan["llm_resolution"]["fallback_providers"][1]["warnings"]
    )


def test_bridge_generate_returns_generated_via_fallback_when_secondary_provider_succeeds():
    import cellscientist.legacy.notebook_bridge as bridge_module

    original_load = bridge_module._load_legacy_experiment_config
    original_execution_module = sys.modules.get("cellscientist.core.execution_workflow")
    original_orchestrator_module = sys.modules.get("cellscientist.core.prompt_orchestrator")

    with tempfile.TemporaryDirectory() as tmpdir:
        def _fake_load():
            return (
                {
                    "dataset_name": "BBBC036",
                    "paths": {
                        "design_execution_root": tmpdir,
                        "data_root": tmpdir,
                        "data_h5_filename": "missing.h5",
                    },
                    "prompt_branch": {
                        "save_root": tmpdir,
                        "prompt_file": "prompts/pipeline_prompt.yaml",
                    },
                    "llm": {
                        "model": "gpt-5.4",
                        "base_url": "https://primary.example/v1",
                        "api_key": "sk-primary",
                    },
                    "llm_fallbacks": [
                        {
                            "name": "fallback_1",
                            "model": "gpt-5.4-mini",
                            "base_url": "https://fallback.example/v1",
                            "api_key": "sk-fallback",
                        }
                    ],
                    "prompts": {},
                },
                {
                    "pipeline_config": "/tmp/pipeline_config.json",
                    "experiment_config": "/tmp/experiment_config.json",
                },
            )

        def _fake_phase_generate(cfg, spec_path, run_name=None):
            trial_dir = os.path.join(tmpdir, run_name or "trial")
            os.makedirs(trial_dir, exist_ok=True)
            if cfg["llm"]["base_url"] == "https://primary.example/v1":
                raise RuntimeError("HTTP 502 from primary")
            notebook_path = os.path.join(trial_dir, "notebook_prompt.ipynb")
            with open(notebook_path, "w", encoding="utf-8") as handle:
                handle.write("{}")
            return {"trial_dir": trial_dir, "notebook_path": notebook_path}

        fake_execution_module = types.ModuleType("cellscientist.core.execution_workflow")
        fake_execution_module._inject_api_key = lambda _cfg: None
        fake_execution_module._setup_stage1_resources = lambda _cfg, _enable_idea, spec_path=None: None

        fake_orchestrator_module = types.ModuleType("cellscientist.core.prompt_orchestrator")
        fake_orchestrator_module.phase_generate = _fake_phase_generate

        bridge_module._load_legacy_experiment_config = _fake_load
        sys.modules["cellscientist.core.execution_workflow"] = fake_execution_module
        sys.modules["cellscientist.core.prompt_orchestrator"] = fake_orchestrator_module
        try:
            result = bridge_module.bridge_generate_notebook("帮我基于BBBC036生成一个实验设计")
        finally:
            bridge_module._load_legacy_experiment_config = original_load
            if original_execution_module is None:
                sys.modules.pop("cellscientist.core.execution_workflow", None)
            else:
                sys.modules["cellscientist.core.execution_workflow"] = original_execution_module
            if original_orchestrator_module is None:
                sys.modules.pop("cellscientist.core.prompt_orchestrator", None)
            else:
                sys.modules["cellscientist.core.prompt_orchestrator"] = original_orchestrator_module

    assert result["status"] == "generated_via_fallback"
    assert result["notebook_path"] is not None
    assert result["details"]["final_provider_used"]["provider_name"] == "fallback_1"
    assert len(result["details"]["llm_attempts"]) == 2
    assert result["details"]["llm_attempts"][0]["status"] == "provider_retryable_error"
    assert result["details"]["llm_attempts"][1]["status"] == "success"


def test_degraded_notebook_helper_writes_stub_and_trace():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _create_degraded_notebook(
            cfg={
                "dataset_name": "BBBC036",
                "paths": {"data_root": tmpdir, "data_h5_filename": "missing.h5"},
            },
            query="帮我基于BBBC036生成一个实验设计",
            prompt_file="/tmp/pipeline_prompt.yaml",
            trial_dir=tmpdir,
            llm_resolution={"primary": {"model": "gpt-5.4"}},
            llm_attempts=[
                {
                    "provider": "primary",
                    "model": "gpt-5.4",
                    "base_url": "https://vip.yi-zhan.top/v1",
                    "status": "provider_retryable_error",
                    "error_summary": "HTTP 404",
                }
            ],
            degraded_reason="All providers failed",
        )

        assert result["notebook_path"] is not None
        assert result["run_log_path"] is not None
        assert os.path.exists(result["notebook_path"])
        assert os.path.exists(result["run_log_path"])
        assert os.path.exists(os.path.join(tmpdir, "final_keep", "notebook_prompt_degraded.ipynb"))
