import os

import cellscientist.pipeline.report as report_mod
from cellscientist.pipeline.report import generate_report_from_orchestrator


def test_generate_report_from_orchestrator_creates_canonical_bundle(tmp_path, monkeypatch):
    dataset = "TEST_DS"
    task_id = "task-123"

    # Make project-root relative dirs used by report generator.
    results_root = tmp_path / "results" / dataset
    runs_root = tmp_path / "runs" / task_id / "modeling"
    logs_root = tmp_path / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    # Seed minimal input artifacts.
    (runs_root / "history_state.json").write_text("[]", encoding="utf-8")
    (logs_root / "experiment.log").write_text("[PipelineSummary] best=PCC=0.2\n", encoding="utf-8")
    (logs_root / "review.log").write_text("review tail\n", encoding="utf-8")

    result = {
        "task_id": task_id,
        "status": "terminated",
        "best_accuracy": 0.2,
        "total_iterations": 3,
        "experiment_success_count": 0,
        "iteration_history": [{"iteration": 1, "score": 0.2}],
        "metric": "PCC",
    }

    monkeypatch.setattr(report_mod, "project_root", lambda: str(tmp_path))
    out_path = generate_report_from_orchestrator(result, {}, dataset, str(logs_root))

    assert out_path
    assert os.path.exists(out_path)
    canonical_dir = os.path.dirname(out_path)
    assert os.path.exists(os.path.join(canonical_dir, "artifact_index.json"))
    assert os.path.exists(os.path.join(canonical_dir, "materials", "final_prompt_user.txt"))
    assert os.path.exists(os.path.join(logs_root, "pipeline_report.md"))
