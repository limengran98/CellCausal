from __future__ import annotations

from pathlib import Path

from evals.run_eval import CASE_FILES, load_case_suites, run_all_evals


def test_eval_runner_can_load_case_files_and_run_small_fallback_subset(tmp_path: Path):
    suites = load_case_suites()

    assert set(CASE_FILES) <= set(suites)
    assert len(suites["drug_analysis"]) >= 10
    assert len(suites["notebook_workflow"]) >= 8
    assert len(suites["fallback"]) >= 5

    summary, details, result_dir = run_all_evals(
        selected_suites=["fallback"],
        max_cases=2,
        output_root=tmp_path,
    )

    assert summary["total_cases"] == 2
    assert len(details) == 2
    assert result_dir.exists()
    assert (result_dir / "summary.json").exists()
    assert (result_dir / "details.jsonl").exists()
