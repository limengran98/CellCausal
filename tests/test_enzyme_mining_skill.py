from __future__ import annotations

from pathlib import Path

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.skills.enzyme_mining import EnzymeMiningSkill
from cellscientist.tools.enzyme_lookup import lookup_enzyme_candidates
from cellscientist.tools.enzyme_processing.candidate_sequence_mapping import (
    build_candidate_sequence_rows,
)
from cellscientist.tools.enzyme_ranking.catapro_bridge import build_catapro_ranking_bridge


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
    assert result["candidate_sequence_rows_status"]["status"] == "candidate_sequence_rows_ready"
    assert result["candidate_sequence_row_count"] > 0
    assert result["candidate_enzymes"]
    assert result["filtering_steps"]
    assert result["ranking_status"]
    assert result["ranking_ready"] is False
    assert result["ranking_model"]["name"] == "CataPro"
    assert isinstance(result["ranking_results"], list)
    assert result["why_not_runnable"]
    assert result["required_assets"]
    assert result["resolved_model_paths"]["catapro_root"]
    assert result["asset_check_details"]["prot_t5"]["path"]
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
    assert result["candidate_sequence_rows_status"]["status"] == "candidate_sequence_rows_ready"
    assert result["candidate_sequence_row_count"] > 0
    assert result["ranking_status"] in {
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
        "candidate_sequence_rows_csv",
        "candidate_sequence_mapping_status_json",
        "filtering_steps_json",
        "ranking_status_json",
        "ranking_run_details_json",
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
    assert "asset_check_details" in ranking_status_payload
    assert "resolved_model_paths" in ranking_status_payload
    artifact_types = {artifact.type for artifact in state.artifacts}
    assert "enzyme_candidate_table" in artifact_types
    assert "enzyme_candidate_sequence_rows" in artifact_types
    assert "enzyme_filtering_summary" in artifact_types
    assert "enzyme_ranking_result" in artifact_types


def test_catapro_bridge_reports_precise_missing_prott5_asset_details(monkeypatch):
    payload = lookup_enzyme_candidates("lipid_metabolism")
    _, candidate_sequence_rows = build_candidate_sequence_rows(
        candidate_enzymes=payload["candidate_enzymes"],
        query_focus=payload["query_focus"],
        zip_path=str(Path("references/enzyme_mining/output_sequences.zip")),
    )
    monkeypatch.setattr(
        "cellscientist.tools.enzyme_ranking.catapro_bridge._dependency_status",
        lambda: {"torch": True, "transformers": True, "rdkit": True, "pandas": True, "numpy": True},
    )

    bridge = build_catapro_ranking_bridge(
        query="根据这个 SMILES 挖可作用的候选酶并排序: C(C(C(=O)O)N)S",
        candidate_sequence_rows=candidate_sequence_rows,
        candidate_sequences_status={"status": "local_bundle_profiled_with_exact_dedupe"},
        candidate_sequence_rows_status={"status": "candidate_sequence_rows_ready"},
    )

    assert bridge["ranking_status"] == "blocked_incomplete_model_assets"
    assert bridge["input_not_ready"] == []
    assert "prot_t5_assets" in bridge["runtime_not_ready"]
    assert bridge["asset_check_details"]["prot_t5"]["dir_exists"] is True
    assert bridge["asset_check_details"]["prot_t5"]["weight_files_found"] == []
    assert bridge["asset_check_details"]["prot_t5"]["missing_weight_files"]
    assert any("ProtT5" in item for item in bridge["required_assets"])


def test_catapro_bridge_attempts_real_ranking_when_assets_are_resolved(monkeypatch, tmp_path):
    root = tmp_path / "catapro_root"
    (root / "inference").mkdir(parents=True)
    (root / "samples").mkdir(parents=True)
    (root / "models" / "kcat_models").mkdir(parents=True)
    (root / "models" / "Km_models").mkdir(parents=True)
    (root / "models" / "act_models").mkdir(parents=True)

    (root / "inference" / "predict.py").write_text("print('stub')\n", encoding="utf-8")
    (root / "samples" / "sample_inp.csv").write_text("row_id,Enzyme_id,type,sequence,smiles\n", encoding="utf-8")

    for index in range(10):
        for folder in ("kcat_models", "Km_models", "act_models"):
            (root / "models" / folder / f"{index}_bestmodel.pth").write_text("stub", encoding="utf-8")

    prot_t5_dir = tmp_path / "prot_t5_override"
    prot_t5_dir.mkdir(parents=True)
    for filename in ("config.json", "tokenizer_config.json", "spiece.model", "special_tokens_map.json", "pytorch_model.bin"):
        (prot_t5_dir / filename).write_text("stub", encoding="utf-8")

    molt5_dir = tmp_path / "molt5_override"
    molt5_dir.mkdir(parents=True)
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "spiece.model",
        "special_tokens_map.json",
        "pytorch_model.bin",
    ):
        (molt5_dir / filename).write_text("stub", encoding="utf-8")

    monkeypatch.setenv("CELLSCIENTIST_CATAPRO_ROOT", str(root))
    monkeypatch.setenv("CELLSCIENTIST_CATAPRO_PROTT5_DIR", str(prot_t5_dir))
    monkeypatch.setenv("CELLSCIENTIST_CATAPRO_MOLT5_DIR", str(molt5_dir))
    monkeypatch.setattr(
        "cellscientist.tools.enzyme_ranking.catapro_bridge._dependency_status",
        lambda: {"torch": True, "transformers": True, "rdkit": True, "pandas": True, "numpy": True},
    )

    captured: dict[str, object] = {}

    def fake_run(*, root, resolved_model_paths, input_rows):
        captured["root"] = str(root)
        captured["resolved_model_paths"] = dict(resolved_model_paths)
        captured["input_rows"] = list(input_rows)
        return {
            "status": "ranking_completed",
            "results": [
                {
                    "enzyme_id": "E1",
                    "smiles": "NC(CS)C(=O)O",
                    "pred_log10_kcat_s^-1": "1.2",
                    "pred_log10_km_mM": "-0.5",
                    "pred_log10_kcat_over_km": "1.7",
                    "rank": 1,
                }
            ],
            "run_dir": str(tmp_path / "run"),
            "input_csv": str(tmp_path / "run" / "input.csv"),
            "output_csv": str(tmp_path / "run" / "output.csv"),
            "command": ["python", "predict.py"],
            "cwd": str(root / "inference"),
            "used_device": "cpu",
            "model_dpath": str(root / "models"),
            "assembled_runtime_model_dir": str(tmp_path / "run" / "runtime_models"),
            "used_path_override_bundle": True,
            "stdout_summary": "",
            "stderr_summary": "",
            "returncode": 0,
        }

    monkeypatch.setattr(
        "cellscientist.tools.enzyme_ranking.catapro_bridge._run_catapro_prediction",
        fake_run,
    )

    bridge = build_catapro_ranking_bridge(
        query="根据这个 SMILES 挖可作用的候选酶并排序: C(C(C(=O)O)N)S",
        candidate_sequence_rows=[
            {
                "enzyme_id": "E1",
                "sequence": "MSTNPKPQRKTKRNTNRRPQDVKFPGG",
                "source": "test_bundle",
            }
        ],
        candidate_sequences_status={"status": "local_bundle_profiled_with_exact_dedupe"},
        candidate_sequence_rows_status={"status": "candidate_sequence_rows_ready"},
    )

    assert bridge["ranking_status"] == "ranking_completed"
    assert bridge["ranking_ready"] is True
    assert bridge["ranking_results"]
    assert bridge["resolved_model_paths"]["path_sources"]["prot_t5_dir"].startswith("env:")
    assert bridge["asset_check_details"]["prot_t5"]["ready"] is True
    assert captured["input_rows"]
