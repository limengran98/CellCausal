from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..pipeline.utils import project_root


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(project_root()).resolve()))
    except Exception:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _candidate_rows(candidate_enzymes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidate_enzymes:
        rows.append(
            {
                "enzyme_id": item.get("id") or item.get("enzyme_id") or item.get("enzyme"),
                "enzyme": item.get("enzyme"),
                "role": item.get("role"),
                "confidence": item.get("confidence"),
                "source": item.get("source") or "seeded_candidate_panel",
                "rationale": item.get("rationale") or item.get("role"),
            }
        )
    return rows


def _ranking_preview_rows(preview_payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(preview_payload, Mapping):
        return []
    rows = preview_payload.get("preview_rows")
    if not isinstance(rows, Sequence):
        return []
    return [dict(item) for item in rows if isinstance(item, Mapping)]


def _ranking_result_rows(ranking_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in ranking_results if isinstance(item, Mapping)]


def export_enzyme_mining_artifacts(
    *,
    session_id: str,
    focus_key: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = (
        Path(project_root())
        / "results"
        / "enzyme_mining"
        / f"{timestamp}_{focus_key}_{session_id[:8]}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    candidate_sources_path = result_dir / "candidate_sources.json"
    candidate_sequences_status_path = result_dir / "candidate_sequences_status.json"
    candidate_table_path = result_dir / "candidate_table.csv"
    filtering_steps_path = result_dir / "filtering_steps.json"
    ranking_status_path = result_dir / "ranking_status.json"
    ranking_input_preview_path = result_dir / "ranking_input_preview.csv"
    ranking_results_path = result_dir / "ranking_results.csv"
    enzyme_result_path = result_dir / "enzyme_mining_result.json"
    experiment_scaffold_path = result_dir / "experiment_scaffold.json"

    _write_json(candidate_sources_path, result.get("candidate_sources") or [])
    _write_json(candidate_sequences_status_path, result.get("candidate_sequences_status") or {})
    _write_json(filtering_steps_path, result.get("filtering_steps") or [])

    candidate_rows = _candidate_rows(result.get("candidate_enzymes") or [])
    _write_csv(
        candidate_table_path,
        candidate_rows,
        ["enzyme_id", "enzyme", "role", "confidence", "source", "rationale"],
    )

    preview_rows = _ranking_preview_rows(result.get("prepared_input_preview"))
    _write_csv(
        ranking_input_preview_path,
        preview_rows,
        ["Enzyme_id", "type", "sequence_preview", "smiles"],
    )

    ranking_rows = _ranking_result_rows(result.get("ranking_results") or [])
    _write_csv(
        ranking_results_path,
        ranking_rows,
        [
            "rank",
            "enzyme_id",
            "smiles",
            "pred_log10_kcat_s^-1",
            "pred_log10_km_mM",
            "pred_log10_kcat_over_km",
        ],
    )

    ranking_status_payload = {
        "current_status": result.get("ranking_status"),
        "ranking_model": result.get("ranking_model"),
        "whether_real_ranking_completed": str(result.get("ranking_status") or "") == "ranking_completed",
        "ranking_ready": result.get("ranking_ready"),
        "substrate_context": result.get("substrate_context"),
        "substrate_smiles": result.get("substrate_smiles"),
        "why_not_runnable": result.get("why_not_runnable") or [],
        "required_assets": result.get("required_assets") or [],
        "prepared_input_preview": result.get("prepared_input_preview") or {},
        "next_step_instructions": result.get("next_step_instructions") or [],
        "ranking_run_details": result.get("ranking_run_details") or {},
    }
    _write_json(ranking_status_path, ranking_status_payload)

    scaffold_written = False
    if result.get("notebook_ready") and isinstance(result.get("experiment_scaffold"), Mapping):
        _write_json(experiment_scaffold_path, result.get("experiment_scaffold"))
        scaffold_written = True

    files = {
        "candidate_sources_json": _relative_path(candidate_sources_path),
        "candidate_sequences_status_json": _relative_path(candidate_sequences_status_path),
        "candidate_table_csv": _relative_path(candidate_table_path),
        "filtering_steps_json": _relative_path(filtering_steps_path),
        "ranking_status_json": _relative_path(ranking_status_path),
        "ranking_input_preview_csv": _relative_path(ranking_input_preview_path),
        "ranking_results_csv": _relative_path(ranking_results_path),
        "enzyme_mining_result_json": _relative_path(enzyme_result_path),
    }
    if scaffold_written:
        files["experiment_scaffold_json"] = _relative_path(experiment_scaffold_path)

    result_payload = dict(result)
    result_payload["artifact_export"] = {
        "result_dir": _relative_path(result_dir),
        "files": files,
    }
    _write_json(enzyme_result_path, result_payload)

    return {
        "result_dir": _relative_path(result_dir),
        "files": files,
    }
