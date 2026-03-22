from __future__ import annotations

import csv
import importlib.util
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from ...core.bio_kb.smiles_resolver import canonicalize_smiles
from ...pipeline.utils import project_root

_SMILES_TOKEN_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{6,}")
_SUBSTRATE_KEYWORDS = (
    "substrate",
    "底物",
    "smiles",
    "底物信息",
    "given substrate",
    "给定底物",
)


def _catapro_root() -> Path:
    return Path(project_root()) / "references" / "enzyme_mining" / "CataPro-master" / "CataPro-master"


def _extract_smiles_candidate(text: str) -> Dict[str, Any]:
    candidates = sorted(set(_SMILES_TOKEN_RE.findall(text)), key=len, reverse=True)
    for candidate in candidates:
        if not any(ch in candidate for ch in "=#()[]\\/"):
            continue
        canon = canonicalize_smiles(candidate)
        if canon.get("canonical_smiles"):
            return {
                "raw_smiles": candidate,
                "canonical_smiles": str(canon["canonical_smiles"]),
                "inchikey": canon.get("inchikey"),
            }
    return {
        "raw_smiles": None,
        "canonical_smiles": None,
        "inchikey": None,
    }


def _extract_substrate_context(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    matched_keywords = [keyword for keyword in _SUBSTRATE_KEYWORDS if keyword in lowered]
    smiles_match = _extract_smiles_candidate(text)
    canonical_smiles = smiles_match.get("canonical_smiles")
    if canonical_smiles:
        status = "substrate_smiles_provided"
        mention_type = "smiles"
        notes = [
            "A substrate-like SMILES string was found and canonicalized for ranking preparation.",
        ]
    elif matched_keywords:
        status = "substrate_context_without_explicit_smiles"
        mention_type = "substrate_context"
        notes = [
            "The query requests substrate-aware ranking, but no canonicalizable SMILES string was found yet.",
        ]
    else:
        status = "no_substrate_context"
        mention_type = None
        notes = [
            "No explicit substrate context was found in the current query.",
        ]

    return {
        "status": status,
        "mention_type": mention_type,
        "matched_keywords": matched_keywords,
        "raw_smiles_candidate": smiles_match.get("raw_smiles"),
        "canonical_smiles": canonical_smiles,
        "inchikey": smiles_match.get("inchikey"),
        "notes": notes,
    }


def _dependency_status() -> Dict[str, bool]:
    modules = ["torch", "transformers", "rdkit", "pandas", "numpy"]
    return {name: bool(importlib.util.find_spec(name)) for name in modules}


def _asset_status(root: Path) -> Dict[str, Any]:
    predict_script = root / "inference" / "predict.py"
    run_script = root / "inference" / "run_catapro.sh"
    sample_input = root / "samples" / "sample_inp.csv"
    output_preview = root / "inference" / "catapro_test-pred.csv"
    models_dir = root / "models"
    prot_t5_dir = models_dir / "prot_t5_xl_uniref50"
    molt5_dir = models_dir / "molt5-base-smiles2caption"
    kcat_dir = models_dir / "kcat_models"
    km_dir = models_dir / "Km_models"
    act_dir = models_dir / "act_models"

    prot_t5_has_weights = any(p.name.startswith("pytorch_model") or p.name.endswith(".safetensors") for p in prot_t5_dir.glob("*"))
    molt5_has_weights = any(p.name.startswith("pytorch_model") or p.name.endswith(".safetensors") for p in molt5_dir.glob("*"))

    return {
        "root_path": str(root),
        "predict_script": str(predict_script),
        "run_script": str(run_script),
        "sample_input": str(sample_input),
        "output_preview": str(output_preview),
        "predict_script_exists": predict_script.is_file(),
        "sample_input_exists": sample_input.is_file(),
        "output_preview_exists": output_preview.is_file(),
        "kcat_fold_models": len(list(kcat_dir.glob("*_bestmodel.pth"))),
        "km_fold_models": len(list(km_dir.glob("*_bestmodel.pth"))),
        "act_fold_models": len(list(act_dir.glob("*_bestmodel.pth"))),
        "prot_t5_tokenizer_dir_exists": prot_t5_dir.is_dir(),
        "prot_t5_weight_files_available": prot_t5_has_weights,
        "molt5_dir_exists": molt5_dir.is_dir(),
        "molt5_weight_files_available": molt5_has_weights,
    }


def _build_input_rows(
    candidate_enzymes: Sequence[dict[str, Any]],
    substrate_smiles: str | None,
) -> list[dict[str, Any]]:
    if not substrate_smiles:
        return []
    rows: list[dict[str, Any]] = []
    for item in candidate_enzymes[:5]:
        enzyme_name = str(item.get("enzyme") or "").strip()
        sequence = str(item.get("sequence") or "").strip()
        if not enzyme_name or not sequence:
            continue
        rows.append(
            {
                "Enzyme_id": enzyme_name,
                "type": "wild",
                "sequence": sequence,
                "smiles": substrate_smiles,
            }
        )
    return rows


def _build_preview_rows(input_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for row in input_rows[:5]:
        sequence = str(row.get("sequence") or "")
        preview.append(
            {
                "Enzyme_id": row.get("Enzyme_id"),
                "type": row.get("type"),
                "sequence_preview": sequence[:30] + ("..." if len(sequence) > 30 else ""),
                "smiles": row.get("smiles"),
            }
        )
    return preview


def _missing_requirements(
    *,
    substrate_smiles: str | None,
    candidate_rows_ready: bool,
    dependency_status: Dict[str, bool],
    asset_status: Dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    if not substrate_smiles:
        missing.append("substrate_smiles")
    if not candidate_rows_ready:
        missing.append("candidate_sequence_mapping")
    for module_name, ok in dependency_status.items():
        if not ok:
            missing.append(f"python_module:{module_name}")
    if not asset_status["predict_script_exists"]:
        missing.append("catapro_predict_script")
    if not asset_status["sample_input_exists"]:
        missing.append("catapro_sample_input")
    if asset_status["kcat_fold_models"] < 10:
        missing.append("catapro_kcat_fold_models")
    if asset_status["km_fold_models"] < 10:
        missing.append("catapro_km_fold_models")
    if asset_status["act_fold_models"] < 10:
        missing.append("catapro_activity_fold_models")
    if not asset_status["molt5_weight_files_available"]:
        missing.append("molt5_weights")
    if not asset_status["prot_t5_weight_files_available"]:
        missing.append("prot_t5_weights")
    return missing


def _next_step_instructions(
    *,
    substrate_context: Dict[str, Any],
    candidate_rows_ready: bool,
    missing_requirements: Sequence[str],
) -> list[str]:
    instructions: list[str] = []
    if substrate_context.get("status") == "substrate_context_without_explicit_smiles":
        instructions.append("Provide a canonicalizable substrate SMILES string so the ranking bridge can prepare CataPro inputs.")
    elif substrate_context.get("status") == "no_substrate_context":
        instructions.append("Add a substrate or SMILES string if you want sequence-level candidate ranking instead of only candidate mining.")
    if not candidate_rows_ready:
        instructions.append("Attach enzyme sequences to the shortlisted candidates so the ranking bridge can write CataPro input rows.")
    if "prot_t5_weights" in missing_requirements:
        instructions.append("Restore the missing ProtT5 weight files under the local CataPro model directory before attempting real ranking.")
    if any(item.startswith("python_module:") for item in missing_requirements):
        instructions.append("Install the missing Python dependencies in the current runtime environment before attempting CataPro inference.")
    if not instructions:
        instructions.append("Ranking inputs are ready; the bridge can attempt local CataPro inference without triggering notebook execution.")
    return instructions[:5]


def _run_catapro_prediction(
    *,
    root: Path,
    input_rows: Sequence[dict[str, Any]],
) -> Dict[str, Any]:
    run_root = Path(project_root()) / "results" / "enzyme_mining" / "catapro_runs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / f"catapro_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_csv = run_dir / "catapro_input.csv"
    output_csv = run_dir / "catapro_output.csv"

    with open(input_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "Enzyme_id", "type", "sequence", "smiles"])
        for index, row in enumerate(input_rows):
            writer.writerow([index, row["Enzyme_id"], row["type"], row["sequence"], row["smiles"]])

    cmd = [
        sys.executable,
        "predict.py",
        "-inp_fpath",
        str(input_csv),
        "-model_dpath",
        str(root / "models"),
        "-batch_size",
        "64",
        "-device",
        "cpu",
        "-out_fpath",
        str(output_csv),
    ]
    proc = subprocess.run(
        cmd,
        cwd=root / "inference",
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )

    if proc.returncode != 0 or not output_csv.exists():
        return {
            "status": "ranking_run_failed",
            "results": [],
            "run_dir": str(run_dir),
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "stderr_summary": (proc.stderr or proc.stdout or "").strip()[:800],
        }

    rows: list[dict[str, Any]] = []
    with open(output_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                {
                    "enzyme_id": raw.get("fasta_id"),
                    "smiles": raw.get("smiles"),
                    "pred_log10_kcat_s^-1": raw.get("pred_log10[kcat(s^-1)]"),
                    "pred_log10_km_mM": raw.get("pred_log10[Km(mM)]"),
                    "pred_log10_kcat_over_km": raw.get("pred_log10[kcat/Km(s^-1mM^-1)]"),
                }
            )
    rows.sort(key=lambda item: float(item.get("pred_log10_kcat_over_km") or "-inf"), reverse=True)
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank

    return {
        "status": "ranking_completed",
        "results": rows,
        "run_dir": str(run_dir),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "stderr_summary": None,
    }


def build_catapro_ranking_bridge(
    *,
    query: str,
    candidate_enzymes: Sequence[dict[str, Any]],
    candidate_sequences_status: Dict[str, Any],
) -> Dict[str, Any]:
    root = _catapro_root()
    dependency_status = _dependency_status()
    asset_status = _asset_status(root)
    substrate_context = _extract_substrate_context(query)
    substrate_smiles = substrate_context.get("canonical_smiles")
    input_rows = _build_input_rows(candidate_enzymes, substrate_smiles)
    preview_rows = _build_preview_rows(input_rows)

    has_dependencies = all(dependency_status.values())
    has_fold_models = (
        asset_status["kcat_fold_models"] >= 10
        and asset_status["km_fold_models"] >= 10
        and asset_status["act_fold_models"] >= 10
    )
    has_model_assets = (
        asset_status["predict_script_exists"]
        and asset_status["sample_input_exists"]
        and has_fold_models
        and asset_status["molt5_weight_files_available"]
    )
    prot_t5_ready = asset_status["prot_t5_weight_files_available"]
    candidate_rows_ready = bool(input_rows)
    missing_requirements = _missing_requirements(
        substrate_smiles=substrate_smiles,
        candidate_rows_ready=candidate_rows_ready,
        dependency_status=dependency_status,
        asset_status=asset_status,
    )
    next_step_instructions = _next_step_instructions(
        substrate_context=substrate_context,
        candidate_rows_ready=candidate_rows_ready,
        missing_requirements=missing_requirements,
    )
    ranking_ready = bool(
        substrate_smiles
        and candidate_rows_ready
        and has_dependencies
        and has_model_assets
        and prot_t5_ready
    )

    if not substrate_smiles:
        ranking_status = "awaiting_substrate_smiles"
    elif not candidate_rows_ready:
        ranking_status = "awaiting_candidate_sequence_mapping"
    elif not has_dependencies:
        ranking_status = "blocked_missing_runtime_dependencies"
    elif not has_model_assets or not prot_t5_ready:
        ranking_status = "blocked_incomplete_model_assets"
    else:
        ranking_status = "ranking_input_ready"

    ranking_results: list[dict[str, Any]] = []
    ranking_run_details: Dict[str, Any] = {}
    if ranking_ready:
        run_payload = _run_catapro_prediction(root=root, input_rows=input_rows)
        ranking_status = str(run_payload.get("status") or ranking_status)
        ranking_results = list(run_payload.get("results") or [])
        ranking_run_details = {
            "run_dir": run_payload.get("run_dir"),
            "input_csv": run_payload.get("input_csv"),
            "output_csv": run_payload.get("output_csv"),
            "stderr_summary": run_payload.get("stderr_summary"),
        }

    why_not_runnable = [] if ranking_ready else [
        instruction for instruction in next_step_instructions
    ]
    prepared_input_preview = {
        "status": "preview_ready" if preview_rows else "preview_blocked",
        "substrate_smiles": substrate_smiles,
        "preview_rows": preview_rows,
        "candidate_row_count": len(input_rows),
        "notes": [
            "CataPro expects Enzyme_id/type/sequence/smiles rows.",
            "This bridge builds ranking inputs first and only attempts real inference when runtime dependencies, model assets, and sequence rows are all available.",
        ],
    }

    return {
        "substrate_context": substrate_context,
        "substrate_smiles": substrate_smiles,
        "ranking_status": ranking_status,
        "ranking_ready": ranking_ready,
        "ranking_model": {
            "name": "CataPro",
            "bridge_type": "local_inference_bridge",
            "entrypoint": asset_status["predict_script"],
            "whether_model_assets_available": has_model_assets and prot_t5_ready,
            "whether_python_deps_available": has_dependencies,
            "dependency_status": dependency_status,
            "asset_status": asset_status,
            "required_input_columns": ["Enzyme_id", "type", "sequence", "smiles"],
            "candidate_sequence_bundle_status": candidate_sequences_status.get("status"),
        },
        "ranking_results": ranking_results,
        "why_not_runnable": why_not_runnable,
        "required_assets": missing_requirements,
        "prepared_input_preview": prepared_input_preview,
        "ranking_input_preview": prepared_input_preview,
        "next_step_instructions": next_step_instructions,
        "ranking_run_details": ranking_run_details,
    }
