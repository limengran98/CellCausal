from __future__ import annotations

import csv
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
_WEIGHT_PATTERNS = (
    "pytorch_model.bin",
    "pytorch_model*.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model*.safetensors",
    "model.safetensors.index.json",
    "*.safetensors",
)
_PROT_T5_REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "spiece.model",
    "special_tokens_map.json",
)
_MOLT5_REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "spiece.model",
    "special_tokens_map.json",
)
_PATH_OVERRIDE_ENV = {
    "catapro_root": ("CELLSCIENTIST_CATAPRO_ROOT", "CATAPRO_ROOT"),
    "model_dir": ("CELLSCIENTIST_CATAPRO_MODEL_DIR", "CATAPRO_MODEL_DIR"),
    "prot_t5_dir": ("CELLSCIENTIST_CATAPRO_PROTT5_DIR", "CATAPRO_PROTT5_DIR"),
    "molt5_dir": ("CELLSCIENTIST_CATAPRO_MOLT5_DIR", "CATAPRO_MOLT5_DIR"),
    "device": ("CELLSCIENTIST_CATAPRO_DEVICE", "CATAPRO_DEVICE"),
}


def _default_catapro_root() -> Path:
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


def _first_env(names: Sequence[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


def _resolve_model_paths(path_overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    overrides = path_overrides or {}
    default_root = _default_catapro_root()
    resolved: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    def resolve_path(
        key: str,
        *,
        default: Path,
        relative_to: Path | None = None,
    ) -> Path:
        override_value = overrides.get(key)
        if override_value:
            sources[key] = "bridge_override"
            return Path(str(override_value)).expanduser()
        env_value, env_name = _first_env(_PATH_OVERRIDE_ENV[key])
        if env_value:
            sources[key] = f"env:{env_name}"
            return Path(env_value).expanduser()
        sources[key] = "default_repo_reference"
        if relative_to is not None:
            return relative_to / default
        return default

    catapro_root = resolve_path("catapro_root", default=default_root)
    model_dir = resolve_path("model_dir", default=Path("models"), relative_to=catapro_root)
    prot_t5_dir = resolve_path("prot_t5_dir", default=Path("prot_t5_xl_uniref50"), relative_to=model_dir)
    molt5_dir = resolve_path("molt5_dir", default=Path("molt5-base-smiles2caption"), relative_to=model_dir)

    device_override = overrides.get("device")
    if device_override:
        device = str(device_override)
        sources["device"] = "bridge_override"
    else:
        env_value, env_name = _first_env(_PATH_OVERRIDE_ENV["device"])
        if env_value:
            device = env_value
            sources["device"] = f"env:{env_name}"
        else:
            device = "cpu"
            sources["device"] = "default_cpu"

    resolved["catapro_root"] = str(catapro_root)
    resolved["model_dir"] = str(model_dir)
    resolved["prot_t5_dir"] = str(prot_t5_dir)
    resolved["molt5_dir"] = str(molt5_dir)
    resolved["device"] = device
    resolved["path_sources"] = sources
    resolved["search_order"] = {
        "catapro_root": ["bridge_override", *list(_PATH_OVERRIDE_ENV["catapro_root"]), "default_repo_reference"],
        "model_dir": ["bridge_override", *list(_PATH_OVERRIDE_ENV["model_dir"]), "default_repo_reference"],
        "prot_t5_dir": ["bridge_override", *list(_PATH_OVERRIDE_ENV["prot_t5_dir"]), "default_repo_reference"],
        "molt5_dir": ["bridge_override", *list(_PATH_OVERRIDE_ENV["molt5_dir"]), "default_repo_reference"],
        "device": ["bridge_override", *list(_PATH_OVERRIDE_ENV["device"]), "default_cpu"],
    }
    return resolved


def _collect_weight_files(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    seen: set[str] = set()
    matches: list[str] = []
    for pattern in _WEIGHT_PATTERNS:
        for path in directory.glob(pattern):
            if path.is_file() and path.name not in seen:
                seen.add(path.name)
                matches.append(path.name)
    return sorted(matches)


def _hf_model_asset_check(directory: Path, required_files: Sequence[str]) -> Dict[str, Any]:
    existing_files = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    missing_files = [name for name in required_files if name not in existing_files]
    weight_files = _collect_weight_files(directory)
    missing_weight_hint = [] if weight_files else ["pytorch_model*.bin or *.safetensors"]
    return {
        "path": str(directory),
        "dir_exists": directory.is_dir(),
        "existing_files": existing_files,
        "required_files": list(required_files),
        "missing_required_files": missing_files,
        "checked_weight_patterns": list(_WEIGHT_PATTERNS),
        "weight_files_found": weight_files,
        "missing_weight_files": missing_weight_hint,
        "ready": directory.is_dir() and not missing_files and bool(weight_files),
    }


def _fold_model_check(directory: Path, label: str) -> Dict[str, Any]:
    existing = sorted(path.name for path in directory.glob("*_bestmodel.pth") if path.is_file())
    expected = [f"{index}_bestmodel.pth" for index in range(10)]
    missing = [name for name in expected if name not in existing]
    return {
        "label": label,
        "path": str(directory),
        "dir_exists": directory.is_dir(),
        "expected_count": 10,
        "available_count": len(existing),
        "available_files": existing,
        "missing_files": missing,
        "ready": directory.is_dir() and not missing,
    }


def _asset_status(root: Path, resolved_model_paths: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    model_dir = Path(str(resolved_model_paths["model_dir"]))
    prot_t5_dir = Path(str(resolved_model_paths["prot_t5_dir"]))
    molt5_dir = Path(str(resolved_model_paths["molt5_dir"]))

    predict_script = root / "inference" / "predict.py"
    run_script = root / "inference" / "run_catapro.sh"
    sample_input = root / "samples" / "sample_inp.csv"
    output_preview = root / "inference" / "catapro_test-pred.csv"
    kcat_dir = model_dir / "kcat_models"
    km_dir = model_dir / "Km_models"
    act_dir = model_dir / "act_models"

    prot_t5_check = _hf_model_asset_check(prot_t5_dir, _PROT_T5_REQUIRED_FILES)
    molt5_check = _hf_model_asset_check(molt5_dir, _MOLT5_REQUIRED_FILES)
    kcat_check = _fold_model_check(kcat_dir, "kcat")
    km_check = _fold_model_check(km_dir, "Km")
    act_check = _fold_model_check(act_dir, "activity")

    asset_status = {
        "root_path": str(root),
        "model_dir": str(model_dir),
        "predict_script": str(predict_script),
        "run_script": str(run_script),
        "sample_input": str(sample_input),
        "output_preview": str(output_preview),
        "predict_script_exists": predict_script.is_file(),
        "sample_input_exists": sample_input.is_file(),
        "output_preview_exists": output_preview.is_file(),
        "kcat_fold_models": kcat_check["available_count"],
        "km_fold_models": km_check["available_count"],
        "act_fold_models": act_check["available_count"],
        "prot_t5_tokenizer_dir_exists": prot_t5_dir.is_dir(),
        "prot_t5_weight_files_available": bool(prot_t5_check["weight_files_found"]),
        "molt5_dir_exists": molt5_dir.is_dir(),
        "molt5_weight_files_available": bool(molt5_check["weight_files_found"]),
        "resolved_model_paths": dict(resolved_model_paths),
    }
    asset_check_details = {
        "resolved_model_paths": dict(resolved_model_paths),
        "predict_script": {
            "path": str(predict_script),
            "exists": predict_script.is_file(),
        },
        "sample_input": {
            "path": str(sample_input),
            "exists": sample_input.is_file(),
        },
        "output_preview": {
            "path": str(output_preview),
            "exists": output_preview.is_file(),
        },
        "prot_t5": prot_t5_check,
        "molt5": molt5_check,
        "kcat_models": kcat_check,
        "km_models": km_check,
        "activity_models": act_check,
    }
    return asset_status, asset_check_details


def _build_input_rows(
    candidate_sequence_rows: Sequence[dict[str, Any]],
    substrate_smiles: str | None,
) -> list[dict[str, Any]]:
    if not substrate_smiles:
        return []
    rows: list[dict[str, Any]] = []
    for item in candidate_sequence_rows[:10]:
        enzyme_name = str(item.get("enzyme_id") or item.get("matched_candidate") or "").strip()
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
    asset_check_details: Dict[str, Any],
) -> tuple[list[str], list[str]]:
    input_not_ready: list[str] = []
    runtime_not_ready: list[str] = []
    if not substrate_smiles:
        input_not_ready.append("substrate_smiles")
    if not candidate_rows_ready:
        input_not_ready.append("candidate_sequence_rows")
    for module_name, ok in dependency_status.items():
        if not ok:
            runtime_not_ready.append(f"python_module:{module_name}")
    if not asset_status["predict_script_exists"]:
        runtime_not_ready.append("catapro_predict_script")
    if not asset_status["sample_input_exists"]:
        runtime_not_ready.append("catapro_sample_input")
    if not asset_check_details["kcat_models"]["ready"]:
        runtime_not_ready.append("catapro_kcat_fold_models")
    if not asset_check_details["km_models"]["ready"]:
        runtime_not_ready.append("catapro_km_fold_models")
    if not asset_check_details["activity_models"]["ready"]:
        runtime_not_ready.append("catapro_activity_fold_models")
    if not asset_check_details["molt5"]["ready"]:
        runtime_not_ready.append("molt5_assets")
    if not asset_check_details["prot_t5"]["ready"]:
        runtime_not_ready.append("prot_t5_assets")
    return input_not_ready, runtime_not_ready


def _required_asset_descriptions(
    *,
    input_not_ready: Sequence[str],
    runtime_not_ready: Sequence[str],
    asset_check_details: Dict[str, Any],
) -> list[str]:
    required: list[str] = []
    if "substrate_smiles" in input_not_ready:
        required.append("Provide a canonicalizable substrate SMILES string.")
    if "candidate_sequence_rows" in input_not_ready:
        required.append("Provide at least one sequence-level candidate row with Enzyme_id/type/sequence/smiles compatibility.")
    if "prot_t5_assets" in runtime_not_ready:
        prot = asset_check_details["prot_t5"]
        if prot["dir_exists"] and prot["missing_weight_files"]:
            required.append(
                f"ProtT5 weights missing under {prot['path']}; expected a file matching {', '.join(prot['missing_weight_files'])}."
            )
        elif not prot["dir_exists"]:
            required.append(f"ProtT5 model directory not found: {prot['path']}.")
        elif prot["missing_required_files"]:
            required.append(
                f"ProtT5 directory is incomplete at {prot['path']}; missing files: {', '.join(prot['missing_required_files'])}."
            )
    if "molt5_assets" in runtime_not_ready:
        molt = asset_check_details["molt5"]
        if molt["dir_exists"] and molt["missing_weight_files"]:
            required.append(
                f"MolT5 weights missing under {molt['path']}; expected a file matching {', '.join(molt['missing_weight_files'])}."
            )
        elif not molt["dir_exists"]:
            required.append(f"MolT5 model directory not found: {molt['path']}.")
        elif molt["missing_required_files"]:
            required.append(
                f"MolT5 directory is incomplete at {molt['path']}; missing files: {', '.join(molt['missing_required_files'])}."
            )
    for code, label in (
        ("catapro_kcat_fold_models", "kcat_models"),
        ("catapro_km_fold_models", "km_models"),
        ("catapro_activity_fold_models", "activity_models"),
    ):
        if code in runtime_not_ready:
            details = asset_check_details[label]
            required.append(
                f"{label} under {details['path']} is incomplete; missing files: {', '.join(details['missing_files']) or 'unknown'}."
            )
    return required[:8]


def _next_step_instructions(
    *,
    substrate_context: Dict[str, Any],
    candidate_rows_ready: bool,
    input_not_ready: Sequence[str],
    runtime_not_ready: Sequence[str],
    asset_check_details: Dict[str, Any],
) -> list[str]:
    instructions: list[str] = []
    if substrate_context.get("status") == "substrate_context_without_explicit_smiles":
        instructions.append("Provide a canonicalizable substrate SMILES string so the ranking bridge can prepare CataPro inputs.")
    elif substrate_context.get("status") == "no_substrate_context":
        instructions.append("Add a substrate or SMILES string if you want sequence-level candidate ranking instead of only candidate mining.")
    if not candidate_rows_ready:
        instructions.append("Attach or derive real candidate sequence rows so the ranking bridge can write CataPro input rows.")
    if "prot_t5_assets" in runtime_not_ready:
        prot = asset_check_details["prot_t5"]
        instructions.append(
            f"Populate the ProtT5 directory at {prot['path']} with the missing weight files ({', '.join(prot['missing_weight_files']) or 'see checked patterns'}) before attempting real ranking."
        )
    if "molt5_assets" in runtime_not_ready:
        molt = asset_check_details["molt5"]
        instructions.append(
            f"Populate the MolT5 directory at {molt['path']} with the required tokenizer/weight files before attempting real ranking."
        )
    if any(item.startswith("python_module:") for item in runtime_not_ready):
        instructions.append("Install the missing Python dependencies in the current runtime environment before attempting CataPro inference.")
    if not instructions and not input_not_ready and not runtime_not_ready:
        instructions.append("Ranking inputs and local assets are ready; the bridge can attempt local CataPro inference without triggering notebook execution.")
    return instructions[:6]


def _ensure_dir_link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def _prepare_runtime_model_dir(
    *,
    run_dir: Path,
    resolved_model_paths: Mapping[str, Any],
) -> Dict[str, Any]:
    model_dir = Path(str(resolved_model_paths["model_dir"]))
    prot_t5_dir = Path(str(resolved_model_paths["prot_t5_dir"]))
    molt5_dir = Path(str(resolved_model_paths["molt5_dir"]))
    expected_prot = model_dir / "prot_t5_xl_uniref50"
    expected_molt = model_dir / "molt5-base-smiles2caption"

    if expected_prot == prot_t5_dir and expected_molt == molt5_dir:
        return {
            "model_dpath": str(model_dir),
            "assembled_runtime_model_dir": None,
            "used_path_override_bundle": False,
        }

    runtime_model_dir = run_dir / "runtime_models"
    runtime_model_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dir_link(model_dir / "kcat_models", runtime_model_dir / "kcat_models")
    _ensure_dir_link(model_dir / "Km_models", runtime_model_dir / "Km_models")
    _ensure_dir_link(model_dir / "act_models", runtime_model_dir / "act_models")
    _ensure_dir_link(prot_t5_dir, runtime_model_dir / "prot_t5_xl_uniref50")
    _ensure_dir_link(molt5_dir, runtime_model_dir / "molt5-base-smiles2caption")
    return {
        "model_dpath": str(runtime_model_dir),
        "assembled_runtime_model_dir": str(runtime_model_dir),
        "used_path_override_bundle": True,
    }


def _run_catapro_prediction(
    *,
    root: Path,
    resolved_model_paths: Mapping[str, Any],
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

    runtime_model_bundle = _prepare_runtime_model_dir(
        run_dir=run_dir,
        resolved_model_paths=resolved_model_paths,
    )
    cmd = [
        sys.executable,
        "predict.py",
        "-inp_fpath",
        str(input_csv),
        "-model_dpath",
        str(runtime_model_bundle["model_dpath"]),
        "-batch_size",
        "64",
        "-device",
        str(resolved_model_paths["device"]),
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

    base_payload = {
        "run_dir": str(run_dir),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "command": cmd,
        "cwd": str(root / "inference"),
        "used_device": str(resolved_model_paths["device"]),
        "model_dpath": str(runtime_model_bundle["model_dpath"]),
        "assembled_runtime_model_dir": runtime_model_bundle["assembled_runtime_model_dir"],
        "used_path_override_bundle": runtime_model_bundle["used_path_override_bundle"],
        "stdout_summary": (proc.stdout or "").strip()[:800] or None,
        "stderr_summary": (proc.stderr or "").strip()[:800] or None,
        "returncode": proc.returncode,
    }
    if proc.returncode != 0 or not output_csv.exists():
        return {
            "status": "ranking_run_failed",
            "results": [],
            **base_payload,
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
        **base_payload,
    }


def build_catapro_ranking_bridge(
    *,
    query: str,
    candidate_sequence_rows: Sequence[dict[str, Any]],
    candidate_sequences_status: Dict[str, Any],
    candidate_sequence_rows_status: Dict[str, Any] | None = None,
    path_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_model_paths = _resolve_model_paths(path_overrides)
    root = Path(str(resolved_model_paths["catapro_root"]))
    dependency_status = _dependency_status()
    asset_status, asset_check_details = _asset_status(root, resolved_model_paths)
    substrate_context = _extract_substrate_context(query)
    substrate_smiles = substrate_context.get("canonical_smiles")
    input_rows = _build_input_rows(candidate_sequence_rows, substrate_smiles)
    preview_rows = _build_preview_rows(input_rows)

    candidate_rows_ready = bool(input_rows)
    input_not_ready, runtime_not_ready = _missing_requirements(
        substrate_smiles=substrate_smiles,
        candidate_rows_ready=candidate_rows_ready,
        dependency_status=dependency_status,
        asset_status=asset_status,
        asset_check_details=asset_check_details,
    )
    required_assets = _required_asset_descriptions(
        input_not_ready=input_not_ready,
        runtime_not_ready=runtime_not_ready,
        asset_check_details=asset_check_details,
    )
    next_step_instructions = _next_step_instructions(
        substrate_context=substrate_context,
        candidate_rows_ready=candidate_rows_ready,
        input_not_ready=input_not_ready,
        runtime_not_ready=runtime_not_ready,
        asset_check_details=asset_check_details,
    )
    ranking_ready = not input_not_ready and not runtime_not_ready

    if not substrate_smiles:
        ranking_status = "awaiting_substrate_smiles"
    elif not candidate_rows_ready:
        ranking_status = "awaiting_candidate_sequence_mapping"
    elif any(item.startswith("python_module:") for item in runtime_not_ready):
        ranking_status = "blocked_missing_runtime_dependencies"
    elif runtime_not_ready:
        ranking_status = "blocked_incomplete_model_assets"
    else:
        ranking_status = "ranking_input_ready"

    ranking_results: list[dict[str, Any]] = []
    ranking_run_details: Dict[str, Any] = {
        "attempted": False,
        "resolved_model_paths": resolved_model_paths,
    }
    if ranking_ready:
        run_payload = _run_catapro_prediction(
            root=root,
            resolved_model_paths=resolved_model_paths,
            input_rows=input_rows,
        )
        ranking_status = str(run_payload.get("status") or ranking_status)
        ranking_results = list(run_payload.get("results") or [])
        ranking_run_details = {
            "attempted": True,
            "run_dir": run_payload.get("run_dir"),
            "input_csv": run_payload.get("input_csv"),
            "output_csv": run_payload.get("output_csv"),
            "command": run_payload.get("command"),
            "cwd": run_payload.get("cwd"),
            "used_device": run_payload.get("used_device"),
            "model_dpath": run_payload.get("model_dpath"),
            "assembled_runtime_model_dir": run_payload.get("assembled_runtime_model_dir"),
            "used_path_override_bundle": run_payload.get("used_path_override_bundle"),
            "stdout_summary": run_payload.get("stdout_summary"),
            "stderr_summary": run_payload.get("stderr_summary"),
            "returncode": run_payload.get("returncode"),
            "result_row_count": len(ranking_results),
        }

    asset_runtime_codes = {
        "catapro_predict_script",
        "catapro_sample_input",
        "catapro_kcat_fold_models",
        "catapro_km_fold_models",
        "catapro_activity_fold_models",
        "molt5_assets",
        "prot_t5_assets",
    }
    if ranking_status == "ranking_run_failed":
        why_not_runnable = [
            str(ranking_run_details.get("stderr_summary") or ranking_run_details.get("stdout_summary") or "CataPro inference failed after input preparation.")
        ]
    elif ranking_ready:
        why_not_runnable = []
    else:
        why_not_runnable = list(required_assets or next_step_instructions)
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
        "resolved_model_paths": resolved_model_paths,
        "asset_check_details": asset_check_details,
        "ranking_model": {
            "name": "CataPro",
            "bridge_type": "local_inference_bridge",
            "entrypoint": asset_status["predict_script"],
            "whether_model_assets_available": not any(code in asset_runtime_codes for code in runtime_not_ready),
            "whether_python_deps_available": all(dependency_status.values()),
            "dependency_status": dependency_status,
            "asset_status": asset_status,
            "asset_check_details": asset_check_details,
            "resolved_model_paths": resolved_model_paths,
            "required_input_columns": ["Enzyme_id", "type", "sequence", "smiles"],
            "candidate_sequence_bundle_status": candidate_sequences_status.get("status"),
            "candidate_sequence_rows_status": (
                candidate_sequence_rows_status.get("status")
                if isinstance(candidate_sequence_rows_status, Mapping)
                else None
            ),
        },
        "ranking_results": ranking_results,
        "why_not_runnable": why_not_runnable,
        "input_not_ready": input_not_ready,
        "runtime_not_ready": runtime_not_ready,
        "required_assets": required_assets,
        "prepared_input_preview": prepared_input_preview,
        "ranking_input_preview": prepared_input_preview,
        "next_step_instructions": next_step_instructions,
        "ranking_run_details": ranking_run_details,
    }
