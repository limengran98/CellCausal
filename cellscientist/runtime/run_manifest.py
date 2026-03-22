from __future__ import annotations

import json
import platform as platform_lib
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..pipeline.config import load_pipeline_config
from ..pipeline.utils import project_root, resolve_h5_path_unified
from .state import SessionState


_KEY_PACKAGE_SPECS = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("requests", "requests"),
    ("PyYAML", "pyyaml"),
    ("nbclient", "nbclient"),
    ("jupyter_client", "jupyter_client"),
    ("rdkit", "rdkit"),
)

_NOTEBOOK_PROMPT_FILES = (
    "prompts/pipeline_prompt.yaml",
    "prompts/review_optimize.yaml",
    "prompts/autofix.yml",
    "prompts/experiment_report.yaml",
)


@dataclass
class RunManifest:
    """Lightweight JSON-serializable record for runs and evals."""

    run_id: str
    timestamp: str
    context: str
    git_commit: str | None
    branch_name: str | None
    python_version: str
    platform: str
    key_package_versions: dict[str, str | None]
    task_type: str | None
    requested_actions: list[str] = field(default_factory=list)
    final_status: str | None = None
    query: str | None = None
    skill_trace: list[str] = field(default_factory=list)
    skill_versions: list[dict[str, Any]] = field(default_factory=list)
    model_provider_info: dict[str, Any] = field(default_factory=dict)
    fallback_chain: list[dict[str, Any]] = field(default_factory=list)
    prompt_template_versions: list[dict[str, Any]] = field(default_factory=list)
    dataset_versions: list[dict[str, Any]] = field(default_factory=list)
    case_file: str | None = None
    eval_case_files: list[str] = field(default_factory=list)
    trial_dir: str | None = None
    artifact_summary: list[dict[str, Any]] = field(default_factory=list)
    llm_resolution: dict[str, Any] | None = None
    final_provider_used: Any = None
    notebook_record: dict[str, Any] | None = None
    data_availability: dict[str, Any] = field(default_factory=dict)
    code_availability: dict[str, Any] = field(default_factory=dict)
    reporting_summary: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    value = completed.stdout.strip()
    return value or None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution_name, label in _KEY_PACKAGE_SPECS:
        try:
            versions[label] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _relative_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(Path(project_root()).resolve()))
    except Exception:
        return str(raw)


def _file_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    exists = target.exists()
    record = {
        "path": _relative_path(target) or str(target),
        "exists": exists,
        "size_bytes": target.stat().st_size if exists and target.is_file() else None,
        "modified_at": (
            datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if exists
            else None
        ),
    }
    return record


def _artifact_summary(state: SessionState) -> list[dict[str, Any]]:
    return [
        {
            "type": artifact.type,
            "name": artifact.name,
            "metadata": dict(artifact.metadata),
        }
        for artifact in state.artifacts
    ]


def _find_nested_key(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_nested_key(value, key)
            if found is not None:
                return found
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            found = _find_nested_key(value, key)
            if found is not None:
                return found
    return None


def _parse_skill_names(skill_trace: Iterable[str]) -> list[str]:
    skill_names: list[str] = []
    seen = set()
    for trace_entry in skill_trace:
        _, _, skill_name = str(trace_entry).partition(":")
        if not skill_name or skill_name in seen:
            continue
        seen.add(skill_name)
        skill_names.append(skill_name)
    return skill_names


def _matched_skill_versions(
    skill_catalog: Sequence[dict[str, Any]] | None,
    skill_trace: Sequence[str],
) -> list[dict[str, Any]]:
    if not skill_catalog:
        return []

    catalog_by_name = {item.get("name"): item for item in skill_catalog if item.get("name")}
    matched: list[dict[str, Any]] = []
    for skill_name in _parse_skill_names(skill_trace):
        metadata = catalog_by_name.get(skill_name)
        if not metadata:
            continue
        matched.append(
            {
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "aliases": metadata.get("aliases", []),
                "triggers": metadata.get("triggers", []),
                "supported_task_types": metadata.get("supported_task_types", []),
                "skill_package": metadata.get("skill_package", {}),
            }
        )
    return matched


def _prompt_template_versions(task_type: str | None, skill_trace: Sequence[str]) -> list[dict[str, Any]]:
    if task_type != "legacy_notebook" and not any("notebook" in trace for trace in skill_trace):
        return []

    root = Path(project_root())
    return [_file_record(root / relative_path) for relative_path in _NOTEBOOK_PROMPT_FILES]


def _dataset_versions(task_type: str | None) -> list[dict[str, Any]]:
    if task_type == "legacy_notebook":
        try:
            cfg = load_pipeline_config()
        except Exception:
            return []

        h5_path = resolve_h5_path_unified(cfg)
        dataset_record = {
            "dataset_name": cfg.get("dataset_name"),
            "split_name": cfg.get("split_name"),
            "data_path": _relative_path(h5_path) if h5_path else None,
            "source": "pipeline_config",
        }
        if h5_path:
            dataset_record["file_record"] = _file_record(h5_path)
        return [dataset_record]

    if task_type == "drug_analysis":
        return [
            {
                "dataset_name": "drug_analysis_local_seeded_context",
                "data_path": None,
                "source": "repo_local_lookup_and_biokb",
            }
        ]

    return []


def _extract_trial_dir(state: SessionState, result: Any) -> str | None:
    candidates: list[Any] = []
    if isinstance(result, Mapping):
        candidates.extend(
            [
                result.get("trial_dir"),
                (result.get("details") or {}).get("target_trial_dir") if isinstance(result.get("details"), Mapping) else None,
            ]
        )

    if state.last_notebook_artifact is not None:
        candidates.append(state.last_notebook_artifact.trial_dir)
    if state.last_notebook_run_result is not None:
        candidates.append(state.last_notebook_run_result.trial_dir)

    for artifact in state.artifacts:
        candidates.append(artifact.metadata.get("trial_dir"))

    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def _extract_notebook_record(state: SessionState, result: Any) -> dict[str, Any] | None:
    notebook_path = None
    review_report_path = None
    error_log_path = None
    patched_notebook_path = None
    status = None

    if isinstance(result, Mapping):
        notebook_path = result.get("notebook_path") or result.get("target_notebook_path")
        review_report_path = result.get("review_report_path")
        error_log_path = result.get("error_log_path")
        patched_notebook_path = result.get("patched_notebook_path")
        status = result.get("status")

    if state.last_notebook_run_result is not None:
        notebook_path = notebook_path or state.last_notebook_run_result.notebook_path
        error_log_path = error_log_path or state.last_notebook_run_result.error_log_path
        status = status or state.last_notebook_run_result.status

    if state.last_notebook_artifact is not None:
        notebook_path = notebook_path or state.last_notebook_artifact.path

    for artifact in state.artifacts:
        if artifact.type == "review_report":
            review_report_path = review_report_path or artifact.metadata.get("report_path")
        if artifact.type == "notebook_run":
            error_log_path = error_log_path or artifact.metadata.get("error_log_path")
            notebook_path = notebook_path or artifact.metadata.get("notebook_path")
            status = status or artifact.metadata.get("status")
        if artifact.type == "patched_notebook":
            patched_notebook_path = patched_notebook_path or artifact.metadata.get("path")

    trial_dir = _extract_trial_dir(state, result)
    if not any([notebook_path, review_report_path, error_log_path, patched_notebook_path, trial_dir]):
        return None

    return {
        "notebook_path": _relative_path(notebook_path) if notebook_path else None,
        "review_report_path": _relative_path(review_report_path) if review_report_path else None,
        "error_log_path": _relative_path(error_log_path) if error_log_path else None,
        "patched_notebook_path": _relative_path(patched_notebook_path) if patched_notebook_path else None,
        "trial_dir": _relative_path(trial_dir) if trial_dir else None,
        "status": str(status or "unknown"),
    }


def _default_data_availability() -> dict[str, Any]:
    return {
        "data_repository_target": "",
        "accession_or_doi": "",
        "url": "",
        "controlled_access_reason": "",
        "notes": "Populate with repository target, DOI/accession, and any controlled-access rationale for publication.",
    }


def _default_code_availability() -> dict[str, Any]:
    return {
        "code_release_target": "",
        "url": "",
        "license": "",
        "notes": "Populate with code host, release tag, and final software license for the manuscript.",
    }


def _default_reporting_summary(task_type: str | None) -> dict[str, Any]:
    return {
        "study_domain": task_type or "unknown",
        "reporting_summary_target": "Nature Portfolio life sciences reporting summary",
        "notes": (
            "Record experimental design, exclusions, replication, statistics, software, model provider chain, "
            "and any life-science reporting details needed for transparent reproducibility."
        ),
    }


def build_run_manifest(
    *,
    run_id: str,
    context: str,
    query: str | None,
    state: SessionState | None = None,
    result: Any = None,
    skill_catalog: Sequence[dict[str, Any]] | None = None,
    case_file: str | None = None,
    eval_case_files: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RunManifest:
    """Build a JSON-serializable manifest for an interactive run or eval."""

    task_type = state.intent.task_type if state is not None and state.intent is not None else None
    requested_actions = (
        list(state.intent.requested_actions) if state is not None and state.intent is not None else []
    )
    final_status = None
    if isinstance(result, Mapping):
        final_status = str(result.get("status") or "ok")

    skill_trace = list(state.skill_trace) if state is not None else []
    llm_resolution = _find_nested_key(result, "llm_resolution")
    final_provider_used = _find_nested_key(result, "final_provider_used")
    llm_attempts = _find_nested_key(result, "llm_attempts") or []
    notebook_record = _extract_notebook_record(state, result) if state is not None else None
    trial_dir = _extract_trial_dir(state, result) if state is not None else None

    primary_provider = None
    if isinstance(llm_resolution, Mapping):
        primary_provider = (
            llm_resolution.get("primary_provider")
            or llm_resolution.get("primary")
            or llm_resolution.get("provider")
        )

    return RunManifest(
        run_id=run_id,
        timestamp=_utc_now(),
        context=context,
        git_commit=_git_value("rev-parse", "--short", "HEAD"),
        branch_name=_git_value("branch", "--show-current"),
        python_version=sys.version.split()[0],
        platform=platform_lib.platform(),
        key_package_versions=_package_versions(),
        task_type=task_type,
        requested_actions=requested_actions,
        final_status=final_status,
        query=query,
        skill_trace=skill_trace,
        skill_versions=_matched_skill_versions(skill_catalog, skill_trace),
        model_provider_info={
            "primary_provider": primary_provider,
            "final_provider_used": final_provider_used,
        },
        fallback_chain=list(llm_attempts) if isinstance(llm_attempts, list) else [],
        prompt_template_versions=_prompt_template_versions(task_type, skill_trace),
        dataset_versions=_dataset_versions(task_type),
        case_file=case_file,
        eval_case_files=[str(path) for path in (eval_case_files or [])],
        trial_dir=_relative_path(trial_dir) if trial_dir else None,
        artifact_summary=_artifact_summary(state) if state is not None else [],
        llm_resolution=dict(llm_resolution) if isinstance(llm_resolution, Mapping) else None,
        final_provider_used=final_provider_used,
        notebook_record=notebook_record,
        data_availability=_default_data_availability(),
        code_availability=_default_code_availability(),
        reporting_summary=_default_reporting_summary(task_type),
        extra=dict(extra or {}),
    )


def write_manifest_json(manifest: RunManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, ensure_ascii=False, indent=2)
    return target


def append_notebook_record(
    *,
    state: SessionState,
    result: Any,
    run_id: str,
    context: str,
    case_id: str | None = None,
) -> Path | None:
    notebook_record = _extract_notebook_record(state, result)
    if notebook_record is None:
        return None

    index_path = Path(project_root()) / "records" / "notebooks" / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "context": context,
        "case_id": case_id,
        "timestamp": _utc_now(),
        **notebook_record,
    }
    with open(index_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return index_path


def record_runtime_run(
    *,
    run_id: str,
    query: str,
    state: SessionState,
    result: Any,
    skill_catalog: Sequence[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write interactive/runtime run manifests to records/ and trial dirs."""

    manifest = build_run_manifest(
        run_id=run_id,
        context="interactive_run",
        query=query,
        state=state,
        result=result,
        skill_catalog=skill_catalog,
    )

    records_root = Path(project_root()) / "records" / "system"
    system_path = write_manifest_json(
        manifest,
        records_root / f"{manifest.timestamp.replace(':', '').replace('-', '')}_{run_id}_run_manifest.json",
    )

    written_paths = {"system_manifest": str(system_path)}

    if manifest.trial_dir:
        trial_manifest_path = Path(project_root()) / manifest.trial_dir / "run_manifest.json"
        write_manifest_json(manifest, trial_manifest_path)
        written_paths["trial_manifest"] = str(trial_manifest_path)

    notebook_index = append_notebook_record(
        state=state,
        result=result,
        run_id=run_id,
        context="interactive_run",
    )
    if notebook_index is not None:
        written_paths["notebook_index"] = str(notebook_index)

    return written_paths


def record_eval_run(
    *,
    run_id: str,
    summary: Mapping[str, Any],
    case_files: Sequence[str],
    result_dir: str | Path,
    skill_catalog: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Write a paper-oriented eval manifest under records/evals/."""

    manifest = build_run_manifest(
        run_id=run_id,
        context="eval_run",
        query=None,
        state=None,
        result={"status": "completed"},
        skill_catalog=skill_catalog,
        eval_case_files=case_files,
        extra={
            "eval_summary": dict(summary),
            "result_dir": _relative_path(result_dir) or str(result_dir),
        },
    )

    target = Path(project_root()) / "records" / "evals" / f"{run_id}_eval_manifest.json"
    return write_manifest_json(manifest, target)
