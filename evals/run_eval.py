#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.runtime.orchestrator_v2 import OrchestratorV2
from cellscientist.runtime.run_manifest import append_notebook_record, record_eval_run
from cellscientist.skills.defaults import build_default_skills


CASE_FILES = {
    "drug_analysis": "drug_analysis_cases.json",
    "enzyme_mining": "enzyme_mining_cases.json",
    "notebook_workflow": "notebook_workflow_cases.json",
    "fallback": "fallback_cases.json",
}

MISSING = object()


def build_eval_orchestrator() -> tuple[OrchestratorV2, dict[str, dict[str, Any]]]:
    """Build the current repo-native runtime used by run_interactive.py."""

    registry = SkillRegistry(build_default_skills())
    catalog = {item["name"]: item for item in registry.skill_catalog()}
    return OrchestratorV2(registry), catalog


def load_case_file(path: str | Path) -> list[dict[str, Any]]:
    """Load one golden-case file."""

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Case file must contain a JSON list: {path}")
    return [dict(item) for item in data]


def load_case_suites(base_dir: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load all eval suites from the evals directory."""

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    return {
        suite_name: load_case_file(root / filename)
        for suite_name, filename in CASE_FILES.items()
    }


def _get_by_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return MISSING
    return current


def _payload_excerpt(
    *,
    task_type: str | None,
    requested_actions: Sequence[str],
    skill_trace: Sequence[str],
    artifacts: Sequence[dict[str, Any]],
    result: Any,
    max_chars: int = 1200,
) -> str:
    bundle = {
        "task_type": task_type,
        "requested_actions": list(requested_actions),
        "skill_trace": list(skill_trace),
        "artifacts": list(artifacts),
        "result": result,
    }
    raw = json.dumps(bundle, ensure_ascii=False, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "... [TRUNCATED]"


def _skill_packages_for_trace(
    skill_trace: Sequence[str],
    skill_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    seen = set()
    for trace_entry in skill_trace:
        _, _, skill_name = trace_entry.partition(":")
        if not skill_name or skill_name in seen:
            continue
        seen.add(skill_name)
        metadata = skill_catalog.get(skill_name, {})
        packages.append(
            {
                "name": skill_name,
                "skill_package": metadata.get("skill_package", {}),
            }
        )
    return packages


def _evaluate_case(
    orchestrator: OrchestratorV2,
    skill_catalog: dict[str, dict[str, Any]],
    suite_name: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    notes: list[str] = []
    state = None
    result: Any = None
    actual_task_type: str | None = None
    actual_requested_actions: list[str] = []
    actual_status = "exception"
    skill_trace: list[str] = []
    artifacts: list[dict[str, Any]] = []
    notebook_record_path: str | None = None

    try:
        state, result = orchestrator.run(case["query"])
        if state.intent is not None:
            actual_task_type = state.intent.task_type
            actual_requested_actions = list(state.intent.requested_actions)
        skill_trace = list(state.skill_trace)
        artifacts = [
            {
                "type": artifact.type,
                "name": artifact.name,
                "metadata": artifact.metadata,
            }
            for artifact in state.artifacts
        ]
        if isinstance(result, dict):
            actual_status = str(result.get("status") or "ok")
        else:
            actual_status = "non_dict_result"
        if state is not None:
            record_path = append_notebook_record(
                state=state,
                result=result,
                run_id=f"{suite_name}:{case.get('id')}",
                context="eval_case",
                case_id=str(case.get("id")),
            )
            notebook_record_path = str(record_path) if record_path is not None else None
    except Exception as exc:
        if case.get("should_not_crash"):
            notes.append("case crashed unexpectedly")
        notes.append(f"exception:{type(exc).__name__}")
        result = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }

    if case.get("expected_task_type") is not None and actual_task_type != case.get("expected_task_type"):
        notes.append(
            f"task_type mismatch: expected {case.get('expected_task_type')} got {actual_task_type}"
        )

    if case.get("expected_input_type") is not None:
        actual_input_type = result.get("input_type") if isinstance(result, dict) else None
        if actual_input_type != case.get("expected_input_type"):
            notes.append(
                f"input_type mismatch: expected {case.get('expected_input_type')} got {actual_input_type}"
            )

    expected_entity_contains = str(case.get("expected_entity_contains") or "").strip()
    if expected_entity_contains:
        entity_blob = json.dumps(
            (result.get("normalized_entity") if isinstance(result, dict) else {}),
            ensure_ascii=False,
            default=str,
        ).lower()
        if expected_entity_contains.lower() not in entity_blob:
            notes.append(
                f"normalized_entity missing expected token: {expected_entity_contains}"
            )

    for field_path in case.get("required_result_fields", []):
        if not isinstance(result, dict) or _get_by_path(result, str(field_path)) is MISSING:
            notes.append(f"missing required result field: {field_path}")

    expected_actions = case.get("expected_requested_actions")
    if expected_actions is not None and list(expected_actions) != actual_requested_actions:
        notes.append(
            "requested_actions mismatch: expected "
            f"{list(expected_actions)} got {actual_requested_actions}"
        )

    acceptable_statuses = case.get("acceptable_statuses")
    if acceptable_statuses is not None and actual_status not in list(acceptable_statuses):
        notes.append(
            f"status mismatch: expected one of {list(acceptable_statuses)} got {actual_status}"
        )

    expected_status = case.get("expected_status")
    if expected_status is not None and actual_status != expected_status:
        notes.append(f"status mismatch: expected {expected_status} got {actual_status}")

    required_trace = case.get("require_skill_trace_contains", [])
    for token in required_trace:
        if not any(str(token) in entry for entry in skill_trace):
            notes.append(f"skill_trace missing token: {token}")

    searchable_text = json.dumps(
        {
            "task_type": actual_task_type,
            "requested_actions": actual_requested_actions,
            "skill_trace": skill_trace,
            "artifacts": artifacts,
            "result": result,
        },
        ensure_ascii=False,
        default=str,
    ).lower()

    must_include_any = [str(item).lower() for item in case.get("must_include_any", []) if str(item).strip()]
    if must_include_any and not any(token in searchable_text for token in must_include_any):
        notes.append(f"must_include_any mismatch: {must_include_any}")

    must_not_include_any = [str(item).lower() for item in case.get("must_not_include_any", []) if str(item).strip()]
    forbidden = [token for token in must_not_include_any if token in searchable_text]
    if forbidden:
        notes.append(f"must_not_include_any matched: {forbidden}")

    passed = not notes
    if passed:
        notes = ["passed"]

    return {
        "suite": suite_name,
        "case_id": case.get("id"),
        "query": case.get("query"),
        "expected": case,
        "actual_task_type": actual_task_type,
        "actual_requested_actions": actual_requested_actions,
        "status": actual_status,
        "pass": passed,
        "notes": notes,
        "skill_trace": skill_trace,
        "artifact_types": [artifact["type"] for artifact in artifacts],
        "matched_skill_packages": _skill_packages_for_trace(skill_trace, skill_catalog),
        "notebook_record_path": notebook_record_path,
        "raw_result_excerpt": _payload_excerpt(
            task_type=actual_task_type,
            requested_actions=actual_requested_actions,
            skill_trace=skill_trace,
            artifacts=artifacts,
            result=result,
        ),
    }


def _build_summary(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_suite: dict[str, dict[str, int]] = {}
    for suite_name in CASE_FILES:
        suite_details = [detail for detail in details if detail["suite"] == suite_name]
        by_suite[suite_name] = {
            "total_cases": len(suite_details),
            "passed_cases": sum(1 for detail in suite_details if detail["pass"]),
            "failed_cases": sum(1 for detail in suite_details if not detail["pass"]),
        }

    failure_counter = Counter(
        note
        for detail in details
        if not detail["pass"]
        for note in detail["notes"]
        if note and note != "passed"
    )

    return {
        "total_cases": len(details),
        "passed_cases": sum(1 for detail in details if detail["pass"]),
        "failed_cases": sum(1 for detail in details if not detail["pass"]),
        "by_suite": by_suite,
        "failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in failure_counter.most_common(12)
        ],
    }


def _write_results(
    summary: dict[str, Any],
    details: Sequence[dict[str, Any]],
    output_root: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = output_root / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "summary.json"
    details_path = result_dir / "details.jsonl"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with open(details_path, "w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False, default=str) + "\n")

    return result_dir


def run_all_evals(
    *,
    base_dir: str | Path | None = None,
    selected_suites: Sequence[str] | None = None,
    max_cases: int | None = None,
    output_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Run the lightweight golden-set eval harness and persist results."""

    suites = load_case_suites(base_dir)
    suite_order = list(selected_suites) if selected_suites else list(CASE_FILES.keys())
    orchestrator, skill_catalog = build_eval_orchestrator()

    details: list[dict[str, Any]] = []
    for suite_name in suite_order:
        cases = list(suites[suite_name])
        if max_cases is not None:
            cases = cases[:max_cases]

        print(f"[eval] suite={suite_name} cases={len(cases)}")
        for case in cases:
            detail = _evaluate_case(orchestrator, skill_catalog, suite_name, case)
            details.append(detail)
            outcome = "PASS" if detail["pass"] else "FAIL"
            print(
                f"[eval] {outcome} {suite_name}/{detail['case_id']} "
                f"task={detail['actual_task_type']} status={detail['status']}"
            )

    summary = _build_summary(details)
    results_root = Path(output_root) if output_root is not None else Path(__file__).resolve().parent / "results"
    result_dir = _write_results(summary, details, results_root)
    eval_manifest_path = record_eval_run(
        run_id=result_dir.name,
        summary=summary,
        case_files=[str((Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent) / CASE_FILES[name]) for name in suite_order],
        result_dir=result_dir,
        skill_catalog=list(skill_catalog.values()),
    )
    summary["eval_manifest_path"] = str(eval_manifest_path)
    with open(result_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[eval] results written to {result_dir}")
    return summary, details, result_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight CellCausal eval harness.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted(CASE_FILES.keys()),
        help="Only run the selected suite(s). Repeat to run multiple suites.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional per-suite case limit for smoke runs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional custom output root. Defaults to evals/results.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_all_evals(
        selected_suites=args.suite,
        max_cases=args.max_cases,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
