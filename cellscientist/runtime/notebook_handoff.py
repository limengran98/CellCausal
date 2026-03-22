from __future__ import annotations

from typing import Any, Iterable, Sequence

from .state import Artifact, SessionState


_NOTEBOOK_HANDOFF_KEYWORDS = (
    "验证",
    "做实验",
    "实验验证",
    "后续验证",
    "进一步验证",
    "生成 notebook",
    "notebook",
    "可验证",
    "验证框架",
    "notebook 框架",
    "notebook scaffold",
    "validation notebook",
    "experiment scaffold",
    "further validation",
    "validate",
)


def wants_notebook_handoff(query: str, constraints: Sequence[str] | None = None) -> bool:
    lowered = query.lower()
    if constraints and "notebook_ready" in constraints:
        return True
    return any(keyword in lowered for keyword in _NOTEBOOK_HANDOFF_KEYWORDS)


def build_notebook_handoff_payload(
    *,
    focus: str,
    validation_questions: Sequence[str],
    recommended_notebook_sections: Sequence[str],
    required_inputs: Sequence[str],
    suggested_analysis_modes: Sequence[str],
    workflow_path: str = "generic_data",
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "focus": focus,
        "validation_questions": list(validation_questions),
        "recommended_notebook_sections": list(recommended_notebook_sections),
        "required_inputs": list(required_inputs),
        "suggested_analysis_modes": list(suggested_analysis_modes),
        "notes": list(notes or []),
        "handoff": {
            "surface": "notebook_execution_surface",
            "workflow_path": workflow_path,
            "mode": "scaffold_only",
            "auto_execute": False,
            "recommended_next_step": (
                "Use this scaffold as structured handoff metadata for a later notebook-generate step. "
                "Do not auto-trigger notebook execution from the native scientific skill."
            ),
        },
    }


def append_experiment_scaffold_artifact(
    state: SessionState,
    *,
    name: str,
    content: dict[str, Any],
    source_skill: str,
    focus: str,
    workflow_path: str = "generic_data",
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "skill": source_skill,
        "focus": focus,
        "workflow_path": workflow_path,
        "auto_execute": False,
        "handoff_mode": "scaffold_only",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    state.artifacts.append(
        Artifact(
            type="experiment_scaffold",
            name=name,
            content=content,
            metadata=metadata,
        )
    )


def unique_nonempty(values: Iterable[str], *, limit: int | None = None) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
        if limit is not None and len(ordered) >= limit:
            break
    return ordered
