from __future__ import annotations

from typing import Any, Dict, List

from ..runtime.notebook_handoff import (
    append_experiment_scaffold_artifact,
    build_notebook_handoff_payload,
    unique_nonempty,
    wants_notebook_handoff,
)
from ..runtime.state import Artifact, SessionState
from ..tools.enzyme_lookup import lookup_enzyme_candidates, normalize_enzyme_focus
from .base import BaseSkill


def _build_next_questions(
    *,
    query_focus: str,
    candidate_enzymes: List[Dict[str, Any]],
    pathway_context: List[str],
) -> list[str]:
    top_names = [str(item.get("enzyme") or "").strip() for item in candidate_enzymes[:2] if str(item.get("enzyme") or "").strip()]
    questions: list[str] = []
    if top_names:
        questions.append(
            f"Which experimental readout best separates the top candidates ({', '.join(top_names)}) in the {query_focus} context?"
        )
    if pathway_context:
        questions.append(
            f"Which pathway branch within {', '.join(pathway_context[:2])} matters most for prioritizing enzymes in this question?"
        )
    questions.append(
        "What orthogonal evidence source should be added next: expression, perturbation phenotype, metabolite profiling, or literature curation?"
    )
    return questions[:3]


def _build_experiment_scaffold(
    *,
    query_focus: str,
    candidate_enzymes: List[Dict[str, Any]],
    pathway_context: List[str],
    next_questions: List[str],
) -> dict[str, Any]:
    candidate_names = [str(item.get("enzyme") or "").strip() for item in candidate_enzymes if str(item.get("enzyme") or "").strip()]
    return build_notebook_handoff_payload(
        focus=f"Validate enzyme candidates for {query_focus}",
        validation_questions=unique_nonempty(
            list(next_questions)
            + [
                f"Can the notebook rank {', '.join(candidate_names[:3])} against the stated phenotype or pathway signal?",
                "Which validation dataset or assay will be used to test the candidate ranking?",
            ],
            limit=5,
        ),
        recommended_notebook_sections=[
            "Question framing and focus normalization",
            "Candidate enzyme evidence table",
            "Pathway context and prioritization logic",
            "Validation dataset assumptions",
            "Ranking outputs and uncertainty summary",
        ],
        required_inputs=[
            "user-provided validation dataset or assay readout",
            "candidate enzyme list",
            "pathway or phenotype context",
            "optional literature or prior-knowledge table",
        ],
        suggested_analysis_modes=[
            "candidate_ranking",
            "pathway_context_review",
            "expression_or_phenotype_validation",
            "evidence_summary_table",
        ],
        notes=[
            "This is scaffold-only handoff metadata for a later notebook generation step.",
            "The enzyme-mining skill should not auto-trigger notebook execution.",
        ],
    )


class EnzymeMiningSkill(BaseSkill):
    """Native enzyme-mining MVP with evidence-driven candidate synthesis."""

    name = "enzyme-mining"
    description = "Native enzyme-mining skill for pathway-focused candidate enzyme prioritization with evidence and follow-up validation scaffolds."
    aliases = ["enzyme-mining", "enzyme mining", "pathway-enzyme-analysis"]
    triggers = ["候选酶", "脂代谢", "胆固醇代谢", "enzyme candidates", "pathway enzyme"]
    supported_task_types = ["enzyme_mining"]

    def run(self, state: SessionState) -> dict[str, object]:
        normalized_focus = normalize_enzyme_focus(state.user_query)
        payload = lookup_enzyme_candidates(str(normalized_focus["focus_key"]))
        candidate_enzymes = list(payload.get("candidate_enzymes") or [])
        pathway_context = list(payload.get("pathway_context") or [])
        evidence = list(payload.get("evidence") or [])
        rationale = list(payload.get("rationale") or [])
        next_questions = _build_next_questions(
            query_focus=str(payload.get("query_focus") or normalized_focus["query_focus"]),
            candidate_enzymes=candidate_enzymes,
            pathway_context=pathway_context,
        )

        for item in evidence:
            evidence_id = item.get("id")
            if evidence_id and evidence_id not in state.evidence_ids:
                state.evidence_ids.append(str(evidence_id))

        result: dict[str, object] = {
            "task": "enzyme_mining",
            "query_focus": payload.get("query_focus") or normalized_focus["query_focus"],
            "normalized_focus": normalized_focus,
            "candidate_enzymes": candidate_enzymes,
            "rationale": rationale,
            "pathway_context": pathway_context,
            "evidence": evidence,
            "next_questions": next_questions,
        }

        should_handoff = wants_notebook_handoff(
            state.user_query,
            state.intent.constraints if state.intent is not None else [],
        )
        if should_handoff:
            scaffold = _build_experiment_scaffold(
                query_focus=str(result["query_focus"]),
                candidate_enzymes=candidate_enzymes,
                pathway_context=pathway_context,
                next_questions=next_questions,
            )
            result["notebook_ready"] = True
            result["experiment_scaffold"] = scaffold
            append_experiment_scaffold_artifact(
                state,
                name=f"{str(normalized_focus['focus_key'])}_enzyme_validation_scaffold",
                content=scaffold,
                source_skill=self.name,
                focus=str(result["query_focus"]),
                extra_metadata={"task": "enzyme_mining"},
            )
        else:
            result["notebook_ready"] = False

        state.artifacts.append(
            Artifact(
                type="enzyme_mining",
                name=f"{str(normalized_focus['focus_key'])}_enzyme_mining",
                content=result,
                metadata={
                    "skill": self.name,
                    "focus_key": normalized_focus["focus_key"],
                    "query_focus": result["query_focus"],
                    "candidate_count": len(candidate_enzymes),
                    "evidence_count": len(evidence),
                    "notebook_ready": result["notebook_ready"],
                },
            )
        )
        return result
