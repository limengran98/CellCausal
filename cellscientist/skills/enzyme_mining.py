from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..pipeline.utils import project_root
from ..runtime.notebook_handoff import (
    append_experiment_scaffold_artifact,
    build_notebook_handoff_payload,
    unique_nonempty,
    wants_notebook_handoff,
)
from ..runtime.state import Artifact, SessionState
from ..tools.enzyme_export import export_enzyme_mining_artifacts
from ..tools.enzyme_processing.candidate_sequence_mapping import build_candidate_sequence_rows
from ..tools.enzyme_processing.dedupe_sequences import summarize_exact_dedupe_from_zip
from ..tools.enzyme_processing.domain_filter import build_domain_filtering_steps
from ..tools.enzyme_processing.fasta_merge import summarize_local_sequence_bundle
from ..tools.enzyme_lookup import lookup_enzyme_candidates, normalize_enzyme_focus
from ..tools.enzyme_ranking.catapro_bridge import build_catapro_ranking_bridge
from ..tools.enzyme_sources.ebi_search import summarize_ebi_candidate_source
from ..tools.enzyme_sources.jgi_fetch import summarize_jgi_candidate_source
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
    substrate_context: Dict[str, Any],
    substrate_smiles: str | None,
    ranking_status: str,
    ranking_ready: bool,
    next_step_instructions: List[str],
) -> dict[str, Any]:
    candidate_names = [str(item.get("enzyme") or "").strip() for item in candidate_enzymes if str(item.get("enzyme") or "").strip()]
    required_inputs = [
        "user-provided validation dataset or assay readout",
        "candidate enzyme list",
        "pathway or phenotype context",
        "optional literature or prior-knowledge table",
    ]
    notes = [
        "This is scaffold-only handoff metadata for a later notebook generation step.",
        "The enzyme-mining skill should not auto-trigger notebook execution.",
        f"Current ranking bridge status: {ranking_status}",
    ]
    if substrate_smiles:
        notes.append(f"Canonical substrate SMILES available for ranking preparation: {substrate_smiles}")
    if ranking_status == "awaiting_substrate_smiles":
        required_inputs.append("substrate SMILES for CataPro ranking preview")
        notes.append("CataPro ranking cannot be prepared until a substrate SMILES string is provided.")
    elif ranking_status == "awaiting_candidate_sequence_mapping":
        required_inputs.append("candidate enzyme sequences mapped to final shortlisted enzyme IDs")
        notes.append("CataPro preview still needs sequence-level candidate mapping even if a substrate is known.")
    elif ranking_ready:
        notes.append("Ranking inputs are complete enough for a direct CataPro run when sequence rows and local assets are available.")

    if substrate_context.get("status") == "substrate_context_without_explicit_smiles":
        required_inputs.append("explicit substrate SMILES for ranking validation")

    notes.extend(unique_nonempty(next_step_instructions, limit=3))
    return build_notebook_handoff_payload(
        focus=f"Validate enzyme candidates for {query_focus}",
        validation_questions=unique_nonempty(
            list(next_questions)
            + [
                f"Can the notebook rank {', '.join(candidate_names[:3])} against the stated phenotype or pathway signal?",
                "Does the substrate-SMILES normalization stay consistent from query parsing through ranking input preparation?",
                "Which top-k candidate enzymes remain plausible after sequence-ranking review and evidence reconciliation?",
                "Which validation dataset or assay will be used to test the candidate ranking?",
            ],
            limit=5,
        ),
        recommended_notebook_sections=[
            "Question framing and focus normalization",
            "Candidate enzyme evidence table",
            "Pathway context and prioritization logic",
            "Sequence curation and filtering status",
            "Ranking bridge readiness and substrate assumptions",
            "Substrate-SMILES consistency check",
            "Sequence-ranking validation",
            "Top-k candidate review",
            "Evidence-table handoff",
            "Validation dataset assumptions",
            "Ranking outputs and uncertainty summary",
        ],
        required_inputs=required_inputs,
        suggested_analysis_modes=[
            "candidate_ranking",
            "substrate_smiles_consistency_check",
            "top_k_candidate_review",
            "pathway_context_review",
            "expression_or_phenotype_validation",
            "evidence_summary_table",
        ],
        notes=notes,
    )


def _candidate_bundle_path() -> Path:
    return Path(project_root()) / "references" / "enzyme_mining" / "output_sequences.zip"


def _build_candidate_sources(
    *,
    query_focus: str,
    bundle_summary: Dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        summarize_ebi_candidate_source(query_focus),
        summarize_jgi_candidate_source(query_focus),
        {
            "source_name": "ncbi_domainhits_batch",
            "status": (
                "local_domain_batch_bundle_available"
                if str(bundle_summary.get("status")) == "local_sequence_bundle_profiled"
                else "local_domain_batch_bundle_missing"
            ),
            "query_focus": query_focus,
            "input_bundle_path": str(bundle_summary.get("bundle_path") or ""),
            "split_fasta_files": int(bundle_summary.get("split_fasta_files") or 0),
            "raw_sequence_count": int(bundle_summary.get("raw_sequence_count") or 0),
            "workflow_mode": "playwright_batch_cdsearch_from_zipped_fasta",
            "notes": [
                "The user notebooks use NCBI CD-search / DomainHits as an annotation-filtering stage over the merged candidate FASTA pool.",
                "This runtime bridge currently reports bundle availability and filtering readiness without launching external web automation.",
            ],
        },
    ]


def _build_candidate_sequences_status(
    *,
    bundle_summary: Dict[str, Any],
    dedupe_summary: Dict[str, Any],
    candidate_enzymes: List[Dict[str, Any]],
) -> dict[str, Any]:
    sequence_mapped_candidates = sum(
        1 for item in candidate_enzymes if str(item.get("sequence") or "").strip()
    )
    return {
        "status": "local_bundle_profiled_with_exact_dedupe",
        "bundle_path": str(bundle_summary.get("bundle_path") or dedupe_summary.get("bundle_path") or ""),
        "split_fasta_files": int(bundle_summary.get("split_fasta_files") or 0),
        "raw_sequence_count": int(dedupe_summary.get("raw_sequence_count") or bundle_summary.get("raw_sequence_count") or 0),
        "unique_sequence_count": int(dedupe_summary.get("unique_sequence_count") or 0),
        "duplicate_sequence_count": int(dedupe_summary.get("duplicate_sequence_count") or 0),
        "dedupe_strategy": str(dedupe_summary.get("dedupe_strategy") or "exact_amino_acid_identity_keep_first"),
        "sequence_mapped_candidate_count": sequence_mapped_candidates,
        "sequence_mapping_status": (
            "candidate_sequences_available"
            if sequence_mapped_candidates > 0
            else "seeded_candidate_panel_without_sequence_mapping"
        ),
        "example_headers": list(bundle_summary.get("example_headers") or dedupe_summary.get("example_headers") or []),
        "notes": unique_nonempty(
            list(bundle_summary.get("notes") or [])
            + list(dedupe_summary.get("notes") or [])
            + [
                "This status summarizes the locally bundled candidate sequence pool independently of BBBC036 or notebook execution.",
            ],
            limit=6,
        ),
    }


class EnzymeMiningSkill(BaseSkill):
    """Native enzyme-mining MVP with evidence-driven candidate synthesis."""

    name = "enzyme-mining"
    description = "Native enzyme-mining skill for pathway-focused candidate enzyme prioritization with evidence and follow-up validation scaffolds."
    aliases = ["enzyme-mining", "enzyme mining", "pathway-enzyme-analysis"]
    triggers = ["候选酶", "脂代谢", "胆固醇代谢", "enzyme candidates", "pathway enzyme", "底物", "substrate", "SMILES"]
    supported_task_types = ["enzyme_mining"]

    def run(self, state: SessionState) -> dict[str, object]:
        normalized_focus = normalize_enzyme_focus(state.user_query)
        payload = lookup_enzyme_candidates(str(normalized_focus["focus_key"]))
        candidate_enzymes = list(payload.get("candidate_enzymes") or [])
        pathway_context = list(payload.get("pathway_context") or [])
        evidence = list(payload.get("evidence") or [])
        rationale = list(payload.get("rationale") or [])
        bundle_path = _candidate_bundle_path()
        bundle_summary = summarize_local_sequence_bundle(str(bundle_path))
        dedupe_summary = summarize_exact_dedupe_from_zip(str(bundle_path))
        candidate_sequences_status = _build_candidate_sequences_status(
            bundle_summary=bundle_summary,
            dedupe_summary=dedupe_summary,
            candidate_enzymes=candidate_enzymes,
        )
        candidate_sources = _build_candidate_sources(
            query_focus=str(payload.get("query_focus") or normalized_focus["query_focus"]),
            bundle_summary=bundle_summary,
        )
        candidate_sequence_rows_status, candidate_sequence_rows = build_candidate_sequence_rows(
            candidate_enzymes=candidate_enzymes,
            query_focus=str(payload.get("query_focus") or normalized_focus["query_focus"]),
            zip_path=str(bundle_path),
        )
        filtering_steps = build_domain_filtering_steps(
            raw_sequence_count=int(candidate_sequences_status["raw_sequence_count"]),
            unique_sequence_count=int(candidate_sequences_status["unique_sequence_count"]),
        )
        ranking_bridge = build_catapro_ranking_bridge(
            query=state.user_query,
            candidate_sequence_rows=candidate_sequence_rows,
            candidate_sequences_status=candidate_sequences_status,
            candidate_sequence_rows_status=candidate_sequence_rows_status,
        )
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
            "substrate_context": ranking_bridge["substrate_context"],
            "substrate_smiles": ranking_bridge["substrate_smiles"],
            "candidate_sources": candidate_sources,
            "candidate_sequences_status": candidate_sequences_status,
            "candidate_sequence_rows_status": candidate_sequence_rows_status,
            "candidate_sequence_row_count": len(candidate_sequence_rows),
            "candidate_sequence_rows": candidate_sequence_rows,
            "candidate_enzymes": candidate_enzymes,
            "filtering_steps": filtering_steps,
            "ranking_status": ranking_bridge["ranking_status"],
            "ranking_ready": ranking_bridge["ranking_ready"],
            "ranking_model": ranking_bridge["ranking_model"],
            "ranking_results": ranking_bridge["ranking_results"],
            "resolved_model_paths": ranking_bridge["resolved_model_paths"],
            "asset_check_details": ranking_bridge["asset_check_details"],
            "why_not_runnable": ranking_bridge["why_not_runnable"],
            "input_not_ready": ranking_bridge["input_not_ready"],
            "runtime_not_ready": ranking_bridge["runtime_not_ready"],
            "required_assets": ranking_bridge["required_assets"],
            "prepared_input_preview": ranking_bridge["prepared_input_preview"],
            "next_step_instructions": ranking_bridge["next_step_instructions"],
            "ranking_run_details": ranking_bridge["ranking_run_details"],
            "ranking_input_preview": ranking_bridge["ranking_input_preview"],
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
                substrate_context=ranking_bridge["substrate_context"],
                substrate_smiles=ranking_bridge["substrate_smiles"],
                ranking_status=str(result["ranking_status"]),
                ranking_ready=bool(result["ranking_ready"]),
                next_step_instructions=list(ranking_bridge["next_step_instructions"]),
            )
            result["notebook_ready"] = True
            result["experiment_scaffold"] = scaffold
        else:
            result["notebook_ready"] = False

        artifact_export = export_enzyme_mining_artifacts(
            session_id=state.session_id,
            focus_key=str(normalized_focus["focus_key"]),
            result=result,
        )
        result["artifact_export"] = artifact_export

        files = dict(artifact_export.get("files") or {})
        result_dir = artifact_export.get("result_dir")

        if should_handoff:
            append_experiment_scaffold_artifact(
                state,
                name=f"{str(normalized_focus['focus_key'])}_enzyme_validation_scaffold",
                content=result["experiment_scaffold"],
                source_skill=self.name,
                focus=str(result["query_focus"]),
                extra_metadata={
                    "task": "enzyme_mining",
                    "ranking_status": result["ranking_status"],
                    "ranking_model": result["ranking_model"].get("name"),
                    "result_dir": result_dir,
                    "artifact_path": files.get("experiment_scaffold_json"),
                },
            )

        state.artifacts.append(
            Artifact(
                type="enzyme_candidate_table",
                name=f"{str(normalized_focus['focus_key'])}_candidate_table",
                content={"path": files.get("candidate_table_csv")},
                metadata={
                    "skill": self.name,
                    "result_dir": result_dir,
                    "artifact_path": files.get("candidate_table_csv"),
                    "candidate_count": len(candidate_enzymes),
                },
            )
        )
        state.artifacts.append(
            Artifact(
                type="enzyme_filtering_summary",
                name=f"{str(normalized_focus['focus_key'])}_filtering_steps",
                content={"path": files.get("filtering_steps_json")},
                metadata={
                    "skill": self.name,
                    "result_dir": result_dir,
                    "artifact_path": files.get("filtering_steps_json"),
                    "step_count": len(filtering_steps),
                },
            )
        )
        state.artifacts.append(
            Artifact(
                type="enzyme_candidate_sequence_rows",
                name=f"{str(normalized_focus['focus_key'])}_candidate_sequence_rows",
                content={
                    "status_path": files.get("candidate_sequence_mapping_status_json"),
                    "rows_path": files.get("candidate_sequence_rows_csv"),
                },
                metadata={
                    "skill": self.name,
                    "result_dir": result_dir,
                    "artifact_path": files.get("candidate_sequence_rows_csv"),
                    "status_path": files.get("candidate_sequence_mapping_status_json"),
                    "candidate_sequence_row_count": len(candidate_sequence_rows),
                    "mapping_mode": candidate_sequence_rows_status.get("mapping_mode"),
                },
            )
        )
        state.artifacts.append(
            Artifact(
                type="enzyme_ranking_result",
                name=f"{str(normalized_focus['focus_key'])}_ranking_status",
                content={
                    "status_path": files.get("ranking_status_json"),
                    "result_path": files.get("ranking_results_csv"),
                    "preview_path": files.get("ranking_input_preview_csv"),
                    "run_details_path": files.get("ranking_run_details_json"),
                },
                metadata={
                    "skill": self.name,
                    "result_dir": result_dir,
                    "ranking_status": result["ranking_status"],
                    "ranking_ready": result["ranking_ready"],
                    "status_path": files.get("ranking_status_json"),
                    "result_path": files.get("ranking_results_csv"),
                    "preview_path": files.get("ranking_input_preview_csv"),
                    "run_details_path": files.get("ranking_run_details_json"),
                },
            )
        )

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
                    "candidate_source_count": len(candidate_sources),
                    "raw_sequence_count": candidate_sequences_status["raw_sequence_count"],
                    "unique_sequence_count": candidate_sequences_status["unique_sequence_count"],
                    "candidate_sequence_row_count": len(candidate_sequence_rows),
                    "evidence_count": len(evidence),
                    "substrate_smiles": result["substrate_smiles"],
                    "ranking_status": result["ranking_status"],
                    "ranking_ready": result["ranking_ready"],
                    "ranking_model_name": result["ranking_model"].get("name"),
                    "notebook_ready": result["notebook_ready"],
                    "result_dir": result_dir,
                    "result_json_path": files.get("enzyme_mining_result_json"),
                },
            )
        )
        return result
