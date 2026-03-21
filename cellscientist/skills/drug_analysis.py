from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..core.bio_kb import (
    biokb_table_to_evidence_items,
    generate_biokb_semantic_table,
    persist_biokb_semantic_table,
)
from ..core.bio_kb.smiles_resolver import canonicalize_smiles
from ..evidence.models import EvidenceItem
from ..pipeline.config import load_pipeline_config
from ..pipeline.utils import project_root
from ..runtime.state import Artifact, SessionState
from ..tools.drug_lookup import (
    canonicalize_drug_name,
    find_drug_name_in_text,
    lookup_drug_name_by_smiles,
    lookup_drug_profile,
    lookup_seeded_smiles_for_drug,
)
from .base import BaseSkill

_SMILES_TOKEN_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{6,}")
_ENTITY_STOPWORDS = {
    "analyze",
    "analysis",
    "drug",
    "drugs",
    "smiles",
    "mechanism",
    "target",
    "targets",
    "indication",
    "indications",
    "safety",
    "risk",
    "adverse",
    "effects",
    "根据这个",
}


def _quiet_log(_message: str) -> None:
    """Suppress noisy BioKB logging inside the native drug-analysis skill."""


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "drug_analysis"


def _extract_smiles_candidate(query: str) -> Optional[str]:
    candidates = sorted(set(_SMILES_TOKEN_RE.findall(query)), key=len, reverse=True)
    for candidate in candidates:
        if not any(ch in candidate for ch in "=#()[]\\/"):
            continue
        canon = canonicalize_smiles(candidate)
        if canon.get("inchikey"):
            return candidate
    return None


def _extract_drug_name_candidate(state: SessionState) -> str:
    direct = find_drug_name_in_text(state.user_query)
    if direct != "unknown":
        return canonicalize_drug_name(direct)

    entities = list(state.intent.entities) if state.intent is not None else []
    for entity in entities:
        lowered = entity.lower()
        if lowered in _ENTITY_STOPWORDS:
            continue
        return canonicalize_drug_name(entity)

    lowered_query = state.user_query.lower()
    ascii_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", lowered_query)
    for token in ascii_tokens:
        if token in _ENTITY_STOPWORDS:
            continue
        return canonicalize_drug_name(token)
    return "unknown"


def _order_unique(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _workspace_for_entity(normalized_label: str) -> str:
    return os.path.join(project_root(), "results", "drug_analysis", _slugify(normalized_label))


def _build_drug_analysis_cfg(smiles: str, workspace_dir: str) -> Dict[str, Any]:
    pipe_cfg = load_pipeline_config()
    literature_cfg = deepcopy(pipe_cfg.get("literature") or {})
    bio_kb_cfg = deepcopy(literature_cfg.get("bio_kb") or {})

    literature_cfg["enabled"] = False
    literature_cfg["task_keywords"] = "drug target mechanism indication safety analysis"
    bio_kb_cfg["enabled"] = True
    bio_kb_cfg["smiles_list"] = [smiles]
    bio_kb_cfg["h5_path"] = None
    bio_kb_cfg["chembl_timeout"] = min(int(bio_kb_cfg.get("chembl_timeout") or 30), 8)
    bio_kb_cfg["reactome_timeout"] = min(int(bio_kb_cfg.get("reactome_timeout") or 30), 8)
    literature_cfg["bio_kb"] = bio_kb_cfg

    return {
        "dataset_name": "drug_analysis",
        "paths": {
            "literature_dir": os.path.join(workspace_dir, "literature"),
            "literature_knowledge_json": os.path.join(workspace_dir, "literature", "domain_knowledge.json"),
        },
        "literature": literature_cfg,
    }


def _biokb_item_to_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    citation = item.get("title") or item.get("url") or "BioKB semantic enrichment"
    return asdict(
        EvidenceItem(
            id=str(item.get("eid") or "B0"),
            source=str(item.get("source") or "biokb"),
            claim=str(item.get("snippet") or "BioKB-derived structure-to-target prior."),
            citation=str(citation),
            confidence=0.68 if str(item.get("eid") or "").startswith("B") else 0.3,
            metadata={
                "url": item.get("url"),
                "published": item.get("published"),
                "scraped_excerpt": item.get("scraped_excerpt"),
            },
        )
    )


def _run_biokb_analysis(smiles: Optional[str], workspace_dir: str) -> Dict[str, Any]:
    if not smiles:
        return {
            "semantic_table": None,
            "evidence": [],
            "targets": [],
            "pathways": [],
            "processes": [],
            "workspace_dir": workspace_dir,
            "status": "missing_smiles",
        }

    cfg = _build_drug_analysis_cfg(smiles, workspace_dir)
    semantic_table = generate_biokb_semantic_table(cfg, stage="drug_analysis", log=_quiet_log)
    persist_biokb_semantic_table(semantic_table, workspace_dir, log=_quiet_log)

    molecules = semantic_table.get("molecules") or []
    first_molecule = molecules[0] if molecules else {}
    targets = []
    for target in first_molecule.get("targets") or []:
        target_name = str(target.get("target_name") or target.get("gene_symbol") or "").strip()
        gene_symbol = str(target.get("gene_symbol") or "").strip()
        mechanism = str(target.get("mechanism") or "").strip()
        label = target_name or gene_symbol
        if gene_symbol and gene_symbol not in label:
            label = f"{label} ({gene_symbol})"
        if mechanism:
            label = f"{label} - {mechanism}" if label else mechanism
        if label:
            targets.append(label)

    pathways = [
        str(pathway.get("pathway_name") or pathway.get("pathway_id") or "").strip()
        for pathway in first_molecule.get("pathways") or []
        if str(pathway.get("pathway_name") or pathway.get("pathway_id") or "").strip()
    ]
    processes = [
        str(process.get("process") or "").strip()
        for process in first_molecule.get("inferred_processes") or []
        if str(process.get("process") or "").strip()
    ]
    evidence = [_biokb_item_to_evidence(item) for item in biokb_table_to_evidence_items(semantic_table)]

    return {
        "semantic_table": semantic_table,
        "evidence": evidence,
        "targets": _order_unique(targets),
        "pathways": _order_unique(pathways),
        "processes": _order_unique(processes),
        "workspace_dir": workspace_dir,
        "status": "biokb_ready",
    }


def _merge_evidence(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for group in groups:
        for item in group:
            item_id = str(item.get("id") or "")
            if item_id and item_id in seen:
                continue
            if item_id:
                seen.add(item_id)
            ordered.append(item)
    return ordered


def _synthesize_summary(
    *,
    normalized_name: Optional[str],
    input_type: str,
    profile: Dict[str, Any],
    biokb_context: Dict[str, Any],
    evidence_count: int,
) -> str:
    label = normalized_name or "This entity"
    local_targets = len(profile.get("targets") or [])
    biokb_targets = len(biokb_context.get("targets") or [])
    local_safety = len(profile.get("adverse_effects") or [])
    structure_phrase = (
        "with structure-native BioKB support"
        if biokb_context.get("targets") or biokb_context.get("processes")
        else "with limited structure-native support"
    )
    return (
        f"{label.capitalize()} was analyzed as a {input_type.replace('_', ' ')} input. "
        f"The current conclusion combines repo-local drug knowledge and resource-native mechanism priors {structure_phrase}. "
        f"Evidence currently supports {local_targets + biokb_targets} target or pathway-facing signals and "
        f"{local_safety} seeded safety considerations across {evidence_count} evidence items."
    )


def _synthesize_mechanism(
    *,
    normalized_name: Optional[str],
    profile: Dict[str, Any],
    biokb_context: Dict[str, Any],
) -> str:
    label = normalized_name or "This entity"
    local_targets = profile.get("targets") or []
    processes = biokb_context.get("processes") or []
    mechanisms = []
    if local_targets:
        mechanisms.append(
            f"{label.capitalize()} is locally framed around {', '.join(local_targets[:3])}."
        )
    if processes:
        mechanisms.append(
            f"Structure-native BioKB priors additionally point toward {', '.join(processes[:3])}."
        )
    if not mechanisms:
        mechanisms.append(
            f"No high-confidence mechanism synthesis is available yet for {label}; more target/pathway evidence is needed."
        )
    return " ".join(mechanisms)


def _build_next_questions(
    *,
    normalized_name: Optional[str],
    normalized_entity: Dict[str, Any],
    profile: Dict[str, Any],
    biokb_context: Dict[str, Any],
) -> List[str]:
    label = normalized_name or "this entity"
    questions: List[str] = []
    if not normalized_entity.get("canonical_smiles"):
        questions.append(f"Can we confirm a canonical structure for {label} to strengthen structure-native target mapping?")
    if not biokb_context.get("targets"):
        questions.append(f"Which curated target source should be used next to validate the mechanism claims for {label}?")
    if not (profile.get("adverse_effects") or []):
        questions.append(f"What primary safety liabilities or monitoring concerns should be added for {label}?")
    if not (profile.get("indications") or []):
        questions.append(f"What disease contexts or indication boundaries are best supported for {label}?")
    if not questions:
        questions.append(f"Which indication context and patient-risk subgroup matter most for the next {label} analysis pass?")
    return questions[:3]


class DrugAnalysisSkill(BaseSkill):
    """Native drug-analysis skill: normalize entity, gather evidence, and synthesize a structured conclusion."""

    name = "drug-analysis"
    description = "Native drug analysis skill with entity normalization, mechanism/target synthesis, safety framing, and evidence-driven next questions."
    aliases = ["drug-analysis", "drug analysis", "smiles-analysis"]
    triggers = ["药物分析", "机制和安全性", "根据这个 SMILES", "靶点和风险"]
    supported_task_types = ["drug_analysis"]

    def run(self, state: SessionState) -> dict[str, object]:
        parsed = self._normalize_input(state)
        normalized_name = parsed.get("normalized_drug_name")
        normalized_entity = parsed["normalized_entity"]

        profile = lookup_drug_profile(normalized_name or "unknown")
        workspace_dir = _workspace_for_entity(
            normalized_name
            or normalized_entity.get("canonical_smiles")
            or normalized_entity.get("raw_input")
            or "drug_analysis"
        )
        os.makedirs(workspace_dir, exist_ok=True)

        smiles_for_biokb = normalized_entity.get("canonical_smiles")
        biokb_context = _run_biokb_analysis(smiles_for_biokb, workspace_dir)

        targets = _order_unique(list(profile.get("targets") or []) + list(biokb_context.get("targets") or []))
        indications = _order_unique(list(profile.get("indications") or []))
        safety = _order_unique(list(profile.get("adverse_effects") or []))
        evidence = _merge_evidence(list(profile.get("evidence") or []), list(biokb_context.get("evidence") or []))

        if not indications:
            indications = ["Coverage gap: no curated indication synthesis is available yet for this entity."]
        if not safety:
            safety = ["Coverage gap: no curated safety synthesis is available yet for this entity."]

        for item in evidence:
            evidence_id = item.get("id")
            if evidence_id and evidence_id not in state.evidence_ids:
                state.evidence_ids.append(str(evidence_id))

        result = {
            "task": "drug_analysis",
            "input_type": parsed["input_type"],
            "normalized_entity": normalized_entity,
            "summary": _synthesize_summary(
                normalized_name=normalized_name,
                input_type=parsed["input_type"],
                profile=profile,
                biokb_context=biokb_context,
                evidence_count=len(evidence),
            ),
            "mechanism": _synthesize_mechanism(
                normalized_name=normalized_name,
                profile=profile,
                biokb_context=biokb_context,
            ),
            "targets": targets,
            "indications": indications,
            "safety": safety,
            "evidence": evidence,
            "next_questions": _build_next_questions(
                normalized_name=normalized_name,
                normalized_entity=normalized_entity,
                profile=profile,
                biokb_context=biokb_context,
            ),
        }

        state.artifacts.append(
            Artifact(
                type="drug_analysis",
                name=f"{_slugify(normalized_name or normalized_entity.get('raw_input') or 'drug')}_analysis",
                content=result,
                metadata={
                    "skill": self.name,
                    "input_type": parsed["input_type"],
                    "normalized_name": normalized_name,
                    "workspace_dir": workspace_dir,
                    "evidence_count": len(evidence),
                    "biokb_status": biokb_context.get("status"),
                },
            )
        )
        return result

    def _normalize_input(self, state: SessionState) -> Dict[str, Any]:
        query = state.user_query
        raw_smiles = _extract_smiles_candidate(query)
        if raw_smiles:
            canonical = canonicalize_smiles(raw_smiles)
            matched_drug_name = lookup_drug_name_by_smiles(canonical.get("canonical_smiles") or raw_smiles)
            normalized_name = matched_drug_name if matched_drug_name != "unknown" else None
            return {
                "input_type": "smiles",
                "normalized_drug_name": normalized_name,
                "normalized_entity": {
                    "raw_input": raw_smiles,
                    "name": normalized_name,
                    "canonical_smiles": canonical.get("canonical_smiles") or raw_smiles,
                    "inchikey": canonical.get("inchikey"),
                    "normalization_status": (
                        "matched_seeded_drug_from_smiles" if normalized_name else "canonicalized_smiles_only"
                    ),
                },
            }

        normalized_name = _extract_drug_name_candidate(state)
        seeded_smiles = lookup_seeded_smiles_for_drug(normalized_name) if normalized_name != "unknown" else None
        canonical = canonicalize_smiles(seeded_smiles) if seeded_smiles else {"canonical_smiles": None, "inchikey": None}
        return {
            "input_type": "drug_name",
            "normalized_drug_name": normalized_name if normalized_name != "unknown" else None,
            "normalized_entity": {
                "raw_input": normalized_name if normalized_name != "unknown" else state.user_query,
                "name": normalized_name if normalized_name != "unknown" else None,
                "canonical_smiles": canonical.get("canonical_smiles"),
                "inchikey": canonical.get("inchikey"),
                "normalization_status": (
                    "canonical_drug_name" if normalized_name != "unknown" else "unresolved_drug_name"
                ),
            },
        }
