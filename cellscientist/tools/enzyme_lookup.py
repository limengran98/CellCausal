from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from ..evidence.models import EvidenceItem
from .base import BaseTool


def _evidence(
    evidence_id: str,
    source: str,
    claim: str,
    citation: str,
    confidence: float,
    **metadata: Any,
) -> Dict[str, Any]:
    return asdict(
        EvidenceItem(
            id=evidence_id,
            source=source,
            claim=claim,
            citation=citation,
            confidence=confidence,
            metadata=metadata,
        )
    )


_FOCUS_ALIASES = {
    "lipid_metabolism": (
        "lipid metabolism",
        "lipid",
        "fat metabolism",
        "fatty acid",
        "脂代谢",
        "脂质代谢",
        "脂肪代谢",
    ),
    "cholesterol_metabolism": (
        "cholesterol metabolism",
        "cholesterol",
        "胆固醇代谢",
        "胆固醇",
    ),
    "fatty_acid_oxidation": (
        "fatty acid oxidation",
        "beta oxidation",
        "β-oxidation",
        "脂肪酸氧化",
    ),
}


_ENZYME_MINING_DB: Dict[str, Dict[str, Any]] = {
    "lipid_metabolism": {
        "query_focus": "lipid metabolism",
        "candidate_enzymes": [
            {
                "enzyme": "FASN",
                "role": "de novo fatty acid synthesis",
                "confidence": 0.84,
            },
            {
                "enzyme": "ACACA",
                "role": "acetyl-CoA to malonyl-CoA commitment step",
                "confidence": 0.82,
            },
            {
                "enzyme": "CPT1A",
                "role": "fatty acid mitochondrial import and beta-oxidation control",
                "confidence": 0.8,
            },
            {
                "enzyme": "SCD",
                "role": "fatty acid desaturation and lipid composition remodeling",
                "confidence": 0.77,
            },
        ],
        "pathway_context": [
            "de novo lipogenesis",
            "fatty acid beta-oxidation",
            "triglyceride and lipid droplet balance",
        ],
        "rationale": [
            "A useful first-pass lipid metabolism panel should balance synthesis (FASN, ACACA) against oxidation control (CPT1A).",
            "SCD is often informative when the biological question depends on unsaturated fatty acid remodeling rather than only total lipid flux.",
        ],
        "evidence": [
            _evidence(
                "E-LIP-1",
                "local-curated-enzyme-db",
                "FASN and ACACA are standard candidate enzymes when the goal is to capture de novo lipogenesis rather than only downstream lipid handling.",
                "Repo-local curated enzyme note: lipid_metabolism/core_lipogenesis/v1.0",
                0.83,
                focus="lipid_metabolism",
                category="candidate_prioritization",
            ),
            _evidence(
                "E-LIP-2",
                "local-curated-enzyme-db",
                "CPT1A is a common counterweight candidate because it anchors mitochondrial fatty acid entry and can separate synthesis-dominant from oxidation-dominant states.",
                "Repo-local curated enzyme note: lipid_metabolism/oxidation_gate/v1.0",
                0.81,
                focus="lipid_metabolism",
                category="pathway_context",
            ),
        ],
    },
    "cholesterol_metabolism": {
        "query_focus": "cholesterol metabolism",
        "candidate_enzymes": [
            {
                "enzyme": "HMGCR",
                "role": "rate-limiting cholesterol biosynthesis step",
                "confidence": 0.91,
            },
            {
                "enzyme": "SQLE",
                "role": "squalene epoxidation and sterol synthesis commitment",
                "confidence": 0.84,
            },
            {
                "enzyme": "CYP7A1",
                "role": "cholesterol catabolism toward bile acid synthesis",
                "confidence": 0.82,
            },
            {
                "enzyme": "SOAT2",
                "role": "cholesteryl esterification and storage handling",
                "confidence": 0.74,
            },
        ],
        "pathway_context": [
            "mevalonate pathway",
            "sterol biosynthesis",
            "bile acid synthesis",
            "cholesteryl ester storage",
        ],
        "rationale": [
            "HMGCR and SQLE cover the biosynthetic arm, while CYP7A1 captures a catabolic outlet that often changes the interpretation of cholesterol accumulation phenotypes.",
            "SOAT2 is useful when the biological readout is storage or packaging rather than total synthesis alone.",
        ],
        "evidence": [
            _evidence(
                "E-CHOL-1",
                "local-curated-enzyme-db",
                "HMGCR remains the most canonical entry point for cholesterol biosynthesis-focused enzyme prioritization.",
                "Repo-local curated enzyme note: cholesterol_metabolism/hmgcr_core/v1.0",
                0.9,
                focus="cholesterol_metabolism",
                category="candidate_prioritization",
            ),
            _evidence(
                "E-CHOL-2",
                "local-curated-enzyme-db",
                "Pairing HMGCR with SQLE and CYP7A1 gives a more balanced biosynthesis-versus-catabolism view than using a single enzyme marker.",
                "Repo-local curated enzyme note: cholesterol_metabolism/balance_panel/v1.0",
                0.85,
                focus="cholesterol_metabolism",
                category="pathway_context",
            ),
        ],
    },
    "fatty_acid_oxidation": {
        "query_focus": "fatty acid oxidation",
        "candidate_enzymes": [
            {
                "enzyme": "CPT1A",
                "role": "mitochondrial fatty acid import gate",
                "confidence": 0.9,
            },
            {
                "enzyme": "ACADM",
                "role": "medium-chain acyl-CoA dehydrogenation",
                "confidence": 0.79,
            },
            {
                "enzyme": "ACOX1",
                "role": "peroxisomal fatty acid oxidation entry",
                "confidence": 0.77,
            },
        ],
        "pathway_context": [
            "mitochondrial beta-oxidation",
            "peroxisomal fatty acid oxidation",
            "energy stress adaptation",
        ],
        "rationale": [
            "CPT1A distinguishes substrate entry control from downstream oxidation capacity.",
            "ACADM and ACOX1 broaden the panel across mitochondrial and peroxisomal oxidation compartments.",
        ],
        "evidence": [
            _evidence(
                "E-FAO-1",
                "local-curated-enzyme-db",
                "CPT1A is usually the first candidate when the question explicitly targets fatty acid oxidation control.",
                "Repo-local curated enzyme note: fatty_acid_oxidation/cpt1a_gate/v1.0",
                0.88,
                focus="fatty_acid_oxidation",
                category="candidate_prioritization",
            ),
        ],
    },
}


def normalize_enzyme_focus(query: str) -> dict[str, Any]:
    lowered = query.lower().strip()
    for canonical, aliases in _FOCUS_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias.lower() in lowered:
                return {
                    "query_focus": _ENZYME_MINING_DB[canonical]["query_focus"],
                    "focus_key": canonical,
                    "matched_alias": alias,
                    "normalization_status": "matched_seeded_focus",
                }
    return {
        "query_focus": "lipid metabolism",
        "focus_key": "lipid_metabolism",
        "matched_alias": None,
        "normalization_status": "defaulted_to_seeded_focus",
    }


def lookup_enzyme_candidates(focus_key: str) -> dict[str, Any]:
    base = _ENZYME_MINING_DB.get(focus_key)
    if base is None:
        base = _ENZYME_MINING_DB["lipid_metabolism"]
    return {
        "query_focus": base["query_focus"],
        "candidate_enzymes": list(base["candidate_enzymes"]),
        "rationale": list(base["rationale"]),
        "pathway_context": list(base["pathway_context"]),
        "evidence": list(base["evidence"]),
    }


class EnzymeLookupTool(BaseTool):
    name = "enzyme-lookup"

    def run(self, query: str) -> dict[str, Any]:
        normalized = normalize_enzyme_focus(query)
        payload = lookup_enzyme_candidates(str(normalized["focus_key"]))
        payload["normalized_focus"] = normalized
        return payload
