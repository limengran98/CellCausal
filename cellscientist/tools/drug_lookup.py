from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from ..evidence.models import EvidenceItem
from .base import BaseTool

_CANONICAL_ALIASES = {
    "metformin": ("metformin", "glucophage", "二甲双胍"),
    "aspirin": ("aspirin", "acetylsalicylic acid", "阿司匹林"),
    "ibuprofen": ("ibuprofen", "布洛芬"),
}

_ALIAS_TO_CANONICAL = {
    alias.lower(): canonical
    for canonical, aliases in _CANONICAL_ALIASES.items()
    for alias in aliases
}


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


_LOCAL_DRUG_DB: Dict[str, Dict[str, Any]] = {
    "metformin": {
        "summary": (
            "Metformin is an oral biguanide used mainly for glycemic control in type 2 diabetes. "
            "This V1.1 local lookup captures its commonly discussed indirect target biology, core "
            "metabolic indications, and a small set of clinically important adverse effects."
        ),
        "targets": [
            "Mitochondrial respiratory-chain complex I (putative/indirect)",
            "AMPK signaling axis (downstream activation)",
            "Hepatic gluconeogenesis pathways",
        ],
        "indications": [
            "Type 2 diabetes mellitus",
            "Prediabetes in selected high-risk patients",
            "Polycystic ovary syndrome (common off-label use)",
        ],
        "adverse_effects": [
            "Nausea and diarrhea",
            "Abdominal discomfort",
            "Vitamin B12 deficiency with long-term use",
            "Rare lactic acidosis in susceptible patients",
        ],
        "evidence": [
            _evidence(
                "D-METF-1",
                "local-curated-drug-db",
                "Metformin is commonly described as reducing hepatic glucose output and improving insulin sensitivity, with complex I inhibition and downstream AMPK-linked signaling often cited as mechanistic anchors.",
                "Repo-local curated drug note: metformin/mechanism/v1.1",
                0.81,
                drug_name="metformin",
                category="mechanism",
            ),
            _evidence(
                "D-METF-2",
                "local-curated-drug-db",
                "Metformin is a standard structured answer for type 2 diabetes indications and is frequently discussed for prediabetes or PCOS in selected clinical contexts.",
                "Repo-local curated drug note: metformin/indications/v1.1",
                0.87,
                drug_name="metformin",
                category="indication",
            ),
            _evidence(
                "D-METF-3",
                "local-curated-drug-db",
                "The most typical adverse effects are gastrointestinal, while vitamin B12 deficiency and lactic acidosis are lower-frequency but clinically relevant safety considerations.",
                "Repo-local curated drug note: metformin/safety/v1.1",
                0.84,
                drug_name="metformin",
                category="safety",
            ),
        ],
    },
    "aspirin": {
        "summary": (
            "Aspirin is a salicylate used for analgesic, antipyretic, anti-inflammatory, and antiplatelet purposes. "
            "The local lookup focuses on cyclooxygenase biology, common pain or cardiovascular indications, and bleeding-related risks."
        ),
        "targets": [
            "PTGS1 / COX-1",
            "PTGS2 / COX-2",
            "Platelet thromboxane A2 signaling",
        ],
        "indications": [
            "Pain and fever relief",
            "Inflammatory symptom control",
            "Secondary cardiovascular prevention in antiplatelet use",
        ],
        "adverse_effects": [
            "Gastrointestinal irritation",
            "Bleeding risk",
            "Hypersensitivity or bronchospasm in susceptible patients",
        ],
        "evidence": [
            _evidence(
                "D-ASP-1",
                "local-curated-drug-db",
                "Aspirin irreversibly inhibits cyclooxygenase activity, which supports both prostaglandin-related symptom relief and platelet inhibition.",
                "Repo-local curated drug note: aspirin/mechanism/v1.1",
                0.9,
                drug_name="aspirin",
                category="mechanism",
            ),
            _evidence(
                "D-ASP-2",
                "local-curated-drug-db",
                "Common structured indications include pain and fever control, while antiplatelet use is paired with gastrointestinal and bleeding risks.",
                "Repo-local curated drug note: aspirin/indications-safety/v1.1",
                0.88,
                drug_name="aspirin",
                category="indication_safety",
            ),
        ],
    },
    "ibuprofen": {
        "summary": (
            "Ibuprofen is a nonsteroidal anti-inflammatory drug used for pain, fever, and inflammatory symptoms. "
            "The local lookup summarizes COX inhibition, routine symptomatic indications, and common GI or renal safety concerns."
        ),
        "targets": [
            "PTGS1 / COX-1",
            "PTGS2 / COX-2",
        ],
        "indications": [
            "Mild to moderate pain",
            "Fever reduction",
            "Inflammatory symptoms such as musculoskeletal pain or dysmenorrhea",
        ],
        "adverse_effects": [
            "Dyspepsia or gastrointestinal discomfort",
            "Renal function worsening in susceptible patients",
            "Fluid retention or blood pressure elevation",
        ],
        "evidence": [
            _evidence(
                "D-IBU-1",
                "local-curated-drug-db",
                "Ibuprofen is a reversible COX inhibitor that is commonly structured under analgesic, antipyretic, and anti-inflammatory use cases.",
                "Repo-local curated drug note: ibuprofen/mechanism/v1.1",
                0.89,
                drug_name="ibuprofen",
                category="mechanism",
            ),
            _evidence(
                "D-IBU-2",
                "local-curated-drug-db",
                "Typical adverse-effect summaries emphasize gastrointestinal intolerance and renal risk in susceptible settings or prolonged high-dose use.",
                "Repo-local curated drug note: ibuprofen/safety/v1.1",
                0.85,
                drug_name="ibuprofen",
                category="safety",
            ),
        ],
    },
}


def canonicalize_drug_name(drug_name: str) -> str:
    normalized = drug_name.strip().lower()
    if not normalized:
        return "unknown"
    return _ALIAS_TO_CANONICAL.get(normalized, normalized)


def find_drug_name_in_text(text: str) -> str:
    lowered = text.strip().lower()
    if not lowered:
        return "unknown"

    for alias in sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True):
        if alias in lowered:
            return _ALIAS_TO_CANONICAL[alias]
    return "unknown"


class DrugLookupTool(BaseTool):
    """Minimal repo-local drug lookup for the V1.1 drug-info skill."""

    name = "drug-lookup"

    def run(self, drug_name: str) -> Dict[str, Any]:
        canonical_name = canonicalize_drug_name(drug_name)
        profile = _LOCAL_DRUG_DB.get(canonical_name)

        if profile is None:
            return {
                "drug_name": canonical_name,
                "summary": (
                    f"No curated local drug profile is available for '{drug_name}'. "
                    "The V1.1 lookup currently covers a very small seeded set of examples."
                ),
                "targets": [],
                "indications": [],
                "adverse_effects": [],
                "evidence": [
                    _evidence(
                        "D-LOOKUP-0",
                        "local-curated-drug-db",
                        f"The local V1.1 drug lookup does not yet contain a curated profile for '{drug_name}'.",
                        "Repo-local curated drug note: coverage/fallback/v1.1",
                        0.25,
                        drug_name=canonical_name,
                        category="coverage",
                    )
                ],
            }

        return {
            "drug_name": canonical_name,
            "summary": profile["summary"],
            "targets": list(profile["targets"]),
            "indications": list(profile["indications"]),
            "adverse_effects": list(profile["adverse_effects"]),
            "evidence": [dict(item) for item in profile["evidence"]],
        }


def lookup_drug_profile(drug_name: str) -> Dict[str, Any]:
    """Convenience entrypoint for minimal structured drug lookup."""

    return DrugLookupTool().run(drug_name)
