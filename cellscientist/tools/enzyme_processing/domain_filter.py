from __future__ import annotations

from typing import Dict, List


_DOMAIN_KEYWORDS = [
    "GMC_oxred_C",
    "GMC_oxred_N",
    "BBE",
    "FAD",
]


def build_domain_filtering_steps(*, raw_sequence_count: int, unique_sequence_count: int) -> List[Dict[str, object]]:
    return [
        {
            "step": "merge_fasta_pool",
            "status": "bundle_profiled",
            "details": {
                "raw_sequence_count": raw_sequence_count,
                "goal": "merge multi-source FASTA files into one broad candidate pool",
            },
        },
        {
            "step": "exact_sequence_dedupe",
            "status": "profiled_from_local_bundle",
            "details": {
                "input_sequence_count": raw_sequence_count,
                "output_sequence_count": unique_sequence_count,
                "dedupe_rule": "exact_amino_acid_identity_keep_first",
            },
        },
        {
            "step": "domainhits_repair_and_merge",
            "status": "bridge_ready_not_executed",
            "details": {
                "expected_columns": 11,
                "source": "NCBI DomainHits_rep.txt outputs",
                "goal": "repair broken rows and merge all domain-hit tables",
            },
        },
        {
            "step": "domain_keyword_filter",
            "status": "bridge_ready_not_executed",
            "details": {
                "exclude_incomplete_flags": ["N", "C"],
                "short_name_keywords": list(_DOMAIN_KEYWORDS),
            },
        },
        {
            "step": "intersection_keep",
            "status": "bridge_ready_not_executed",
            "details": {
                "summary_key_source": "summary.txt FASTA-style headers",
                "goal": "retain only deduped sequences whose IDs survive domain shortlist filtering",
            },
        },
        {
            "step": "final_nonredundant_export",
            "status": "pending_after_domain_filter",
            "details": {
                "goal": "write final_unique_sequences.fasta after intersection and second exact dedupe",
            },
        },
    ]
