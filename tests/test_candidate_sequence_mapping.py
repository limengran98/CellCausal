from __future__ import annotations

from pathlib import Path

from cellscientist.tools.enzyme_lookup import lookup_enzyme_candidates
from cellscientist.tools.enzyme_processing.candidate_sequence_mapping import (
    build_candidate_sequence_rows,
)


def test_candidate_sequence_mapping_builds_real_sequence_rows_from_local_bundle():
    payload = lookup_enzyme_candidates("lipid_metabolism")
    status, rows = build_candidate_sequence_rows(
        candidate_enzymes=payload["candidate_enzymes"],
        query_focus=payload["query_focus"],
        zip_path=str(Path("references/enzyme_mining/output_sequences.zip")),
    )

    assert status["status"] == "candidate_sequence_rows_ready"
    assert status["candidate_sequence_row_count"] > 0
    assert rows
    assert rows[0]["enzyme_id"]
    assert rows[0]["sequence"]
    assert rows[0]["source"]
    assert rows[0]["matched_focus"] == payload["query_focus"]
