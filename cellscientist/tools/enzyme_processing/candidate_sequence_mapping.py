from __future__ import annotations

import hashlib
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .fasta_merge import _iter_fasta_records


def _header_id(header: str) -> str:
    return str(header).split()[0].strip()


@lru_cache(maxsize=4)
def _load_unique_bundle_rows(zip_path: str, max_rows: int = 64) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    path = Path(zip_path)
    if not path.exists():
        return (
            {
                "status": "missing_sequence_bundle",
                "bundle_path": str(path),
                "loaded_unique_rows": 0,
            },
            [],
        )

    seen_hashes: set[str] = set()
    rows: List[Dict[str, Any]] = []
    scanned_records = 0

    with zipfile.ZipFile(path) as archive:
        for member_name in archive.namelist():
            if not member_name.endswith(".fasta"):
                continue
            lines = archive.read(member_name).decode("utf-8", errors="ignore").splitlines()
            for header, sequence in _iter_fasta_records(lines):
                scanned_records += 1
                digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                rows.append(
                    {
                        "enzyme_id": _header_id(header),
                        "header": header,
                        "sequence": sequence,
                        "source": f"{path.name}:{member_name}",
                    }
                )
                if len(rows) >= max_rows:
                    return (
                        {
                            "status": "bundle_unique_rows_loaded",
                            "bundle_path": str(path),
                            "loaded_unique_rows": len(rows),
                            "scanned_records": scanned_records,
                        },
                        rows,
                    )

    return (
        {
            "status": "bundle_unique_rows_loaded",
            "bundle_path": str(path),
            "loaded_unique_rows": len(rows),
            "scanned_records": scanned_records,
        },
        rows,
    )


def build_candidate_sequence_rows(
    *,
    candidate_enzymes: Sequence[dict[str, Any]],
    query_focus: str,
    zip_path: str,
    max_rows: int | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidate_names = [str(item.get("enzyme") or "").strip() for item in candidate_enzymes if str(item.get("enzyme") or "").strip()]
    target_rows = max(max_rows or len(candidate_names) or 4, 4)
    pool_meta, pool_rows = _load_unique_bundle_rows(zip_path, max(target_rows * 4, 16))

    if not pool_rows:
        return (
            {
                "status": "candidate_sequence_rows_unavailable",
                "mapping_mode": "bundle_missing",
                "candidate_count": len(candidate_enzymes),
                "candidate_sequence_row_count": 0,
                "exact_name_match_count": 0,
                "proxy_row_count": 0,
                "bundle_path": str(zip_path),
                "notes": [
                    "No local sequence bundle rows could be loaded for candidate sequence mapping.",
                ],
            },
            [],
        )

    exact_rows: List[Dict[str, Any]] = []
    candidate_name_lookup = {name.lower(): name for name in candidate_names}
    for row in pool_rows:
        lowered_header = str(row.get("header") or "").lower()
        lowered_id = str(row.get("enzyme_id") or "").lower()
        matched_name = None
        for lowered_name, original_name in candidate_name_lookup.items():
            if lowered_name and (lowered_name in lowered_header or lowered_name == lowered_id):
                matched_name = original_name
                break
        if matched_name is None:
            continue
        exact_rows.append(
            {
                **row,
                "matched_candidate": matched_name,
                "matched_focus": query_focus,
                "mapping_quality": "exact_header_match",
            }
        )
        if len(exact_rows) >= target_rows:
            break

    if exact_rows:
        return (
            {
                "status": "candidate_sequence_rows_ready",
                "mapping_mode": "exact_bundle_header_match",
                "candidate_count": len(candidate_enzymes),
                "candidate_sequence_row_count": len(exact_rows),
                "exact_name_match_count": len(exact_rows),
                "proxy_row_count": 0,
                "bundle_path": str(zip_path),
                "bundle_unique_rows_considered": pool_meta.get("loaded_unique_rows"),
                "notes": [
                    "Candidate sequence rows were constructed by exact matches between candidate names and bundled FASTA headers.",
                ],
            },
            exact_rows,
        )

    proxy_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(pool_rows[:target_rows]):
        hint = dict(candidate_enzymes[index % len(candidate_enzymes)]) if candidate_enzymes else {}
        proxy_rows.append(
            {
                **row,
                "matched_candidate": hint.get("enzyme"),
                "role": hint.get("role"),
                "confidence": hint.get("confidence"),
                "matched_focus": query_focus,
                "mapping_quality": "focus_proxy_sequence_row",
            }
        )

    return (
        {
            "status": "candidate_sequence_rows_ready",
            "mapping_mode": "focus_proxy_sequence_rows",
            "candidate_count": len(candidate_enzymes),
            "candidate_sequence_row_count": len(proxy_rows),
            "exact_name_match_count": 0,
            "proxy_row_count": len(proxy_rows),
            "bundle_path": str(zip_path),
            "bundle_unique_rows_considered": pool_meta.get("loaded_unique_rows"),
            "notes": [
                "No exact gene-symbol/header match was found between the current seeded candidate panel and the local sequence bundle.",
                "The current MVP therefore uses real sequence rows from the local bundle as focus-aware proxy candidates so ranking input completeness can be tested without pretending exact identity mapping.",
            ],
        },
        proxy_rows,
    )
