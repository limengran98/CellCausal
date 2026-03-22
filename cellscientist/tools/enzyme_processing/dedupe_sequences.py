from __future__ import annotations

import hashlib
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict

from .fasta_merge import _iter_fasta_records, summarize_local_sequence_bundle


@lru_cache(maxsize=8)
def summarize_exact_dedupe_from_zip(zip_path: str) -> Dict[str, object]:
    path = Path(zip_path)
    bundle_summary = summarize_local_sequence_bundle(str(path))
    if not path.exists():
        return {
            "status": "missing_sequence_bundle",
            "raw_sequence_count": 0,
            "unique_sequence_count": 0,
            "duplicate_sequence_count": 0,
            "dedupe_strategy": "exact_amino_acid_identity",
        }

    seen_hashes: set[str] = set()
    raw_sequence_count = 0
    unique_sequence_count = 0

    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".fasta"):
                continue
            lines = archive.read(name).decode("utf-8", errors="ignore").splitlines()
            for _header, seq in _iter_fasta_records(lines):
                raw_sequence_count += 1
                digest = hashlib.sha1(seq.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                unique_sequence_count += 1

    return {
        "status": "exact_dedupe_profiled",
        "bundle_path": str(path),
        "raw_sequence_count": raw_sequence_count,
        "unique_sequence_count": unique_sequence_count,
        "duplicate_sequence_count": raw_sequence_count - unique_sequence_count,
        "dedupe_strategy": "exact_amino_acid_identity_keep_first",
        "example_headers": list(bundle_summary.get("example_headers") or []),
        "notes": [
            "This step preserves the notebook's exact-sequence dedupe rule.",
            "It corresponds to the notebook's initial and final nonredundancy checks.",
        ],
    }
