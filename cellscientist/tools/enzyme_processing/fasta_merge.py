from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


def _iter_fasta_records(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    header: str | None = None
    seq_chunks: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_chunks)
            header = line[1:].strip()
            seq_chunks = []
            continue
        seq_chunks.append(line)
    if header is not None:
        yield header, "".join(seq_chunks)


@lru_cache(maxsize=8)
def summarize_local_sequence_bundle(zip_path: str) -> Dict[str, object]:
    path = Path(zip_path)
    if not path.exists():
        return {
            "status": "missing_sequence_bundle",
            "bundle_path": str(path),
            "split_fasta_files": 0,
            "raw_sequence_count": 0,
            "example_headers": [],
        }

    split_fasta_files = 0
    raw_sequence_count = 0
    example_headers: list[str] = []

    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".fasta"):
                continue
            split_fasta_files += 1
            lines = archive.read(name).decode("utf-8", errors="ignore").splitlines()
            for header, _seq in _iter_fasta_records(lines):
                raw_sequence_count += 1
                if len(example_headers) < 5:
                    example_headers.append(header)

    return {
        "status": "local_sequence_bundle_profiled",
        "bundle_path": str(path),
        "split_fasta_files": split_fasta_files,
        "raw_sequence_count": raw_sequence_count,
        "example_headers": example_headers,
        "merge_ready": split_fasta_files > 0,
        "notes": [
            "This summary profiles the bundled FASTA archive without assuming any BBBC036-specific runtime path.",
            "The original user workflow merges many FASTA files into one broad candidate pool before filtering.",
        ],
    }
