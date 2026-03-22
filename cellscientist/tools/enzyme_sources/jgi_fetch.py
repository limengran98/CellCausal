from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ...pipeline.utils import project_root


def summarize_jgi_candidate_source(query_focus: str) -> Dict[str, Any]:
    notebook_path = Path(project_root()) / "references" / "enzyme_mining" / "JGI.ipynb"
    workflow_notebook_path = Path(project_root()) / "references" / "enzyme_mining" / "挖酶.ipynb"
    return {
        "source_name": "jgi_gene_fetch",
        "status": "adapter_extracted_from_notebook",
        "query_focus": query_focus,
        "supports_text_query": False,
        "requires_gene_ids": True,
        "notebook_paths": [
            str(notebook_path),
            str(workflow_notebook_path),
        ],
        "retrieval_mode": "gene_id_to_fasta_fetch",
        "notes": [
            "The user notebook fetches FASTA records from JGI HTML pages via gene_oid and supports append/resume execution.",
            "This source is better modeled as an ID-based fetch adapter than as a generic text-search adapter.",
        ],
    }
