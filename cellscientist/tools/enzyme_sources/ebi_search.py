from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ...pipeline.utils import project_root


_EBI_DATABASES = [
    "UniRef100",
    "UniRef90",
    "UniRef50",
    "UniProtKB",
    "EPO",
    "JPO",
    "USPTO",
]


def summarize_ebi_candidate_source(query_focus: str) -> Dict[str, Any]:
    notebook_path = Path(project_root()) / "references" / "enzyme_mining" / "EBI.ipynb"
    workflow_notebook_path = Path(project_root()) / "references" / "enzyme_mining" / "挖酶.ipynb"
    return {
        "source_name": "ebi_uniprot_uniref",
        "status": "adapter_extracted_from_notebook",
        "query_focus": query_focus,
        "supports_text_query": True,
        "supported_databases": list(_EBI_DATABASES),
        "notebook_paths": [
            str(notebook_path),
            str(workflow_notebook_path),
        ],
        "retrieval_mode": "text_query_to_ids_then_parallel_fasta_download",
        "notes": [
            "The user notebook already includes ID caching, FASTA download, merge, and failed-ID retry loops.",
            "This source adapter is suitable for future conversion into a real query->FASTA bridge.",
        ],
    }
