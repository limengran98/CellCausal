# -*- coding: utf-8 -*-
"""BioKB Module: SMILES → Targets → Pathways → Processes

This module implements the Bio Knowledge Base semantic enrichment pipeline
with a modular architecture inspired by Biomni.

Public API
----------
- generate_biokb_semantic_table(cfg, stage, log) -> Dict[str, Any]
- persist_biokb_semantic_table(semantic_table, workspace_dir, log) -> None
- biokb_table_to_evidence_items(semantic_table) -> List[Dict]

Architecture
------------
The module is organized into focused submodules:
- config.py: Configuration dataclass
- utils.py: Utility functions and decorators
- data_lake.py: Static keyword mappings
- tool_schemas.py: Tool metadata definitions
- registry.py: Knowledge source registry
- smiles_resolver.py: SMILES extraction and canonicalization
- chembl_client.py: ChEMBL API client
- reactome_client.py: Reactome API client
- process_mapper.py: Pathway → process mapping
- evidence_builder.py: Semantic table generation
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

from .config import BioKBConfig
from .evidence_builder import build_semantic_table, semantic_table_to_evidence_items
from .smiles_resolver import extract_smiles
from .utils import ensure_dir


def generate_biokb_semantic_table(
    cfg: Dict[str, Any],
    stage: str,
    log: Callable[[str], None]
) -> Dict[str, Any]:
    """Generate BioKB semantic table for a pipeline stage.
    
    This is the main entry point for generating biological knowledge base
    enrichment. It extracts SMILES from configuration, queries external APIs
    (ChEMBL, Reactome), and maps pathways to biological processes.
    
    Args:
        cfg: Configuration dictionary (expects cfg["literature"]["bio_kb"])
        stage: Pipeline stage ("design" or "review")
        log: Logging function
        
    Returns:
        Semantic table dictionary with structure:
        {
            "stage": str,
            "generated_at": str,
            "smiles_source": str,
            "molecules": [...],
            "summary": {...}
        }
        
    Example:
        >>> semantic_table = generate_biokb_semantic_table(cfg, "design", print)
        >>> print(f"Generated {len(semantic_table['molecules'])} entries")
    """
    log("[BIOKB] 🧬 Generating BioKB semantic table...")
    
    try:
        # Load configuration
        config = BioKBConfig.from_cfg(cfg)
        
        # Check if BioKB is enabled
        if not config.enabled:
            log("[BIOKB] ⚠️  BioKB disabled in config, returning empty table")
            return {
                "stage": stage,
                "generated_at": "",
                "smiles_source": "disabled",
                "molecules": [],
                "summary": {"total_molecules": 0, "error": "BioKB disabled"}
            }
        
        # Extract SMILES
        smiles_list, source = extract_smiles(cfg, log)
        
        # Build semantic table
        semantic_table = build_semantic_table(smiles_list, config, stage, log)
        semantic_table["smiles_source"] = source
        
        return semantic_table
        
    except Exception as e:
        log(f"[BIOKB][ERROR] Failed to generate semantic table: {e}")
        # Return minimal valid table on error
        return {
            "stage": stage,
            "generated_at": "",
            "smiles_source": "error",
            "molecules": [],
            "summary": {"total_molecules": 0, "error": str(e)}
        }


def persist_biokb_semantic_table(
    semantic_table: Dict[str, Any],
    workspace_dir: str,
    log: Callable[[str], None]
) -> None:
    """Persist semantic table to workspace directory.
    
    Saves the semantic table as JSON to:
    workspace_dir/external_knowledge/biokb_semantic_table_{stage}.json
    
    Args:
        semantic_table: Semantic table from generate_biokb_semantic_table()
        workspace_dir: Workspace directory path
        log: Logging function
        
    Example:
        >>> persist_biokb_semantic_table(table, "./workspace", print)
        [BIOKB] 💾 Saved semantic table: ./workspace/external_knowledge/biokb_semantic_table_design.json
    """
    if not workspace_dir:
        log("[BIOKB][WARN] No workspace directory provided, skipping persistence")
        return
    
    try:
        # Create external_knowledge directory
        ext_kb_dir = os.path.join(workspace_dir, "external_knowledge")
        ensure_dir(ext_kb_dir)
        
        # Determine output filename
        stage = semantic_table.get("stage", "unknown")
        output_path = os.path.join(ext_kb_dir, f"biokb_semantic_table_{stage}.json")
        
        # Write JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(semantic_table, f, ensure_ascii=False, indent=2)
        
        log(f"[BIOKB] 💾 Saved semantic table: {output_path}")
        
    except Exception as e:
        log(f"[BIOKB][WARN] Failed to persist semantic table: {e}")


def biokb_table_to_evidence_items(semantic_table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert semantic table to list of evidence item dictionaries.
    
    Each molecule in the semantic table is converted to an evidence item
    with Evidence ID "B{idx}" (e.g., B1, B2, B3...).
    
    Args:
        semantic_table: Semantic table from generate_biokb_semantic_table()
        
    Returns:
        List of evidence item dictionaries with keys:
        - eid: Evidence ID (e.g., "B1")
        - title: Evidence title
        - url: URL (empty for BioKB)
        - snippet: Short summary
        - source: "biokb"
        - published: Timestamp
        - scraped_excerpt: Detailed information
        
    Example:
        >>> items = biokb_table_to_evidence_items(semantic_table)
        >>> for item in items:
        >>>     print(f"{item['eid']}: {item['title']}")
    """
    try:
        # Get max items from summary or default to 5
        max_items = semantic_table.get("summary", {}).get("total_molecules", 5)
        max_items = min(max_items, 10)  # Cap at 10 to avoid overwhelming output
        
        return semantic_table_to_evidence_items(semantic_table, max_items)
        
    except Exception as e:
        # Return error evidence item on failure
        return [{
            "eid": "B0",
            "title": "BioKB Evidence Generation Failed",
            "url": "",
            "snippet": f"Error converting semantic table to evidence: {str(e)[:200]}",
            "source": "biokb",
            "published": "",
            "scraped_excerpt": f"Full error: {str(e)}"
        }]


# Export public API
__all__ = [
    "generate_biokb_semantic_table",
    "persist_biokb_semantic_table",
    "biokb_table_to_evidence_items",
    "BioKBConfig"
]
