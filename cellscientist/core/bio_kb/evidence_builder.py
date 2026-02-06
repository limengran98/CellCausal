# -*- coding: utf-8 -*-
"""Evidence Builder Module.

This module orchestrates the semantic table generation and
conversion to evidence items.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .chembl_client import query_targets_by_smiles
from .config import BioKBConfig
from .data_lake import DataLake
from .process_mapper import map_pathways_to_processes
from .reactome_client import query_pathways_by_gene
from .smiles_resolver import canonicalize_smiles, extract_smiles
from .utils import now_iso


def build_semantic_table(
    smiles_list: List[str],
    config: BioKBConfig,
    stage: str,
    log: Callable[[str], None]
) -> Dict[str, Any]:
    """Build semantic table linking SMILES to targets, pathways, and processes.
    
    Args:
        smiles_list: List of SMILES strings to process
        config: BioKB configuration
        stage: Pipeline stage ("design" or "review")
        log: Logging function
        
    Returns:
        Semantic table dictionary with structure:
        {
            "stage": str,
            "generated_at": str,
            "smiles_source": str,
            "molecules": [
                {
                    "smiles": str,
                    "canonical_smiles": str,
                    "inchikey": str,
                    "targets": [...],
                    "pathways": [...],
                    "inferred_processes": [...]
                }
            ],
            "summary": {...}
        }
    """
    log(f"[BIOKB] 🧬 Building semantic table for stage={stage}")
    
    data_lake = DataLake()
    molecules = []
    process_distribution: Dict[str, int] = {}
    
    # Limit number of SMILES processed
    max_smiles = min(len(smiles_list), config.max_smiles)
    if len(smiles_list) > max_smiles:
        log(f"[BIOKB] Limiting SMILES processing: {len(smiles_list)} -> {max_smiles}")
    
    for idx, smiles in enumerate(smiles_list[:max_smiles], 1):
        if not smiles or smiles == "SMILES_NOT_FOUND":
            # Add placeholder entry
            molecules.append({
                "smiles": smiles,
                "canonical_smiles": "",
                "inchikey": "",
                "targets": [],
                "pathways": [],
                "inferred_processes": [
                    {"process": "UNKNOWN_PROCESS", "confidence": 0.0, "matched_keywords": []}
                ]
            })
            continue
        
        log(f"[BIOKB] 🔬 Processing molecule {idx}/{max_smiles}: {smiles[:30]}...")
        
        # Step 1: Canonicalize SMILES
        canon_result = canonicalize_smiles(smiles)
        canonical = canon_result["canonical_smiles"]
        inchikey = canon_result["inchikey"]
        
        # Step 2: Query ChEMBL for targets
        targets = query_targets_by_smiles(smiles, inchikey or "", config, log)
        
        # Step 3: Query Reactome for pathways (for each target's gene)
        all_pathways = []
        for target in targets[:3]:  # Limit to top 3 targets to avoid rate limits
            gene_symbol = target.get("gene_symbol", "")
            if gene_symbol:
                pathways = query_pathways_by_gene(gene_symbol, config, log)
                all_pathways.extend(pathways)
        
        # Deduplicate pathways by ID
        unique_pathways = {}
        for p in all_pathways:
            pid = p.get("pathway_id", "")
            if pid and pid not in unique_pathways:
                unique_pathways[pid] = p
        pathway_list = list(unique_pathways.values())
        
        # Step 4: Map pathways to processes
        inferred_processes = map_pathways_to_processes(pathway_list, data_lake)
        
        # Update process distribution
        for proc in inferred_processes:
            process_name = proc["process"]
            process_distribution[process_name] = process_distribution.get(process_name, 0) + 1
        
        molecules.append({
            "smiles": smiles,
            "canonical_smiles": canonical,
            "inchikey": inchikey or "",
            "targets": targets,
            "pathways": pathway_list,
            "inferred_processes": inferred_processes
        })
    
    # Build summary
    summary = {
        "total_molecules": len(molecules),
        "total_targets": sum(len(m["targets"]) for m in molecules),
        "total_pathways": sum(len(m["pathways"]) for m in molecules),
        "process_distribution": process_distribution
    }
    
    semantic_table = {
        "stage": stage,
        "generated_at": now_iso(),
        "smiles_source": "config",  # Will be updated by caller if needed
        "molecules": molecules,
        "summary": summary
    }
    
    log(f"[BIOKB] ✅ Semantic table complete: {len(molecules)} molecules, {summary['total_targets']} targets")
    
    return semantic_table


def semantic_table_to_evidence_items(
    semantic_table: Dict[str, Any],
    max_items: int
) -> List[Dict[str, Any]]:
    """Convert semantic table to evidence item dictionaries.
    
    Args:
        semantic_table: Semantic table from build_semantic_table()
        max_items: Maximum number of evidence items to generate
        
    Returns:
        List of evidence item dictionaries with 'eid' field
    """
    molecules = semantic_table.get("molecules", [])
    items = []
    
    for idx, mol in enumerate(molecules[:max_items], 1):
        eid = f"B{idx}"
        smiles = mol.get("smiles", "")
        
        # Extract key info
        targets = mol.get("targets", [])
        pathways = mol.get("pathways", [])
        processes = mol.get("inferred_processes", [])
        
        # Build snippet
        target_names = [t.get("target_name", "") for t in targets[:3]]
        pathway_names = [p.get("pathway_name", "") for p in pathways[:3]]
        process_names = [p.get("process", "") for p in processes]
        
        snippet = (
            f"BioKB semantic mapping for molecular perturbation.\n"
            f"Predicted Targets: {', '.join(target_names) or 'None'}\n"
            f"Associated Pathways: {', '.join(pathway_names) or 'None'}\n"
            f"Biological Processes: {', '.join(process_names) or 'Unknown'}"
        )
        
        # Build detailed excerpt
        target_lines = [f"- {t.get('target_name', '')} ({t.get('gene_symbol', '')}) - {t.get('mechanism', '')}" 
                       for t in targets[:10]]
        pathway_lines = [f"- {p.get('pathway_name', '')} ({p.get('pathway_id', '')})" 
                        for p in pathways[:10]]
        process_lines = [f"- {p.get('process', '')} (confidence: {p.get('confidence', 0)})" 
                        for p in processes]
        
        scraped_excerpt = (
            f"**SMILES**: {smiles}\n"
            f"**Canonical SMILES**: {mol.get('canonical_smiles', 'N/A')}\n"
            f"**InChIKey**: {mol.get('inchikey', 'N/A')}\n\n"
            f"**Predicted Targets** ({len(targets)}):\n"
            f"{chr(10).join(target_lines) if target_lines else 'None found'}\n\n"
            f"**Associated Pathways** ({len(pathways)}):\n"
            f"{chr(10).join(pathway_lines) if pathway_lines else 'None found'}\n\n"
            f"**Biological Processes**:\n"
            f"{chr(10).join(process_lines) if process_lines else 'Unknown'}\n"
        )
        
        # Generate title
        if target_names:
            title = f"BioKB: {smiles[:40]} → {target_names[0]}"
        else:
            title = f"BioKB: {smiles[:40]}"
        
        item = {
            "eid": eid,
            "title": title,
            "url": "",
            "snippet": snippet,
            "source": "biokb",
            "published": semantic_table.get("generated_at", ""),
            "scraped_excerpt": scraped_excerpt
        }
        items.append(item)
    
    return items
