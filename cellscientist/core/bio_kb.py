# -*- coding: utf-8 -*-
"""BioKB Module: SMILES → Targets → Pathways → Processes

This module implements the Bio Knowledge Base semantic enrichment pipeline:
1. Extract SMILES from config/H5/env
2. Map SMILES to targets via ChEMBL (best-effort + cached)
3. Map targets to pathways via Reactome (best-effort + cached)
4. Map pathways to biological processes (heuristic)
5. Generate structured semantic table and evidence items
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

import requests


# Module-level constants
MAX_MECHANISMS_PER_MOLECULE = 5  # Limit mechanisms to avoid excessive API calls
MAX_TARGETS_FOR_PATHWAYS = 3  # Limit targets queried for pathways to manage rate limits
MAX_PATHWAYS_PER_TARGET = 5  # Limit pathways retrieved per target


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class BioKBSemanticEntry:
    """A single semantic table entry linking SMILES to biological context"""
    smiles: str
    canonical_smiles: str = ""
    inchikey: str = ""
    targets: Optional[List[str]] = None
    pathways: Optional[List[str]] = None
    processes: Optional[List[str]] = None
    source: str = "biokb"
    
    def __post_init__(self):
        if self.targets is None:
            self.targets = []
        if self.pathways is None:
            self.pathways = []
        if self.processes is None:
            self.processes = []


@dataclass
class BioKBSemanticTable:
    """Collection of semantic entries for a stage"""
    stage: str
    generated_at: str
    entries: List[BioKBSemanticEntry]
    smiles_source: str = "unknown"
    

# -----------------------------
# Utilities
# -----------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# SMILES Sourcing
# -----------------------------

def _extract_smiles_from_config(cfg: Dict[str, Any]) -> tuple[List[str], str]:
    """Extract SMILES from config['literature']['bio_kb']['smiles_list']"""
    lit = cfg.get("literature", {}) if isinstance(cfg, dict) else {}
    bio_kb_cfg = lit.get("bio_kb", {}) if isinstance(lit, dict) else {}
    smiles_list = bio_kb_cfg.get("smiles_list", []) if isinstance(bio_kb_cfg, dict) else []
    
    if isinstance(smiles_list, list) and smiles_list:
        return [str(s).strip() for s in smiles_list if s], "config"
    return [], "none"


def _extract_smiles_from_h5(h5_path: str) -> tuple[List[str], str]:
    """Extract SMILES from H5 file (best-effort, supports pandas HDFStore and h5py)"""
    if not h5_path or not os.path.exists(h5_path):
        return [], "none"
    
    smiles = []
    
    # Try pandas HDFStore first
    try:
        import pandas as pd
        with pd.HDFStore(h5_path, "r") as store:
            # Look for common SMILES column names in various keys
            for key in store.keys():
                try:
                    df = store[key]
                    for col in ["smiles", "SMILES", "canonical_smiles", "mol_smiles"]:
                        if col in df.columns:
                            smiles.extend(df[col].dropna().unique().tolist())
                            if smiles:
                                return [str(s).strip() for s in smiles if s], f"h5_pandas:{key}"
                except Exception:
                    continue
    except Exception:
        pass
    
    # Try h5py if pandas failed
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            def search_smiles(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Check if dataset name suggests SMILES
                    if "smiles" in name.lower():
                        try:
                            data = obj[:]
                            if data.dtype.kind in ['S', 'O', 'U']:  # string types
                                smiles.extend([str(s).strip() for s in data if s])
                        except Exception:
                            pass
            
            f.visititems(search_smiles)
            if smiles:
                return list(set(smiles)), "h5_h5py"
    except Exception:
        pass
    
    return [], "none"


def extract_smiles(cfg: Dict[str, Any]) -> tuple[List[str], str]:
    """Extract SMILES with priority: config > H5 > placeholder
    
    Returns:
        (smiles_list, source_description)
    """
    # Priority 1: Config
    smiles, source = _extract_smiles_from_config(cfg)
    if smiles:
        return smiles, source
    
    # Priority 2: H5 from env or config
    h5_path = os.environ.get("STAGE1_H5_PATH", "").strip()
    if not h5_path:
        paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
        h5_path = paths.get("stage1_h5", "") if isinstance(paths, dict) else ""
    
    if h5_path:
        smiles, source = _extract_smiles_from_h5(h5_path)
        if smiles:
            return smiles, source
    
    # Priority 3: Return visible placeholder (no silent fail)
    return ["SMILES_NOT_FOUND"], "placeholder_missing"


# -----------------------------
# SMILES Canonicalization
# -----------------------------

def canonicalize_smiles(smiles: str) -> tuple[str, str]:
    """Generate canonical SMILES and InChIKey using RDKit if available
    
    Returns:
        (canonical_smiles, inchikey)
    """
    # Try RDKit first
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            return canonical, inchikey
    except Exception:
        pass
    
    # Fallback: return original SMILES, try online service for InChIKey (best-effort)
    # For now, just return empty to avoid external dependencies
    return smiles, ""


# -----------------------------
# ChEMBL API Integration
# -----------------------------

def _query_chembl_targets(smiles: str, inchikey: str, cache_dir: str, log: Callable[[str], None]) -> List[str]:
    """Query ChEMBL API to find targets for a molecule (best-effort + cached)"""
    cache_key = _sha1(f"chembl_targets:{smiles}:{inchikey}")
    cache_path = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else ""
    
    # Check cache
    if cache_path and os.path.exists(cache_path):
        try:
            cached = _read_json(cache_path)
            return cached.get("targets", [])
        except Exception:
            pass
    
    targets: List[str] = []
    
    # Try ChEMBL API search by SMILES or InChIKey
    try:
        base_url = "https://www.ebi.ac.uk/chembl/api/data"
        
        # Try by InChIKey first (more reliable)
        if inchikey:
            url = f"{base_url}/molecule.json?molecule_structures__standard_inchi_key={inchikey}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                molecules = data.get("molecules", [])
                if molecules:
                    molecule_id = molecules[0].get("molecule_chembl_id")
                    if molecule_id:
                        # Get activities/mechanisms
                        mech_url = f"{base_url}/mechanism.json?molecule_chembl_id={molecule_id}"
                        mech_resp = requests.get(mech_url, timeout=10)
                        if mech_resp.status_code == 200:
                            mech_data = mech_resp.json()
                            all_mechanisms = mech_data.get("mechanisms", [])
                            if len(all_mechanisms) > MAX_MECHANISMS_PER_MOLECULE:
                                log(f"[BIOKB] Truncating mechanisms: {len(all_mechanisms)} -> {MAX_MECHANISMS_PER_MOLECULE}")
                            for mech in all_mechanisms[:MAX_MECHANISMS_PER_MOLECULE]:
                                target_name = mech.get("target_chembl_id")
                                if target_name:
                                    targets.append(target_name)
        
        # Cache result
        if cache_path:
            _write_json(cache_path, {"targets": targets})
        
    except Exception as e:
        log(f"[BIOKB][WARN] ChEMBL query failed for {smiles[:20]}: {e}")
    
    return targets


# -----------------------------
# Reactome API Integration
# -----------------------------

def _query_reactome_pathways(targets: List[str], cache_dir: str, log: Callable[[str], None]) -> List[str]:
    """Query Reactome API to find pathways for targets (best-effort + cached)"""
    if not targets:
        return []
    
    cache_key = _sha1(f"reactome_pathways:{','.join(sorted(targets))}")
    cache_path = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else ""
    
    # Check cache
    if cache_path and os.path.exists(cache_path):
        try:
            cached = _read_json(cache_path)
            return cached.get("pathways", [])
        except Exception:
            pass
    
    pathways: Set[str] = set()
    
    # Try Reactome API
    try:
        # Reactome content service for pathway search
        base_url = "https://reactome.org/ContentService"
        
        # For each target, query pathways (limit queries)
        for target in targets[:MAX_TARGETS_FOR_PATHWAYS]:
            # Try to query by gene name (best-effort)
            url = f"{base_url}/data/query/{target}/pathways"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    all_pathways = data
                    if len(all_pathways) > MAX_PATHWAYS_PER_TARGET:
                        log(f"[BIOKB] Truncating pathways for {target}: {len(all_pathways)} -> {MAX_PATHWAYS_PER_TARGET}")
                    for pathway in all_pathways[:MAX_PATHWAYS_PER_TARGET]:
                        if isinstance(pathway, dict):
                            name = pathway.get("displayName") or pathway.get("name")
                            if name:
                                pathways.add(name)
        
        # Cache result
        pathway_list = list(pathways)
        if cache_path:
            _write_json(cache_path, {"pathways": pathway_list})
        
        return pathway_list
        
    except Exception as e:
        log(f"[BIOKB][WARN] Reactome query failed for targets {targets[:3]}: {e}")
    
    return []


# -----------------------------
# Process Classification (Heuristic)
# -----------------------------

def _classify_processes(pathways: List[str]) -> List[str]:
    """Classify pathways into biological processes using keyword heuristic"""
    processes: Set[str] = set()
    
    # Simple keyword matching
    keywords = {
        "Proliferation": ["proliferation", "cell cycle", "mitosis", "growth", "division", "S phase", "G1", "G2", "M phase"],
        "Apoptosis": ["apoptosis", "cell death", "programmed cell death", "caspase", "death receptor", "cytochrome c"],
        "EMT": ["EMT", "epithelial", "mesenchymal", "transition", "migration", "invasion", "metastasis", "E-cadherin", "vimentin"],
    }
    
    for pathway in pathways:
        pathway_lower = pathway.lower()
        for process, kws in keywords.items():
            if any(kw.lower() in pathway_lower for kw in kws):
                processes.add(process)
    
    return list(processes)


# -----------------------------
# Main BioKB Generation
# -----------------------------

def generate_biokb_semantic_table(
    cfg: Dict[str, Any],
    stage: str,
    log: Callable[[str], None]
) -> BioKBSemanticTable:
    """Generate BioKB semantic table for a stage
    
    Args:
        cfg: Configuration dictionary
        stage: "design" or "review"
        log: Logging function
    
    Returns:
        BioKBSemanticTable with enriched entries
    """
    log(f"[BIOKB] 🧬 Generating semantic table for stage={stage}")
    
    # Extract SMILES
    smiles_list, source = extract_smiles(cfg)
    log(f"[BIOKB] 📊 Extracted {len(smiles_list)} SMILES from {source}")
    
    # Setup cache
    lit = cfg.get("literature", {}) if isinstance(cfg, dict) else {}
    bio_kb_cfg = lit.get("bio_kb", {}) if isinstance(lit, dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    
    literature_dir = (paths.get("literature_dir") or lit.get("literature_dir") or "").strip()
    cache_dir = os.path.join(literature_dir, "biokb_cache") if literature_dir else ""
    if cache_dir:
        _ensure_dir(cache_dir)
    
    # Process each SMILES
    entries: List[BioKBSemanticEntry] = []
    max_smiles = int(bio_kb_cfg.get("max_smiles", 10) or 10)  # Limit to avoid too many API calls
    
    for idx, smiles in enumerate(smiles_list[:max_smiles], 1):
        if not smiles or smiles == "SMILES_NOT_FOUND":
            # Add placeholder entry
            entries.append(BioKBSemanticEntry(
                smiles=smiles,
                canonical_smiles="",
                inchikey="",
                targets=["NO_TARGETS_AVAILABLE"],
                pathways=["NO_PATHWAYS_AVAILABLE"],
                processes=["UNKNOWN_PROCESS"]
            ))
            continue
        
        log(f"[BIOKB] 🔬 Processing SMILES {idx}/{min(len(smiles_list), max_smiles)}: {smiles[:30]}...")
        
        # Canonicalize
        canonical, inchikey = canonicalize_smiles(smiles)
        
        # Query targets
        targets = _query_chembl_targets(smiles, inchikey, cache_dir, log)
        
        # Query pathways
        pathways = _query_reactome_pathways(targets, cache_dir, log)
        
        # Classify processes
        processes = _classify_processes(pathways)
        
        entry = BioKBSemanticEntry(
            smiles=smiles,
            canonical_smiles=canonical,
            inchikey=inchikey,
            targets=targets if targets else ["NO_TARGETS_FOUND"],
            pathways=pathways if pathways else ["NO_PATHWAYS_FOUND"],
            processes=processes if processes else ["UNKNOWN_PROCESS"]
        )
        entries.append(entry)
    
    table = BioKBSemanticTable(
        stage=stage,
        generated_at=_now_iso(),
        entries=entries,
        smiles_source=source
    )
    
    log(f"[BIOKB] ✅ Generated semantic table with {len(entries)} entries")
    return table


def persist_biokb_semantic_table(
    table: BioKBSemanticTable,
    workspace_dir: str,
    log: Callable[[str], None]
) -> str:
    """Persist semantic table to workspace
    
    Returns:
        Path to saved JSON file
    """
    if not workspace_dir:
        return ""
    
    ext_kb_dir = os.path.join(workspace_dir, "external_knowledge")
    _ensure_dir(ext_kb_dir)
    
    output_path = os.path.join(ext_kb_dir, f"biokb_semantic_table_{table.stage}.json")
    
    try:
        _write_json(output_path, asdict(table))
        log(f"[BIOKB] 💾 Saved semantic table: {output_path}")
        return output_path
    except Exception as e:
        log(f"[BIOKB][WARN] Failed to save semantic table: {e}")
        return ""


def biokb_table_to_evidence_items(table: BioKBSemanticTable) -> List[Dict[str, Any]]:
    """Convert semantic table to evidence items with B* IDs
    
    Returns:
        List of evidence item dicts compatible with EvidenceItem format
    """
    items = []
    
    for idx, entry in enumerate(table.entries, 1):
        eid = f"B{idx}"
        
        # Create a summary text
        targets_str = ", ".join(entry.targets[:3]) if entry.targets else "None"
        pathways_str = ", ".join(entry.pathways[:3]) if entry.pathways else "None"
        processes_str = ", ".join(entry.processes) if entry.processes else "Unknown"
        
        snippet = (
            f"BioKB semantic mapping for molecular perturbation.\n"
            f"Predicted Targets: {targets_str}\n"
            f"Associated Pathways: {pathways_str}\n"
            f"Biological Processes: {processes_str}"
        )
        
        scraped_excerpt = (
            f"**SMILES**: {entry.smiles}\n"
            f"**Canonical SMILES**: {entry.canonical_smiles or 'N/A'}\n"
            f"**InChIKey**: {entry.inchikey or 'N/A'}\n\n"
            f"**Predicted Targets** ({len(entry.targets)}):\n"
            f"{'\n'.join('- ' + t for t in entry.targets[:10])}\n\n"
            f"**Associated Pathways** ({len(entry.pathways)}):\n"
            f"{'\n'.join('- ' + p for p in entry.pathways[:10])}\n\n"
            f"**Biological Processes**: {', '.join(entry.processes)}\n"
        )
        
        item = {
            "eid": eid,
            "title": f"BioKB: {entry.smiles[:40]}",
            "url": "",
            "snippet": snippet,
            "source": "biokb",
            "published": table.generated_at,
            "scraped_excerpt": scraped_excerpt
        }
        items.append(item)
    
    return items