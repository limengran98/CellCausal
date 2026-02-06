# -*- coding: utf-8 -*-
"""SMILES Resolver Module.

This module handles SMILES extraction from various sources and
canonicalization using RDKit (with graceful fallback).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple

from .utils import graceful_fallback


def canonicalize_smiles(smiles: str) -> Dict[str, str]:
    """Generate canonical SMILES and InChIKey using RDKit if available.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Dictionary with "canonical_smiles" and "inchikey" keys
        Falls back gracefully if RDKit unavailable
    """
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            return {"canonical_smiles": canonical, "inchikey": inchikey}
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: return original SMILES
    return {"canonical_smiles": smiles, "inchikey": None}


def extract_smiles_from_config(cfg: Dict[str, Any]) -> Tuple[List[str], str]:
    """Extract SMILES from config['literature']['bio_kb']['smiles_list'].
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (smiles_list, source_description)
    """
    try:
        lit = cfg.get("literature", {}) if isinstance(cfg, dict) else {}
        bio_kb_cfg = lit.get("bio_kb", {}) if isinstance(lit, dict) else {}
        smiles_list = bio_kb_cfg.get("smiles_list", []) if isinstance(bio_kb_cfg, dict) else []
        
        if isinstance(smiles_list, list) and smiles_list:
            return [str(s).strip() for s in smiles_list if s], "config"
    except Exception:
        pass
    
    return [], "none"


@graceful_fallback(([], "none"))
def extract_smiles_from_h5(h5_path: str) -> Tuple[List[str], str]:
    """Extract SMILES from H5 file (best-effort, supports pandas HDFStore and h5py).
    
    Args:
        h5_path: Path to H5 file
        
    Returns:
        Tuple of (smiles_list, source_description)
    """
    if not h5_path or not os.path.exists(h5_path):
        return [], "none"
    
    smiles = []
    
    # Try pandas HDFStore first
    try:
        import pandas as pd  # type: ignore
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
        import h5py  # type: ignore
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


def extract_smiles(cfg: Dict[str, Any], log: Callable[[str], None]) -> Tuple[List[str], str]:
    """Extract SMILES with priority: config > H5 > placeholder.
    
    Args:
        cfg: Configuration dictionary
        log: Logging function
        
    Returns:
        Tuple of (smiles_list, source_description)
    """
    # Priority 1: Config
    smiles, source = extract_smiles_from_config(cfg)
    if smiles:
        log(f"[BIOKB] 📊 Extracted {len(smiles)} SMILES from config")
        return smiles, source
    
    # Priority 2: H5 from env or config
    h5_path = os.environ.get("STAGE1_H5_PATH", "").strip()
    if not h5_path:
        paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
        h5_path = paths.get("stage1_h5", "") if isinstance(paths, dict) else ""
    
    if h5_path:
        smiles, source = extract_smiles_from_h5(h5_path)
        if smiles:
            log(f"[BIOKB] 📊 Extracted {len(smiles)} SMILES from H5: {h5_path}")
            return smiles, source
    
    # Priority 3: Return visible placeholder (no silent fail)
    log("[BIOKB] ⚠️  No SMILES found in config or H5, using placeholder")
    return ["SMILES_NOT_FOUND"], "placeholder_missing"
