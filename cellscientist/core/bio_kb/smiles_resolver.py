# -*- coding: utf-8 -*-
"""SMILES Resolver Module.

This module handles SMILES extraction from various sources and
canonicalization using RDKit (with graceful fallback).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple, Optional
from collections import Counter

from .utils import graceful_fallback
from .field_matcher import FieldMatcher
from .config import BioKBConfig

# Optional imports for H5 processing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


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


def _find_group_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    """Find group key from candidates (case-insensitive)"""
    keys_lower = {k.lower(): k for k in keys}
    for candidate in candidates:
        if candidate.lower() in keys_lower:
            return keys_lower[candidate.lower()]
    # Return first key if no match
    return keys[0] if keys else None


def _sample_smiles_batch_aware(
    df: Any,  # pandas DataFrame
    smiles_field: str,
    plate_field: str,
    config: BioKBConfig,
    metadata: Dict,
    log: Callable[[str], None]
) -> Tuple[List[str], Dict]:
    """
    Batch-aware SMILES sampling strategy.
    
    Strategy:
    1. Prioritize cross-batch SMILES (appear in multiple plates)
    2. Sample diverse SMILES from each batch
    3. Respect max_smiles_per_batch and max_total_smiles limits
    """
    plates = df[plate_field].unique()
    num_plates = len(plates)
    
    log(f"[BIOKB] 🧪 Detected {num_plates} batches/plates")
    
    # Analyze cross-batch SMILES
    smiles_by_plate = df.groupby(plate_field)[smiles_field].apply(lambda x: set(x.dropna())).to_dict()
    
    smiles_counter = Counter()
    for plate, smiles_set in smiles_by_plate.items():
        smiles_counter.update(smiles_set)
    
    cross_batch_smiles = {s: count for s, count in smiles_counter.items() if count > 1}
    single_batch_smiles = {s: count for s, count in smiles_counter.items() if count == 1}
    
    log(f"[BIOKB] 📊 Cross-batch SMILES: {len(cross_batch_smiles)} (robust)")
    log(f"[BIOKB] 📊 Single-batch SMILES: {len(single_batch_smiles)}")
    
    metadata["batch_info"] = {
        "num_plates": num_plates,
        "plates": plates.tolist(),
        "cross_batch_smiles": len(cross_batch_smiles),
        "single_batch_smiles": len(single_batch_smiles),
        "smiles_by_plate": {str(k): len(v) for k, v in smiles_by_plate.items()}
    }
    
    # Sampling strategy from config
    sampling_method = config.sampling_method
    max_per_batch = config.max_smiles_per_batch or 15
    max_total = config.max_total_smiles or len(smiles_counter)
    
    log(f"[BIOKB] 🎯 Sampling: method={sampling_method}, per_batch={max_per_batch}, total={max_total}")
    
    sampled = []
    
    if sampling_method == "cross_batch_first":
        # Strategy 1: Prioritize cross-batch SMILES
        sorted_cross = sorted(cross_batch_smiles.items(), key=lambda x: x[1], reverse=True)
        sampled.extend([s for s, _ in sorted_cross])
        
        # Fill remaining slots with single-batch SMILES
        if len(sampled) < max_total:
            sorted_single = sorted(single_batch_smiles.keys())
            sampled.extend(sorted_single[:max_total - len(sampled)])
    
    elif sampling_method == "diverse":
        # Strategy 2: Diverse sampling across batches
        for plate in plates:
            plate_smiles = list(smiles_by_plate[plate])
            n_sample = min(len(plate_smiles), max_per_batch)
            
            # Prioritize cross-batch within this plate
            plate_cross = [s for s in plate_smiles if s in cross_batch_smiles]
            plate_single = [s for s in plate_smiles if s in single_batch_smiles]
            
            plate_sample = plate_cross[:n_sample]
            if len(plate_sample) < n_sample:
                plate_sample.extend(plate_single[:n_sample - len(plate_sample)])
            
            sampled.extend(plate_sample)
            
            if len(set(sampled)) >= max_total:
                break
    
    elif sampling_method == "frequent":
        # Strategy 3: Most frequent SMILES
        sorted_by_freq = sorted(smiles_counter.items(), key=lambda x: x[1], reverse=True)
        sampled = [s for s, _ in sorted_by_freq]
    
    # Remove duplicates while preserving order
    unique_sampled = list(dict.fromkeys(sampled))[:max_total]
    
    metadata["sampling_info"] = {
        "method": sampling_method,
        "requested_total": max_total,
        "actual_total": len(unique_sampled),
        "cross_batch_selected": len([s for s in unique_sampled if s in cross_batch_smiles]),
        "single_batch_selected": len([s for s in unique_sampled if s in single_batch_smiles])
    }
    
    log(f"[BIOKB] ✅ Sampled {len(unique_sampled)} SMILES:")
    log(f"[BIOKB]    - Cross-batch: {metadata['sampling_info']['cross_batch_selected']}")
    log(f"[BIOKB]    - Single-batch: {metadata['sampling_info']['single_batch_selected']}")
    
    return unique_sampled, metadata


def _sample_smiles_simple(
    unique_smiles: Any,  # np.ndarray or list
    config: BioKBConfig,
    metadata: Dict,
    log: Callable[[str], None]
) -> Tuple[List[str], Dict]:
    """Simple sampling without batch information"""
    max_total = config.max_total_smiles or len(unique_smiles)
    sampled = list(unique_smiles)[:max_total]
    
    metadata["sampling_info"] = {
        "method": "simple",
        "requested_total": max_total,
        "actual_total": len(sampled)
    }
    
    log(f"[BIOKB] ✅ Sampled {len(sampled)} SMILES (simple mode)")
    
    return sampled, metadata


def _try_h5py_fallback(
    h5_path: str,
    config: BioKBConfig,
    matcher: FieldMatcher,
    metadata: Dict,
    log: Callable[[str], None]
) -> Tuple[List[str], Dict]:
    """Fallback to h5py if pandas fails"""
    try:
        import h5py
        smiles = []
        
        with h5py.File(h5_path, "r") as f:
            # Search for SMILES dataset recursively
            def search_smiles(name, obj):
                if isinstance(obj, h5py.Dataset):
                    field_name = name.split("/")[-1]
                    match = matcher.find_field([field_name], "smiles", min_confidence=0.6)
                    
                    if match:
                        try:
                            data = obj[:]
                            if data.dtype.kind in ['S', 'O', 'U']:  # string types
                                smiles.extend([str(s).strip() for s in data if s])
                        except Exception:
                            pass
            
            f.visititems(search_smiles)
        
        if smiles:
            unique_smiles = list(set(smiles))
            metadata["source"] = "h5_h5py"
            metadata["unique_smiles"] = len(unique_smiles)
            
            return _sample_smiles_simple(
                unique_smiles, config, metadata, log
            )
    
    except Exception as e:
        log(f"[BIOKB][ERROR] h5py fallback failed: {e}")
    
    return [], {**metadata, "error": "All extraction methods failed"}


def extract_smiles_from_h5_robust(
    h5_path: str,
    config: BioKBConfig,
    log: Optional[Callable[[str], None]] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Robustly extract SMILES from H5 file with:
    1. Fuzzy field matching
    2. Batch-aware sampling
    3. Comprehensive metadata
    
    Returns:
        (smiles_list, metadata_dict)
        
    metadata_dict contains:
        - source: "h5_pandas" | "h5_h5py" | "none"
        - total_samples: int
        - unique_smiles: int
        - batch_info: {...} if plate/batch field found
        - matched_fields: {...} mapping of target -> actual field names
        - sampling_info: {...} how sampling was performed
    """
    log = log or (lambda msg: print(msg))
    
    if not h5_path or not os.path.exists(h5_path):
        return [], {
            "source": "none",
            "error": f"H5 file not found: {h5_path}"
        }
    
    matcher = FieldMatcher()
    metadata = {
        "source": "unknown",
        "h5_path": h5_path,
        "matched_fields": {},
        "sampling_info": {}
    }
    
    # Try pandas HDFStore first (most common)
    try:
        import pandas as pd
        with pd.HDFStore(h5_path, "r") as store:
            # Find the "combined" group (or similar)
            group_key = _find_group_key(store.keys(), ["combined", "data", "main"])
            
            if not group_key:
                log(f"[BIOKB][WARN] No standard group found in H5. Keys: {store.keys()}")
                return _try_h5py_fallback(h5_path, config, matcher, metadata, log)
            
            df = store[group_key]
            metadata["source"] = "h5_pandas"
            metadata["group_key"] = group_key
            metadata["total_samples"] = len(df)
            
            log(f"[BIOKB] 📂 Loaded H5 group: {group_key} ({len(df)} samples)")
            
            # Fuzzy match required fields
            available_fields = df.columns.tolist()
            log(f"[BIOKB] 🔍 Available fields: {available_fields}")
            
            field_matches = matcher.find_all_fields(
                available_fields,
                targets=["smiles", "plate_id", "dose"],
                min_confidence=0.6
            )
            
            # Build matched_fields metadata
            metadata["matched_fields"] = {}
            for target, match in field_matches.items():
                if match:
                    field, conf, mtype = match
                    metadata["matched_fields"][target] = {
                        "field": field,
                        "confidence": conf,
                        "match_type": mtype
                    }
                else:
                    metadata["matched_fields"][target] = None
            
            # Log field matching results
            for target, match in field_matches.items():
                if match:
                    field, conf, mtype = match
                    log(f"[BIOKB] ✅ Matched '{target}' → '{field}' (conf={conf:.2f}, type={mtype})")
                else:
                    log(f"[BIOKB] ⚠️ Field '{target}' not found (optional)")
            
            # Extract SMILES field (required)
            smiles_match = field_matches.get("smiles")
            if not smiles_match:
                log(f"[BIOKB][ERROR] No SMILES field found!")
                log(f"[BIOKB][ERROR] Tried: {matcher.SYNONYMS['smiles']}")
                return [], {**metadata, "error": "SMILES field not found"}
            
            smiles_field, _, _ = smiles_match
            smiles_series = df[smiles_field].dropna()
            unique_smiles = smiles_series.unique()
            
            metadata["unique_smiles"] = len(unique_smiles)
            metadata["smiles_field"] = smiles_field
            
            # Check for batch/plate information
            plate_match = field_matches.get("plate_id")
            
            if plate_match:
                plate_field, _, _ = plate_match
                metadata["plate_field"] = plate_field
                
                # Batch-aware sampling
                return _sample_smiles_batch_aware(
                    df, smiles_field, plate_field, config, metadata, log
                )
            else:
                # Simple sampling (no batch info)
                log(f"[BIOKB] ⚠️ No batch/plate field found. Using simple sampling.")
                return _sample_smiles_simple(
                    unique_smiles, config, metadata, log
                )
    
    except Exception as e:
        log(f"[BIOKB][WARN] Pandas HDFStore failed: {e}")
        return _try_h5py_fallback(h5_path, config, matcher, metadata, log)
