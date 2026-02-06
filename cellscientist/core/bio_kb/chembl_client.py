# -*- coding: utf-8 -*-
"""ChEMBL API Client Module.

This module provides functions for querying the ChEMBL database
for drug targets and mechanisms with caching and timeout protection.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import requests

from .config import BioKBConfig
from .utils import ensure_dir, sha1_hash, with_timeout


def _get_cache_path(cache_dir: str, cache_key: str) -> str:
    """Get cache file path for a given cache key.
    
    Args:
        cache_dir: Base cache directory
        cache_key: Cache key identifier
        
    Returns:
        Full path to cache file
    """
    if not cache_dir:
        return ""
    chembl_cache = os.path.join(cache_dir, "chembl")
    ensure_dir(chembl_cache)
    return os.path.join(chembl_cache, f"{cache_key}.json")


def _read_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """Read cached data from file.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Cached data or None if not found or invalid
    """
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(cache_path: str, data: Dict[str, Any]) -> None:
    """Write data to cache file.
    
    Args:
        cache_path: Path to cache file
        data: Data to cache
    """
    if not cache_path:
        return
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


@with_timeout(30, fallback=[])
def _query_chembl_api(url: str, timeout: int) -> Any:
    """Query ChEMBL API with timeout protection.
    
    Args:
        url: API endpoint URL
        timeout: Request timeout in seconds
        
    Returns:
        API response data or empty list on failure
    """
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 200:
        return resp.json()
    return []


def query_targets_by_smiles(
    smiles: str,
    inchikey: str,
    config: BioKBConfig,
    log: Callable[[str], None]
) -> List[Dict[str, Any]]:
    """Query ChEMBL API to find targets for a molecule.
    
    Args:
        smiles: SMILES string
        inchikey: InChIKey string
        config: BioKB configuration
        log: Logging function
        
    Returns:
        List of target dictionaries with keys:
        - target_name: Target name
        - target_id: ChEMBL target ID
        - gene_symbol: Gene symbol
        - mechanism: Mechanism of action
    """
    if not config.chembl_enabled:
        return []
    
    # Check cache
    cache_key = sha1_hash(f"chembl_targets:{smiles}:{inchikey}")
    cache_path = _get_cache_path(config.cache_dir, cache_key) if config.cache_enabled else ""
    
    cached = _read_cache(cache_path)
    if cached and "targets" in cached:
        log(f"[BIOKB] 💾 Using cached ChEMBL targets for {smiles[:20]}...")
        return cached["targets"]
    
    targets: List[Dict[str, Any]] = []
    
    try:
        base_url = config.chembl_base_url
        
        # Try by InChIKey first (more reliable)
        if inchikey:
            url = f"{base_url}/molecule.json?molecule_structures__standard_inchi_key={inchikey}"
            data = _query_chembl_api(url, config.chembl_timeout)
            
            if data and isinstance(data, dict):
                molecules = data.get("molecules", [])
                if molecules:
                    molecule_id = molecules[0].get("molecule_chembl_id")
                    if molecule_id:
                        # Get mechanisms
                        mech_url = f"{base_url}/mechanism.json?molecule_chembl_id={molecule_id}"
                        mech_data = _query_chembl_api(mech_url, config.chembl_timeout)
                        
                        if mech_data and isinstance(mech_data, dict):
                            mechanisms = mech_data.get("mechanisms", [])
                            
                            # Limit number of mechanisms
                            max_mechs = min(len(mechanisms), config.max_targets_per_smiles)
                            if len(mechanisms) > max_mechs:
                                log(f"[BIOKB] Truncating mechanisms: {len(mechanisms)} -> {max_mechs}")
                            
                            for mech in mechanisms[:max_mechs]:
                                target_chembl_id = mech.get("target_chembl_id", "")
                                mechanism_action = mech.get("mechanism_of_action", "")
                                
                                # Try to get target details
                                target_name = ""
                                gene_symbol = ""
                                
                                if target_chembl_id:
                                    target_url = f"{base_url}/target/{target_chembl_id}.json"
                                    target_data = _query_chembl_api(target_url, config.chembl_timeout)
                                    
                                    if target_data and isinstance(target_data, dict):
                                        target_name = target_data.get("pref_name", target_chembl_id)
                                        
                                        # Extract gene symbol from target components
                                        components = target_data.get("target_components", [])
                                        if components and len(components) > 0:
                                            accession = components[0].get("accession", "")
                                            gene_symbol = accession or target_chembl_id
                                
                                targets.append({
                                    "target_name": target_name or target_chembl_id,
                                    "target_id": target_chembl_id,
                                    "gene_symbol": gene_symbol or target_chembl_id,
                                    "mechanism": mechanism_action
                                })
        
        # Cache result
        if cache_path:
            _write_cache(cache_path, {"targets": targets})
        
    except Exception as e:
        log(f"[BIOKB][WARN] ChEMBL query failed for {smiles[:20]}: {e}")
    
    return targets


def query_mechanism_by_molecule(
    molecule_id: str,
    config: BioKBConfig,
    log: Callable[[str], None]
) -> Optional[str]:
    """Query mechanism of action for a ChEMBL molecule.
    
    Args:
        molecule_id: ChEMBL molecule ID
        config: BioKB configuration
        log: Logging function
        
    Returns:
        Mechanism of action string or None
    """
    if not config.chembl_enabled:
        return None
    
    try:
        url = f"{config.chembl_base_url}/mechanism.json?molecule_chembl_id={molecule_id}"
        data = _query_chembl_api(url, config.chembl_timeout)
        
        if data and isinstance(data, dict):
            mechanisms = data.get("mechanisms", [])
            if mechanisms:
                return mechanisms[0].get("mechanism_of_action", "")
    except Exception as e:
        log(f"[BIOKB][WARN] Mechanism query failed for {molecule_id}: {e}")
    
    return None
