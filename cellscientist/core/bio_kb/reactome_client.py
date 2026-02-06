# -*- coding: utf-8 -*-
"""Reactome API Client Module.

This module provides functions for querying the Reactome database
for biological pathways with caching and timeout protection.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

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
    reactome_cache = os.path.join(cache_dir, "reactome")
    ensure_dir(reactome_cache)
    return os.path.join(reactome_cache, f"{cache_key}.json")


def _read_cache(cache_path: str) -> Dict[str, Any] | None:
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
def _query_reactome_api(url: str, timeout: int) -> Any:
    """Query Reactome API with timeout protection.
    
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


def query_pathways_by_gene(
    gene_symbol: str,
    config: BioKBConfig,
    log: Callable[[str], None]
) -> List[Dict[str, Any]]:
    """Query Reactome API to find pathways for a gene.
    
    Args:
        gene_symbol: Gene symbol (e.g., "PTGS1", "TP53")
        config: BioKB configuration
        log: Logging function
        
    Returns:
        List of pathway dictionaries with keys:
        - pathway_id: Reactome pathway ID
        - pathway_name: Pathway display name
        - source: "reactome"
    """
    if not config.reactome_enabled:
        return []
    
    # Check cache
    cache_key = sha1_hash(f"reactome_pathways:{gene_symbol}")
    cache_path = _get_cache_path(config.cache_dir, cache_key) if config.cache_enabled else ""
    
    cached = _read_cache(cache_path)
    if cached and "pathways" in cached:
        log(f"[BIOKB] 💾 Using cached Reactome pathways for {gene_symbol}")
        return cached["pathways"]
    
    pathways: List[Dict[str, Any]] = []
    
    try:
        base_url = config.reactome_base_url
        url = f"{base_url}/data/query/{gene_symbol}/pathways"
        
        data = _query_reactome_api(url, config.reactome_timeout)
        
        if data and isinstance(data, list):
            # Limit number of pathways
            max_pathways = min(len(data), config.max_pathways_per_target)
            if len(data) > max_pathways:
                log(f"[BIOKB] Truncating pathways for {gene_symbol}: {len(data)} -> {max_pathways}")
            
            for pathway in data[:max_pathways]:
                if isinstance(pathway, dict):
                    pathway_id = pathway.get("stId") or pathway.get("dbId", "")
                    pathway_name = pathway.get("displayName") or pathway.get("name", "")
                    
                    if pathway_name:
                        pathways.append({
                            "pathway_id": str(pathway_id),
                            "pathway_name": pathway_name,
                            "source": "reactome"
                        })
        
        # Cache result
        if cache_path:
            _write_cache(cache_path, {"pathways": pathways})
        
    except Exception as e:
        log(f"[BIOKB][WARN] Reactome query failed for {gene_symbol}: {e}")
    
    return pathways


def query_pathway_details(
    pathway_id: str,
    config: BioKBConfig,
    log: Callable[[str], None]
) -> Dict[str, Any] | None:
    """Query detailed information about a specific pathway.
    
    Args:
        pathway_id: Reactome pathway ID
        config: BioKB configuration
        log: Logging function
        
    Returns:
        Pathway details dictionary or None on failure
    """
    if not config.reactome_enabled:
        return None
    
    try:
        base_url = config.reactome_base_url
        url = f"{base_url}/data/pathway/{pathway_id}"
        
        data = _query_reactome_api(url, config.reactome_timeout)
        
        if data and isinstance(data, dict):
            return {
                "pathway_id": data.get("stId", pathway_id),
                "pathway_name": data.get("displayName", ""),
                "species": data.get("speciesName", ""),
                "summation": data.get("summation", [{}])[0].get("text", "") if data.get("summation") else ""
            }
    except Exception as e:
        log(f"[BIOKB][WARN] Pathway details query failed for {pathway_id}: {e}")
    
    return None
