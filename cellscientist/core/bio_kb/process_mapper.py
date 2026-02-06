# -*- coding: utf-8 -*-
"""Process Mapper Module.

This module maps biological pathways to biological processes
using keyword-based heuristics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .data_lake import DataLake


def map_pathway_to_process(
    pathway_name: str,
    data_lake: DataLake,
    confidence_threshold: float = 0.1
) -> Optional[Dict[str, Any]]:
    """Map a pathway name to a biological process using keyword matching.
    
    Args:
        pathway_name: Name of the biological pathway
        data_lake: DataLake instance with keyword mappings
        confidence_threshold: Minimum confidence score to return result
        
    Returns:
        Dictionary with keys:
        - process: Process name
        - confidence: Confidence score (0.0-1.0)
        - matched_keywords: List of matched keywords
        Or None if no match found above threshold
    """
    matches = data_lake.search_keywords(pathway_name)
    
    if not matches:
        return None
    
    # Find process with most keyword matches
    best_process = None
    best_score = 0.0
    best_keywords = []
    
    for process, keywords in matches.items():
        # Calculate confidence based on number of matches
        # Normalize by total keywords for that process
        total_keywords = len(data_lake.get_process_keywords(process))
        score = len(keywords) / max(total_keywords, 1)
        
        if score > best_score:
            best_score = score
            best_process = process
            best_keywords = list(keywords)
    
    if best_process and best_score >= confidence_threshold:
        return {
            "process": best_process,
            "confidence": round(best_score, 2),
            "matched_keywords": best_keywords
        }
    
    return None


def map_pathways_to_processes(
    pathways: List[Dict[str, str]],
    data_lake: DataLake
) -> List[Dict[str, Any]]:
    """Map multiple pathways to biological processes.
    
    Args:
        pathways: List of pathway dictionaries with "pathway_name" key
        data_lake: DataLake instance with keyword mappings
        
    Returns:
        List of process mapping dictionaries with deduplicated processes
    """
    process_map: Dict[str, Dict[str, Any]] = {}
    
    for pathway in pathways:
        pathway_name = pathway.get("pathway_name", "")
        if not pathway_name:
            continue
        
        result = map_pathway_to_process(pathway_name, data_lake)
        if result:
            process = result["process"]
            
            # Keep highest confidence and merge keywords
            if process in process_map:
                existing = process_map[process]
                if result["confidence"] > existing["confidence"]:
                    process_map[process] = result
                else:
                    # Merge keywords
                    existing_kw = set(existing["matched_keywords"])
                    new_kw = set(result["matched_keywords"])
                    existing["matched_keywords"] = list(existing_kw | new_kw)
            else:
                process_map[process] = result
    
    return list(process_map.values())


def classify_processes_simple(pathways: List[str]) -> List[str]:
    """Simple classification of pathways into biological processes.
    
    This is a simplified version for backward compatibility.
    
    Args:
        pathways: List of pathway names
        
    Returns:
        List of identified process names
    """
    data_lake = DataLake()
    pathway_dicts = [{"pathway_name": p} for p in pathways]
    results = map_pathways_to_processes(pathway_dicts, data_lake)
    return [r["process"] for r in results]
