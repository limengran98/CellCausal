# -*- coding: utf-8 -*-
"""BioKB Configuration Module.

This module defines the BioKBConfig dataclass with support for:
- Environment variable overrides
- Nested configuration from cfg["literature"]["bio_kb"]
- Sensible defaults for backward compatibility
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BioKBConfig:
    """Configuration for BioKB module."""
    
    enabled: bool = False  # Default disabled for backward compatibility
    
    # SMILES sources (priority: smiles_list > h5_path)
    smiles_list: List[str] = field(default_factory=list)
    h5_path: Optional[str] = None
    
    # API configuration
    chembl_enabled: bool = True
    chembl_base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    chembl_timeout: int = 30
    
    reactome_enabled: bool = True
    reactome_base_url: str = "https://reactome.org/ContentService"
    reactome_timeout: int = 30
    
    # Caching
    cache_enabled: bool = True
    cache_dir: str = ""  # Auto-set to bio_kb/cache/ if empty
    cache_ttl_days: int = 30
    
    # Sampling strategy
    sampling_strategy: str = "adaptive"  # "adaptive" | "fixed"
    sampling_method: str = "cross_batch_first"  # "cross_batch_first" | "diverse" | "frequent"
    
    # Adaptive limits (None = auto-determine)
    max_smiles_per_batch: Optional[int] = 15
    max_total_smiles: Optional[int] = None  # None = use adaptive logic
    
    # Adaptive logic parameters
    adaptive_min: int = 50   # Minimum SMILES to sample
    adaptive_max: int = 300  # Maximum SMILES to sample
    adaptive_ratio: float = 0.2  # Sample 20% of unique SMILES
    
    # Output limits (backward compatibility)
    max_smiles: int = 10
    max_targets_per_smiles: int = 10
    max_pathways_per_target: int = 20
    inject_max_items: int = 5
    include_batch_info: bool = True
    
    # Performance
    parallel_queries: int = 3
    
    # Logging
    log_to_console: bool = True
    
    def __post_init__(self):
        """Post-initialization: apply environment variable overrides."""
        # Environment variable overrides
        if os.getenv("BIOKB_ENABLED"):
            self.enabled = os.getenv("BIOKB_ENABLED", "").lower() in ("true", "1", "yes")
        if os.getenv("STAGE1_H5_PATH") and not self.h5_path:
            self.h5_path = os.getenv("STAGE1_H5_PATH")
        if not self.cache_dir:
            self.cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BioKBConfig":
        """Create BioKBConfig from dictionary, filtering unknown keys.
        
        Args:
            d: Dictionary with configuration values
            
        Returns:
            BioKBConfig instance
        """
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)
    
    def get_adaptive_max_smiles(self, unique_smiles_count: int) -> int:
        """
        Calculate adaptive max_smiles based on actual data size.
        
        Logic:
        - If unique_smiles <= adaptive_min: use all
        - If unique_smiles <= adaptive_max: use adaptive_ratio * count
        - If unique_smiles > adaptive_max: cap at adaptive_max
        """
        if self.sampling_strategy != "adaptive":
            return self.max_total_smiles or unique_smiles_count
        
        if unique_smiles_count <= self.adaptive_min:
            return unique_smiles_count
        
        adaptive_count = int(unique_smiles_count * self.adaptive_ratio)
        return min(max(adaptive_count, self.adaptive_min), self.adaptive_max)
    
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "BioKBConfig":
        """Create BioKBConfig from main config dict.
        
        Supports two config structures:
        1. cfg["literature"]["bio_kb"] (external_knowledge_mirothink.py expected structure)
        2. cfg["bio_kb"] (pipeline_config.json structure)
        
        Args:
            cfg: Main configuration dictionary
            
        Returns:
            BioKBConfig instance
        """
        if not isinstance(cfg, dict):
            return cls()
        
        # Priority 1: Check cfg["literature"]["bio_kb"] (external_knowledge_mirothink.py structure)
        # This takes precedence for backward compatibility with existing external_knowledge integrations
        lit = cfg.get("literature", {})
        if isinstance(lit, dict):
            bio_kb = lit.get("bio_kb", {})
            if isinstance(bio_kb, dict) and bio_kb:
                return cls.from_dict(bio_kb)
        
        # Priority 2: Check cfg["bio_kb"] (pipeline_config.json structure)
        # Fallback to root-level config if nested structure not found
        bio_kb = cfg.get("bio_kb", {})
        if isinstance(bio_kb, dict) and bio_kb:
            return cls.from_dict(bio_kb)
        
        # Default: return disabled config
        return cls()
