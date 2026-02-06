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
    
    # Output limits
    max_smiles: int = 10
    max_targets_per_smiles: int = 10
    max_pathways_per_target: int = 20
    inject_max_items: int = 5
    
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
    
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "BioKBConfig":
        """Create BioKBConfig from main config dict at cfg["literature"]["bio_kb"].
        
        Args:
            cfg: Main configuration dictionary
            
        Returns:
            BioKBConfig instance
        """
        lit = cfg.get("literature", {}) if isinstance(cfg, dict) else {}
        bio_kb = lit.get("bio_kb", {}) if isinstance(lit, dict) else {}
        return cls.from_dict(bio_kb) if isinstance(bio_kb, dict) else cls()
