# -*- coding: utf-8 -*-
"""BioKB Registry Module.

This module implements the registry pattern for managing knowledge sources,
similar to Biomni's ToolRegistry but adapted for BioKB's use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class KnowledgeSource:
    """Metadata for a knowledge source in BioKB."""
    
    name: str
    description: str
    category: str  # "target" | "pathway" | "process"
    query_fn: Callable
    required_params: List[str]
    optional_params: List[str]
    cache_enabled: bool
    timeout_seconds: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (excluding callable).
        
        Returns:
            Dictionary representation of the knowledge source
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "required_params": self.required_params,
            "optional_params": self.optional_params,
            "cache_enabled": self.cache_enabled,
            "timeout_seconds": self.timeout_seconds
        }


class BioKBRegistry:
    """Registry for managing BioKB knowledge sources."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._sources: Dict[str, KnowledgeSource] = {}
    
    def register(self, source: KnowledgeSource) -> None:
        """Register a knowledge source.
        
        Args:
            source: KnowledgeSource to register
        """
        self._sources[source.name] = source
    
    def get(self, name: str) -> Optional[KnowledgeSource]:
        """Get knowledge source by name.
        
        Args:
            name: Source name
            
        Returns:
            KnowledgeSource if found, None otherwise
        """
        return self._sources.get(name)
    
    def list_sources(self, category: Optional[str] = None) -> List[KnowledgeSource]:
        """List all registered knowledge sources, optionally filtered by category.
        
        Args:
            category: Optional category filter ("target", "pathway", "process")
            
        Returns:
            List of knowledge sources
        """
        sources = list(self._sources.values())
        if category:
            sources = [s for s in sources if s.category == category]
        return sources
    
    def list_categories(self) -> List[str]:
        """Get list of all categories in the registry.
        
        Returns:
            List of unique category names
        """
        return list(set(s.category for s in self._sources.values()))
    
    def query(self, source_name: str, **kwargs) -> Any:
        """Query a knowledge source by name.
        
        Args:
            source_name: Name of the source to query
            **kwargs: Query parameters
            
        Returns:
            Query result
            
        Raises:
            KeyError: If source not found
            ValueError: If required parameters missing
        """
        source = self._sources.get(source_name)
        if not source:
            raise KeyError(f"Knowledge source '{source_name}' not found")
        
        # Validate required parameters
        missing = [p for p in source.required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # Call the query function
        return source.query_fn(**kwargs)
    
    def __len__(self) -> int:
        """Get number of registered sources.
        
        Returns:
            Number of sources
        """
        return len(self._sources)
    
    def __contains__(self, name: str) -> bool:
        """Check if source is registered.
        
        Args:
            name: Source name
            
        Returns:
            True if source is registered
        """
        return name in self._sources
