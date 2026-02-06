# -*- coding: utf-8 -*-
"""BioKB Tool Schemas.

This module defines declarative schemas for all knowledge sources
and tools available in BioKB, providing metadata about inputs,
outputs, and requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ToolSchema:
    """Schema definition for a BioKB tool or knowledge source."""
    
    name: str
    description: str
    category: str  # "target" | "pathway" | "process" | "resolver"
    required_params: List[str]
    optional_params: List[str]
    output_type: str
    timeout_seconds: int
    cacheable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation.
        
        Returns:
            Dictionary representation of the schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "required_params": self.required_params,
            "optional_params": self.optional_params,
            "output_type": self.output_type,
            "timeout_seconds": self.timeout_seconds,
            "cacheable": self.cacheable
        }


# Tool schema definitions
SMILES_RESOLVER_SCHEMA = ToolSchema(
    name="smiles_resolver",
    description="Canonicalize SMILES and generate InChIKey using RDKit",
    category="resolver",
    required_params=["smiles"],
    optional_params=[],
    output_type="dict[canonical_smiles, inchikey]",
    timeout_seconds=5,
    cacheable=True
)

CHEMBL_TARGET_SCHEMA = ToolSchema(
    name="chembl_target_query",
    description="Query ChEMBL API for drug targets by SMILES or InChIKey",
    category="target",
    required_params=["smiles"],
    optional_params=["inchikey", "max_targets"],
    output_type="list[dict[target_name, target_id, gene_symbol, mechanism]]",
    timeout_seconds=30,
    cacheable=True
)

REACTOME_PATHWAY_SCHEMA = ToolSchema(
    name="reactome_pathway_query",
    description="Query Reactome API for pathways by gene symbol",
    category="pathway",
    required_params=["gene_symbol"],
    optional_params=["max_pathways"],
    output_type="list[dict[pathway_id, pathway_name, source]]",
    timeout_seconds=30,
    cacheable=True
)

PROCESS_MAPPER_SCHEMA = ToolSchema(
    name="process_mapper",
    description="Map pathway names to biological processes using keyword heuristics",
    category="process",
    required_params=["pathway_name"],
    optional_params=["confidence_threshold"],
    output_type="dict[process, confidence, matched_keywords]",
    timeout_seconds=1,
    cacheable=False
)

# Registry of all tool schemas
TOOL_SCHEMAS: Dict[str, ToolSchema] = {
    "smiles_resolver": SMILES_RESOLVER_SCHEMA,
    "chembl_target_query": CHEMBL_TARGET_SCHEMA,
    "reactome_pathway_query": REACTOME_PATHWAY_SCHEMA,
    "process_mapper": PROCESS_MAPPER_SCHEMA
}


def get_tool_schema(name: str) -> ToolSchema | None:
    """Get tool schema by name.
    
    Args:
        name: Tool name
        
    Returns:
        ToolSchema if found, None otherwise
    """
    return TOOL_SCHEMAS.get(name)


def list_tool_schemas() -> List[ToolSchema]:
    """Get list of all available tool schemas.
    
    Returns:
        List of all tool schemas
    """
    return list(TOOL_SCHEMAS.values())
