# -*- coding: utf-8 -*-
"""BioKB Data Lake Module.

This module provides pre-computed mappings and static knowledge that
don't require external API calls, including:
- Pathway keyword → biological process mappings
- Common gene → process associations
- Drug mechanism classifications
"""

from __future__ import annotations

from typing import Dict, List, Set


# Pathway to Process Keyword Mappings
PATHWAY_TO_PROCESS_KEYWORDS: Dict[str, List[str]] = {
    "proliferation": [
        "cell cycle", "mitosis", "G1/S", "G2/M", "cyclin", "CDK",
        "MAPK", "ERK", "MEK", "RAS", "RAF", "PI3K", "AKT", "mTOR",
        "WNT", "beta-catenin", "MYC", "E2F", "RB1", "proliferation",
        "growth", "division", "S phase", "M phase", "G1 phase", "G2 phase",
        "cell division", "DNA replication", "chromosome segregation"
    ],
    "apoptosis": [
        "apoptosis", "apoptotic", "caspase", "BCL2", "BAX", "BAK",
        "p53", "TP53", "death receptor", "FAS", "TRAIL", "TNF",
        "cytochrome c", "PARP", "annexin", "programmed cell death",
        "cell death", "intrinsic apoptotic", "extrinsic apoptotic",
        "mitochondrial apoptotic", "MOMP", "BH3"
    ],
    "emt": [
        "epithelial-mesenchymal", "EMT", "E-cadherin", "N-cadherin",
        "vimentin", "SNAIL", "SLUG", "TWIST", "ZEB1", "ZEB2",
        "TGF-beta", "WNT", "NOTCH", "fibronectin", "epithelial",
        "mesenchymal", "transition", "migration", "invasion", "metastasis",
        "epithelial to mesenchymal"
    ],
    "autophagy": [
        "autophagy", "LC3", "BECN1", "ATG", "mTOR", "AMPK",
        "ULK1", "p62", "SQSTM1", "autophagic", "autophagosome",
        "lysosome", "macroautophagy", "chaperone-mediated autophagy"
    ],
    "senescence": [
        "senescence", "SASP", "p16", "p21", "CDKN2A", "CDKN1A",
        "telomere", "beta-galactosidase", "senescent", "cellular senescence",
        "replicative senescence", "oncogene-induced senescence"
    ],
    "differentiation": [
        "differentiation", "stem cell", "pluripotency", "lineage commitment",
        "cell fate", "progenitor", "maturation", "developmental"
    ],
    "inflammation": [
        "inflammation", "inflammatory", "cytokine", "chemokine", "interleukin",
        "IL-1", "IL-6", "IL-8", "NF-kB", "NFkappaB", "immune response",
        "innate immunity", "adaptive immunity"
    ],
    "metabolism": [
        "metabolism", "metabolic", "glycolysis", "oxidative phosphorylation",
        "OXPHOS", "TCA cycle", "Krebs cycle", "fatty acid", "glucose",
        "mitochondrial", "ATP"
    ]
}


class DataLake:
    """Data lake for pre-computed biological knowledge."""
    
    def __init__(self):
        """Initialize data lake with keyword mappings."""
        self.pathway_keywords = PATHWAY_TO_PROCESS_KEYWORDS
    
    def get_process_keywords(self, process: str) -> List[str]:
        """Get keywords associated with a biological process.
        
        Args:
            process: Process name (e.g., "apoptosis", "proliferation")
            
        Returns:
            List of keywords for the process
        """
        return self.pathway_keywords.get(process.lower(), [])
    
    def get_all_processes(self) -> List[str]:
        """Get list of all known biological processes.
        
        Returns:
            List of process names
        """
        return list(self.pathway_keywords.keys())
    
    def search_keywords(self, text: str) -> Dict[str, Set[str]]:
        """Search text for process-related keywords.
        
        Args:
            text: Text to search (e.g., pathway name or description)
            
        Returns:
            Dictionary mapping process names to matched keywords
        """
        text_lower = text.lower()
        matches: Dict[str, Set[str]] = {}
        
        for process, keywords in self.pathway_keywords.items():
            matched = set()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matched.add(keyword)
            
            if matched:
                matches[process] = matched
        
        return matches
