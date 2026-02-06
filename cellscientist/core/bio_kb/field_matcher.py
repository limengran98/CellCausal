# -*- coding: utf-8 -*-
"""
Fuzzy field matcher for H5 datasets.
Handles diverse naming conventions with confidence scoring.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


class FieldMatcher:
    """
    Intelligent field name matcher with fuzzy logic and synonyms.
    
    Example:
        matcher = FieldMatcher()
        result = matcher.find_field(
            available_fields=["SMILES", "Dose", "PlateID"],
            target="smiles"
        )
        # Returns: ("SMILES", 1.0, "exact_case_insensitive")
    """
    
    # Field synonym dictionary
    SYNONYMS = {
        "smiles": [
            "smiles", "SMILES", "Smiles",
            "smiles_string", "canonical_smiles", "mol_smiles",
            "compound_smiles", "smiles_canonical", "smiles_str",
            "molecule_smiles", "chem_smiles", "structure"
        ],
        "dose": [
            "dose", "Dose", "DOSE",
            "concentration", "conc", "Conc", "dosage",
            "treatment_dose", "drug_dose", "compound_dose",
            "amount", "quantity"
        ],
        "plate_id": [
            "plate_id", "plate", "Plate", "plate_ID",
            "batch", "Batch", "batch_id", "batch_ID",
            "plate_number", "plate_name", "experiment_plate",
            "assay_plate", "screen_plate"
        ],
        "split_id": [
            "split_id", "split", "Split", "fold",
            "cv_fold", "cross_validation", "fold_id",
            "split_index", "cv_index", "kfold"
        ]
    }
    
    def __init__(self):
        # Pre-compile regex patterns for performance
        self.patterns = {
            target: self._build_pattern(synonyms)
            for target, synonyms in self.SYNONYMS.items()
        }
    
    def _build_pattern(self, synonyms: List[str]) -> re.Pattern:
        """Build case-insensitive regex pattern from synonyms"""
        # Escape special characters and join with |
        escaped = [re.escape(s) for s in synonyms]
        pattern = "|".join(escaped)
        return re.compile(f"^({pattern})$", re.IGNORECASE)
    
    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate string similarity (0.0 to 1.0)"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def find_field(
        self, 
        available_fields: List[str], 
        target: str,
        min_confidence: float = 0.6
    ) -> Optional[Tuple[str, float, str]]:
        """
        Find best matching field from available fields.
        
        Returns:
            (matched_field, confidence, match_type) or None
            
        Match types:
            - "exact": Perfect match
            - "exact_case_insensitive": Case-insensitive match
            - "synonym": Matched via synonym list
            - "fuzzy": Fuzzy string matching (>= min_confidence)
        """
        if not available_fields:
            return None
        
        target_lower = target.lower()
        synonyms = self.SYNONYMS.get(target, [target])
        
        # Strategy 1: Exact match (case-sensitive)
        if target in available_fields:
            return (target, 1.0, "exact")
        
        # Strategy 2: Exact match (case-insensitive)
        for field in available_fields:
            if field.lower() == target_lower:
                return (field, 1.0, "exact_case_insensitive")
        
        # Strategy 3: Synonym match
        pattern = self.patterns.get(target)
        if pattern:
            for field in available_fields:
                if pattern.match(field):
                    return (field, 0.95, "synonym")
        
        # Strategy 4: Fuzzy match (substring + similarity)
        best_match = None
        best_score = 0.0
        
        for field in available_fields:
            # Check if target is substring of field (or vice versa)
            if target_lower in field.lower() or field.lower() in target_lower:
                score = max(
                    self._similarity_score(target, field),
                    0.7  # Bonus for substring match
                )
                if score > best_score:
                    best_score = score
                    best_match = field
            else:
                # Pure similarity
                score = self._similarity_score(target, field)
                if score > best_score and score >= min_confidence:
                    best_score = score
                    best_match = field
        
        if best_match and best_score >= min_confidence:
            return (best_match, best_score, "fuzzy")
        
        return None
    
    def find_all_fields(
        self, 
        available_fields: List[str],
        targets: List[str],
        min_confidence: float = 0.6
    ) -> Dict[str, Optional[Tuple[str, float, str]]]:
        """
        Find all target fields in available fields.
        
        Returns:
            {target: (matched_field, confidence, match_type) or None}
        """
        return {
            target: self.find_field(available_fields, target, min_confidence)
            for target in targets
        }
