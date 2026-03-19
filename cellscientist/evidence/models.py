from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EvidenceItem:
    """Minimal evidence record attached to downstream claims."""

    id: str
    source: str
    claim: str
    citation: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
