from __future__ import annotations

import re
from typing import List

from .state import ResearchIntent, TaskType

_DRUG_KEYWORDS = (
    "drug",
    "drugs",
    "compound",
    "target",
    "targets",
    "indication",
    "indications",
    "adverse",
    "side effect",
    "toxicity",
    "\u836f\u7269",
    "\u9776\u70b9",
    "\u9002\u5e94\u75c7",
    "\u526f\u4f5c\u7528",
)

_LEGACY_NOTEBOOK_KEYWORDS = (
    "notebook",
    "modeling",
    "model",
    "experiment design",
    "pipeline",
    "\u5efa\u6a21",
    "\u5b9e\u9a8c\u8bbe\u8ba1",
    "\u6d41\u7a0b",
)

_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_CONSTRAINT_KEYWORDS = {
    "brief": ("brief", "summary", "\u7b80\u8981", "\u6982\u89c8"),
    "detailed": ("detailed", "detail", "\u6df1\u5165", "\u8be6\u7ec6"),
}
_STOPWORDS = {
    "drug",
    "drugs",
    "compound",
    "target",
    "targets",
    "indication",
    "indications",
    "adverse",
    "side",
    "effect",
    "toxicity",
    "notebook",
    "modeling",
    "model",
    "experiment",
    "design",
    "pipeline",
}


class Planner:
    """A minimal keyword-based planner for the new runtime skeleton."""

    def build_intent(self, query: str) -> ResearchIntent:
        """Classify the query into a coarse task type."""

        normalized_query = query.strip()
        task_type = self._classify_task_type(normalized_query)
        mode = "legacy" if task_type == "legacy_notebook" else "default"

        return ResearchIntent(
            raw_query=normalized_query,
            task_type=task_type,
            entities=_extract_entities(normalized_query),
            constraints=_extract_constraints(normalized_query),
            mode=mode,
        )

    @staticmethod
    def _classify_task_type(query: str) -> TaskType:
        lowered = query.lower()
        if any(keyword in lowered for keyword in _DRUG_KEYWORDS):
            return "drug_info"
        if any(keyword in lowered for keyword in _LEGACY_NOTEBOOK_KEYWORDS):
            return "legacy_notebook"
        return "unknown"


def build_intent(query: str) -> ResearchIntent:
    """Build a normalized intent from raw user input."""

    return Planner().build_intent(query)


def _extract_entities(query: str) -> List[str]:
    """Collect simple ASCII entities without overfitting the first draft."""

    entities: List[str] = []
    seen = set()
    for token in _ENTITY_RE.findall(query):
        lowered = token.lower()
        if lowered in _STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        entities.append(token)
    return entities


def _extract_constraints(query: str) -> List[str]:
    """Capture a few obvious request-shape constraints."""

    lowered = query.lower()
    constraints: List[str] = []
    for name, keywords in _CONSTRAINT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            constraints.append(name)
    return constraints
