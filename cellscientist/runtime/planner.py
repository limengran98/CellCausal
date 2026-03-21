from __future__ import annotations

import re
from typing import List, Sequence

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

_NOTEBOOK_ANCHOR_KEYWORDS = (
    "notebook",
    "modeling",
    "model",
    "experiment design",
    "pipeline",
    "\u5efa\u6a21",
    "\u5b9e\u9a8c\u8bbe\u8ba1",
    "\u6d41\u7a0b",
)

_NOTEBOOK_CONTEXT_KEYWORDS = (
    "review",
    "autofix",
    "execute",
    "retrieval refresh",
    "refresh evidence",
    "run notebook",
    "run this notebook",
    "execute this notebook",
    "fix notebook",
    "repair notebook",
    "check notebook",
    "optimize notebook",
    "\u6267\u884c",
    "\u8fd0\u884c",
    "\u5ba1\u67e5",
    "\u68c0\u67e5",
    "\u4f18\u5316",
    "\u91cd\u65b0\u5ba1\u67e5",
    "\u91cd\u65b0\u68c0\u67e5",
    "\u4fee\u590d",
    "\u62a5\u9519",
    "\u6548\u679c\u4e00\u822c",
    "\u7ed3\u679c\u4e0d\u597d",
    "挖掘生物知识",
    "补充生物学证据",
    "补充证据",
    "检索更多信息",
)

_GENERATE_PATTERNS = (
    "help me generate",
    "generate notebook",
    "create notebook",
    "design notebook",
    "build notebook",
    "\u5e2e\u6211\u751f\u6210",
    "\u751f\u6210\u4e00\u4e2a",
    "\u751f\u6210 notebook",
    "\u8bbe\u8ba1 notebook",
    "\u5b9e\u9a8c\u8bbe\u8ba1",
)

_REVIEW_PATTERNS = (
    "review",
    "re-review",
    "check notebook",
    "inspect notebook",
    "optimize notebook",
    "\u91cd\u65b0\u5ba1\u67e5",
    "\u91cd\u65b0\u68c0\u67e5",
    "\u5ba1\u67e5",
    "\u68c0\u67e5 notebook",
    "\u770b\u770b notebook \u8d28\u91cf",
    "\u6548\u679c\u4e00\u822c",
    "\u7ed3\u679c\u4e0d\u597d",
    "\u91cd\u65b0\u5206\u6790",
    "挖掘生物知识",
)

_EXECUTE_PATTERNS = (
    "then execute",
    "then run",
    "execute this notebook",
    "execute notebook",
    "run this notebook",
    "run notebook",
    "\u7136\u540e\u6267\u884c",
    "\u518d\u6267\u884c",
    "\u6267\u884c\u8fd9\u4e2a notebook",
    "\u8fd0\u884c notebook",
    "\u8fd0\u884c\u8fd9\u4e2a notebook",
    "\u6267\u884c",
)

_AUTOFIX_PATTERNS = (
    "autofix",
    "fix notebook",
    "fix this notebook",
    "repair notebook",
    "notebook error",
    "\u4fee notebook \u62a5\u9519",
    "\u4fee\u590d\u8fd9\u4e2a notebook",
    "\u4fee\u590d notebook",
    "\u6267\u884c\u62a5\u9519",
    "\u8fd0\u884c\u62a5\u9519",
    "\u62a5\u9519\u4e86",
    "\u4fee\u590d",
)

_RETRIEVAL_REFRESH_PATTERNS = (
    "retrieval refresh",
    "refresh evidence",
    "refresh biology evidence",
    "refresh biological evidence",
    "retrieve more evidence",
    "retrieve more information",
    "retrieve more pathway",
    "retrieve more target",
    "retrieve more drug",
    "mine more biology knowledge",
    "supplement evidence",
    "supplement biological evidence",
    "挖掘生物知识",
    "补充生物学证据",
    "补充生物证据",
    "补充证据",
    "检索更多信息",
    "检索更多生物学信息",
    "检索更多药物",
    "检索更多通路",
    "检索更多靶点",
    "补充药物",
    "补充通路",
    "补充靶点",
)

_FAILURE_CONTEXT_KEYWORDS = (
    "error",
    "failed",
    "failure",
    "\u62a5\u9519",
    "\u5931\u8d25",
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

_ACTION_PATTERNS: tuple[tuple[str, Sequence[str]], ...] = (
    ("generate", _GENERATE_PATTERNS),
    ("review", _REVIEW_PATTERNS),
    ("execute", _EXECUTE_PATTERNS),
    ("autofix", _AUTOFIX_PATTERNS),
)

_ACTION_ORDER = {
    "generate": 0,
    "retrieval_refresh": 1,
    "review": 2,
    "execute": 3,
    "autofix": 4,
}


class Planner:
    """Planner-first intent parser for the minimal runtime skeleton.

    This keeps the runtime repo-native and lightweight while moving one step
    closer to the "understand task -> decompose actions -> execute" discipline
    used by planner-first skill systems.
    """

    def build_intent(self, query: str) -> ResearchIntent:
        """Classify the query into a coarse task type."""

        normalized_query = query.strip()
        requested_actions = _extract_requested_actions(normalized_query)
        task_type, secondary_hints = self._classify_task_type(
            normalized_query,
            requested_actions=requested_actions,
        )
        mode = "legacy" if task_type == "legacy_notebook" else "default"

        return ResearchIntent(
            raw_query=normalized_query,
            task_type=task_type,
            requested_actions=requested_actions,
            secondary_task_hints=secondary_hints,
            entities=_extract_entities(normalized_query),
            constraints=_extract_constraints(normalized_query),
            mode=mode,
        )

    @staticmethod
    def _classify_task_type(query: str, *, requested_actions: Sequence[str]) -> tuple[TaskType, List[str]]:
        lowered = query.lower()
        notebook_signal = _looks_like_notebook_query(lowered, requested_actions)
        drug_signal = any(keyword in lowered for keyword in _DRUG_KEYWORDS)

        secondary_hints: List[str] = []
        if notebook_signal:
            if drug_signal:
                secondary_hints.append("drug_info")
            return "legacy_notebook", secondary_hints
        if drug_signal:
            return "drug_info", secondary_hints
        return "unknown", secondary_hints


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


def _extract_requested_actions(query: str) -> List[str]:
    lowered = query.lower()
    ordered_matches: List[tuple[int, int, str]] = []

    for priority, (action, patterns) in enumerate(_ACTION_PATTERNS):
        positions = [lowered.find(pattern) for pattern in patterns if pattern in lowered]
        positions = [position for position in positions if position >= 0]
        if positions:
            ordered_matches.append((min(positions), priority, action))

    ordered_matches.sort()
    requested_actions: List[str] = []
    for _, _, action in ordered_matches:
        if action not in requested_actions:
            requested_actions.append(action)

    if _should_add_retrieval_refresh(lowered, requested_actions) and "retrieval_refresh" not in requested_actions:
        requested_actions.append("retrieval_refresh")

    if "autofix" in requested_actions and any(keyword in lowered for keyword in _FAILURE_CONTEXT_KEYWORDS):
        requested_actions = [action for action in requested_actions if action != "execute"]

    if "review" in requested_actions and any(
        keyword in lowered for keyword in ("\u6548\u679c\u4e00\u822c", "\u7ed3\u679c\u4e0d\u597d", "\u91cd\u65b0\u5ba1\u67e5", "\u91cd\u65b0\u68c0\u67e5")
    ):
        requested_actions = [action for action in requested_actions if action != "generate"]

    requested_actions = _normalize_action_sequence(requested_actions)

    if not requested_actions and _looks_like_notebook_query(lowered, requested_actions):
        return ["generate"]

    return requested_actions


def _looks_like_notebook_query(query: str, requested_actions: Sequence[str]) -> bool:
    if requested_actions:
        return True
    if any(keyword in query for keyword in _NOTEBOOK_ANCHOR_KEYWORDS):
        return True
    if any(keyword in query for keyword in _NOTEBOOK_CONTEXT_KEYWORDS):
        return True
    return False


def _should_add_retrieval_refresh(query: str, requested_actions: Sequence[str]) -> bool:
    if not any(pattern in query for pattern in _RETRIEVAL_REFRESH_PATTERNS):
        return False

    notebook_followup_keywords = (
        "review",
        "execute",
        "notebook",
        "\u5ba1\u67e5",
        "\u91cd\u65b0\u5ba1\u67e5",
        "\u91cd\u65b0\u68c0\u67e5",
        "\u6267\u884c",
        "\u8fd0\u884c",
        "\u6548\u679c\u4e00\u822c",
        "\u7ed3\u679c\u4e0d\u597d",
    )
    if any(action in requested_actions for action in ("review", "execute")):
        return True
    return any(keyword in query for keyword in notebook_followup_keywords)


def _normalize_action_sequence(actions: Sequence[str]) -> List[str]:
    seen = set()
    unique_actions = []
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        unique_actions.append(action)

    return sorted(unique_actions, key=lambda action: _ACTION_ORDER.get(action, 99))
