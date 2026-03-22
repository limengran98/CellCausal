from __future__ import annotations

from typing import List

from .base import BaseSkill
from .drug_analysis import DrugAnalysisSkill
from .drug_info import DrugInfoSkill
from .enzyme_mining import EnzymeMiningSkill
from .legacy_notebook import LegacyNotebookSkill
from .notebook_workflow import NotebookWorkflowSkill


def build_default_skills() -> List[BaseSkill]:
    """Return the minimal repo-native skills for the V2 skeleton."""

    return [
        DrugAnalysisSkill(),
        DrugInfoSkill(),
        EnzymeMiningSkill(),
        NotebookWorkflowSkill(),
        LegacyNotebookSkill(),
    ]
