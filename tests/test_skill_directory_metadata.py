from __future__ import annotations

import os

from cellscientist.registry.skill_registry import SkillRegistry
from cellscientist.skills.drug_analysis import DrugAnalysisSkill
from cellscientist.skills.notebook_workflow import NotebookWorkflowSkill


def test_skill_metadata_exposes_drug_analysis_skill_package_mapping():
    metadata = DrugAnalysisSkill().skill_metadata()
    package = metadata["skill_package"]

    assert metadata["name"] == "drug-analysis"
    assert package["directory_name"] == "drug-analysis"
    assert package["has_skill_package"] is True
    assert package["has_skill_doc"] is True
    assert package["directory_path"].endswith("/skills/drug-analysis")
    assert package["skill_doc_path"].endswith("/skills/drug-analysis/SKILL.md")
    assert os.path.exists(package["skill_doc_path"])


def test_skill_metadata_exposes_notebook_workflow_skill_package_mapping():
    metadata = NotebookWorkflowSkill().skill_metadata()
    package = metadata["skill_package"]

    assert metadata["name"] == "notebook-workflow"
    assert package["directory_name"] == "notebook-workflow"
    assert package["has_skill_package"] is True
    assert package["has_skill_doc"] is True
    assert package["directory_path"].endswith("/skills/notebook-workflow")
    assert package["skill_doc_path"].endswith("/skills/notebook-workflow/SKILL.md")
    assert os.path.exists(package["skill_doc_path"])


def test_registry_catalog_exposes_skill_package_metadata_without_runtime_loader():
    registry = SkillRegistry([DrugAnalysisSkill(), NotebookWorkflowSkill()])
    catalog = {item["name"]: item for item in registry.skill_catalog()}

    assert catalog["drug-analysis"]["skill_package"]["has_skill_package"] is True
    assert catalog["notebook-workflow"]["skill_package"]["has_skill_doc"] is True


def test_registry_suggestions_include_skill_package_metadata():
    registry = SkillRegistry([DrugAnalysisSkill(), NotebookWorkflowSkill()])
    suggestions = registry.suggest_skills(limit=5)
    by_name = {item["name"]: item for item in suggestions}

    assert by_name["drug-analysis"]["skill_package"]["directory_name"] == "drug-analysis"
    assert by_name["notebook-workflow"]["skill_package"]["has_skill_doc"] is True
