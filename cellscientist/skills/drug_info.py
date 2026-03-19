from __future__ import annotations

from ..runtime.state import Artifact, SessionState
from ..tools.drug_lookup import find_drug_name_in_text, lookup_drug_profile
from .base import BaseSkill


class DrugInfoSkill(BaseSkill):
    """Minimal structured drug information skill."""

    name = "drug-info"
    supported_task_types = ["drug_info"]

    def run(self, state: SessionState) -> dict[str, object]:
        drug_name = self._extract_drug_name(state.user_query)
        profile = lookup_drug_profile(drug_name)
        evidence = profile.get("evidence", [])

        for item in evidence:
            evidence_id = item.get("id")
            if evidence_id and evidence_id not in state.evidence_ids:
                state.evidence_ids.append(str(evidence_id))

        result = {
            "task": "drug_info",
            "drug_name": profile["drug_name"],
            "summary": profile["summary"],
            "targets": profile["targets"],
            "indications": profile["indications"],
            "adverse_effects": profile["adverse_effects"],
            "evidence": evidence,
        }

        state.artifacts.append(
            Artifact(
                type="drug_profile",
                name=f"{profile['drug_name']}_profile",
                content=result,
                metadata={
                    "skill": self.name,
                    "query": state.user_query,
                },
            )
        )
        return result

    @staticmethod
    def _extract_drug_name(query: str) -> str:
        return find_drug_name_in_text(query)
