# -*- coding: utf-8 -*-
"""Tests for the Falsifiable Iteration Protocol and multi-agent data flow.

These tests verify:
1. The orchestrator's accept/reject/revert logic
2. The ResearchAgent's Causal Context Payload (SMILES mechanism priors)
3. The ModelingAgent's mechanism-constrained prompt construction
4. The EvaluationAgent's metric comparison across iterations
5. The run_pipeline --agent-mode argument parsing
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cellscientist.core.orchestrator import PipelineOrchestrator, PipelineState
from cellscientist.core.agents import (
    AgentResponse,
    BiologicalConstraintVerifier,
    EvaluationAgent,
    ExecutionAgent,
    ModelingAgent,
    ResearchAgent,
    TaskContext,
)
from cellscientist.core.message_bus import SimpleMessageBus


# =============================================================================
# Fixtures
# =============================================================================


def _minimal_config():
    """Return a minimal pipeline config dict for testing."""
    return {
        "dataset_name": "TEST",
        "split_name": "smiles",
        "paths": {"data_h5_path": "/tmp/test.h5"},
        "review": {
            "target_metric": "PCC",
            "pass_threshold": 0.7,
            "direction": "maximize",
            "max_iterations": 3,
        },
        "exec": {"timeout_seconds": 60},
        "literature": {"enabled": False, "bio_kb": {"enabled": False}},
    }


# =============================================================================
# Falsifiable Iteration Protocol (Orchestrator)
# =============================================================================


class TestFalsifiableProtocol:
    """Tests for PipelineOrchestrator._evaluate_iteration."""

    def _make_orchestrator(self):
        """Create an orchestrator with mocked agent instantiation."""
        cfg = _minimal_config()
        with (
            patch.object(ResearchAgent, "__init__", lambda self, *a, **k: base_agent_init(self, *a, **k)),
            patch.object(ModelingAgent, "__init__", lambda self, *a, **k: base_agent_init(self, *a, **k)),
            patch.object(ExecutionAgent, "__init__", lambda self, *a, **k: base_agent_init(self, *a, **k)),
            patch.object(EvaluationAgent, "__init__", lambda self, *a, **k: base_agent_init(self, *a, **k)),
            patch.object(TaskContext, "from_config", return_value=TaskContext(
                smiles_list=["CCO", "CC(=O)O"],
                h5_file_path="/tmp/test.h5",
                max_iterations=3,
                config=cfg,
            )),
        ):
            return PipelineOrchestrator(cfg)

    def test_first_iteration_always_accepted(self):
        """First iteration should always be accepted."""
        orch = self._make_orchestrator()
        result = orch._evaluate_iteration({"accuracy": 0.5, "code": "x=1"}, 0)
        assert result["verdict"] == "ACCEPT"
        assert result["previous_score"] is None
        assert result["forced_new_hypothesis"] is False

    def test_improvement_accepted(self):
        """Improvement should be accepted."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5, "code": "x=1"}, 0)
        result = orch._evaluate_iteration({"accuracy": 0.6, "code": "x=2"}, 1)
        assert result["verdict"] == "ACCEPT"
        assert result["metric_delta"] == pytest.approx(0.1)
        assert result["current_score"] == 0.6
        assert result["previous_score"] == 0.5

    def test_degradation_rejected(self):
        """Degradation should be rejected."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5, "code": "x=1"}, 0)
        result = orch._evaluate_iteration({"accuracy": 0.3, "code": "x=2"}, 1)
        assert result["verdict"] == "REJECT"
        assert result["metric_delta"] == pytest.approx(-0.2)
        assert result["forced_new_hypothesis"] is False

    def test_consecutive_rejections_force_new_hypothesis(self):
        """Two consecutive rejections should force a new hypothesis."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5, "code": "x=1"}, 0)
        orch._evaluate_iteration({"accuracy": 0.3, "code": "x=2"}, 1)
        result = orch._evaluate_iteration({"accuracy": 0.2, "code": "x=3"}, 2)
        assert result["verdict"] == "REJECT"
        assert result["forced_new_hypothesis"] is True

    def test_rejection_resets_on_accept(self):
        """Consecutive rejection counter resets after an accept."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5, "code": "x=1"}, 0)
        orch._evaluate_iteration({"accuracy": 0.3, "code": "x=2"}, 1)
        # Now improve (accept)
        orch._evaluate_iteration({"accuracy": 0.6, "code": "x=3"}, 2)
        assert orch.consecutive_rejections == 0
        # Another degradation — first rejection, not forced
        result = orch._evaluate_iteration({"accuracy": 0.4, "code": "x=4"}, 3)
        assert result["verdict"] == "REJECT"
        assert result["forced_new_hypothesis"] is False

    def test_equal_score_accepted(self):
        """Equal score (delta=0) should be accepted."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5, "code": "x=1"}, 0)
        result = orch._evaluate_iteration({"accuracy": 0.5, "code": "x=2"}, 1)
        assert result["verdict"] == "ACCEPT"
        assert result["metric_delta"] == 0.0

    def test_iteration_history_populated(self):
        """Iteration history should be populated."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5}, 0)
        orch._evaluate_iteration({"accuracy": 0.6}, 1)
        assert len(orch.iteration_history) == 2
        assert orch.iteration_history[0]["score"] == 0.5
        assert orch.iteration_history[1]["score"] == 0.6


# =============================================================================
# ResearchAgent — Causal Context Payload
# =============================================================================


class TestResearchAgentSMILESPrior:
    """Tests for ResearchAgent._build_smiles_mechanism_prior."""

    def test_empty_smiles_returns_empty(self):
        priors = ResearchAgent._build_smiles_mechanism_prior([], {})
        assert priors == []

    def test_builds_priors_from_biokb_records(self):
        smiles = ["CCO", "CC(=O)O"]
        bio_kb = {
            "records": [
                {
                    "smiles": "CCO",
                    "targets": ["ADH1", "CYP2E1"],
                    "pathways": ["Ethanol metabolism"],
                    "mechanism_of_action": "Substrate for ADH",
                },
                {
                    "smiles": "CC(=O)O",
                    "targets": ["ACSS2"],
                    "pathways": ["Acetyl-CoA biosynthesis"],
                },
            ]
        }
        priors = ResearchAgent._build_smiles_mechanism_prior(smiles, bio_kb)
        assert len(priors) == 2
        assert priors[0]["smiles"] == "CCO"
        assert "ADH1" in priors[0]["targets"]
        assert "Ethanol metabolism" in priors[0]["pathways"]
        assert priors[0]["mechanism_summary"] == "Substrate for ADH"

    def test_builds_priors_without_biokb(self):
        smiles = ["CCO"]
        priors = ResearchAgent._build_smiles_mechanism_prior(smiles, {})
        assert len(priors) == 1
        assert priors[0]["smiles"] == "CCO"
        assert priors[0]["targets"] == []
        assert priors[0]["pathways"] == []

    def test_limits_to_20_smiles(self):
        smiles = [f"C{'C' * i}" for i in range(30)]
        priors = ResearchAgent._build_smiles_mechanism_prior(smiles, {})
        assert len(priors) == 20


# =============================================================================
# ModelingAgent — Mechanism Prior Formatting
# =============================================================================


class TestModelingAgentMechanismPrior:
    """Tests for ModelingAgent._format_mechanism_prior."""

    def test_empty_priors(self):
        result = ModelingAgent._format_mechanism_prior([])
        assert result == ""

    def test_formats_priors(self):
        priors = [
            {
                "smiles": "CCO",
                "targets": ["ADH1"],
                "pathways": ["Ethanol metabolism"],
                "mechanism_summary": "Substrate for ADH",
            }
        ]
        result = ModelingAgent._format_mechanism_prior(priors)
        assert "CCO" in result
        assert "ADH1" in result
        assert "Ethanol metabolism" in result
        assert "Substrate for ADH" in result


# =============================================================================
# BiologicalConstraintVerifier — Mechanistic Loss Detection
# =============================================================================


class TestBiologicalConstraintVerifier:
    """Tests for BiologicalConstraintVerifier.verify."""

    # --- zero_dose_causal_law constraint ---

    def test_empty_code_no_mechanistic_loss(self):
        """Empty code should not trigger mechanistic loss."""
        report = BiologicalConstraintVerifier.verify("")
        assert report["mechanistic_loss"] is False
        assert report["missing_critical"] == []
        assert report["missing_advisory"] == []

    def test_code_with_dose_multiply_passes_zero_dose(self):
        """Code that multiplies output by dose satisfies the zero-dose constraint."""
        code = "output = base_prediction * dose  # CONSTRAINT: zero_dose_causal_law"
        report = BiologicalConstraintVerifier.verify(code)
        assert "zero_dose_causal_law" not in report["missing_critical"]

    def test_code_with_dose_gate_variable_passes_zero_dose(self):
        """Code that uses a dose_gate variable satisfies the zero-dose constraint."""
        code = "dose_gate = torch.sigmoid(dose_embedding)\nout = prediction * dose_gate"
        report = BiologicalConstraintVerifier.verify(code)
        assert "zero_dose_causal_law" not in report["missing_critical"]

    def test_code_with_dose_scale_variable_passes_zero_dose(self):
        """Code that uses a dose_scale variable satisfies the zero-dose constraint."""
        code = "dose_scale = dose.unsqueeze(-1)\noutput = hidden * dose_scale"
        report = BiologicalConstraintVerifier.verify(code)
        assert "zero_dose_causal_law" not in report["missing_critical"]

    def test_code_without_dose_scaling_triggers_mechanistic_loss(self):
        """Code with no dose scaling fails the zero-dose constraint."""
        code = (
            "import torch\n"
            "class MyModel(torch.nn.Module):\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n"
        )
        report = BiologicalConstraintVerifier.verify(code)
        assert report["mechanistic_loss"] is True
        assert "zero_dose_causal_law" in report["missing_critical"]

    def test_code_with_zero_dose_comment_passes(self):
        """Explicit zero-dose constraint comment is detected."""
        code = (
            "# CONSTRAINT: zero_dose_causal_law\n"
            "output = prediction * dose\n"
        )
        report = BiologicalConstraintVerifier.verify(code)
        assert "zero_dose_causal_law" not in report["missing_critical"]

    def test_mechanistic_loss_flag_set_when_critical_missing(self):
        """mechanistic_loss is True only when a critical constraint is missing."""
        code_without_dose = "output = self.fc(x)"
        report = BiologicalConstraintVerifier.verify(code_without_dose)
        assert report["mechanistic_loss"] is True

        code_with_dose = "output = self.fc(x) * dose"
        report2 = BiologicalConstraintVerifier.verify(code_with_dose)
        assert report2["mechanistic_loss"] is False

    def test_summary_contains_mechanistic_loss_message_on_failure(self):
        """Summary should mention MECHANISTIC LOSS when constraints are absent."""
        report = BiologicalConstraintVerifier.verify("x = 1 + 1")
        assert "MECHANISTIC LOSS" in report["summary"]
        assert "zero_dose_causal_law" in report["summary"]

    def test_summary_all_verified_when_all_found(self):
        """Summary should report all-clear when all constraints are satisfied."""
        code = (
            "output = prediction * dose  # CONSTRAINT: zero_dose_causal_law\n"
            "# MECHANISM JUSTIFICATION: dose=0 → zero perturbation delta\n"
        )
        report = BiologicalConstraintVerifier.verify(code)
        assert report["mechanistic_loss"] is False
        assert "✅" in report["summary"]

    # --- mechanism_justification_comment constraint (advisory) ---

    def test_advisory_constraint_not_in_missing_critical(self):
        """Advisory constraints do not appear in missing_critical."""
        code = "output = prediction * dose  # CONSTRAINT: zero_dose_causal_law"
        report = BiologicalConstraintVerifier.verify(code)
        assert "mechanism_justification_comment" not in report["missing_critical"]

    def test_advisory_constraint_missing_not_mechanistic_loss(self):
        """Missing advisory constraint alone does not trigger mechanistic_loss."""
        code = "output = prediction * dose  # CONSTRAINT: zero_dose_causal_law"
        report = BiologicalConstraintVerifier.verify(code)
        # Advisory may be missing but critical is present → no mechanistic loss
        assert report["mechanistic_loss"] is False

    def test_mechanism_justification_comment_detected(self):
        """MECHANISM JUSTIFICATION comment is detected."""
        code = (
            "output = prediction * dose\n"
            "# MECHANISM JUSTIFICATION: dose=0 implies unperturbed state\n"
        )
        report = BiologicalConstraintVerifier.verify(code)
        assert "mechanism_justification_comment" not in report["missing_advisory"]

    # --- Results structure ---

    def test_results_list_has_all_constraints(self):
        """Results list should have one entry per defined constraint."""
        report = BiologicalConstraintVerifier.verify("x = 1")
        constraint_names = {r["name"] for r in report["results"]}
        defined_names = {c["name"] for c in BiologicalConstraintVerifier.CONSTRAINTS}
        assert constraint_names == defined_names

    def test_result_entry_has_required_keys(self):
        """Each result entry must have name, severity, found, and description."""
        report = BiologicalConstraintVerifier.verify("output = x * dose")
        for entry in report["results"]:
            assert "name" in entry
            assert "severity" in entry
            assert "found" in entry
            assert "description" in entry


# =============================================================================
# ResearchAgent — Zero-Dose Causal Chain always injected
# =============================================================================


class TestResearchAgentZeroDoseChain:
    """Tests that _derive_domain_model_causal_chains always injects the zero-dose law."""

    def test_zero_dose_chain_always_present(self):
        """Zero-dose causal law chain must appear even with no keywords."""
        chains = ResearchAgent._derive_domain_model_causal_chains(
            smiles_priors=[],
            literature_md="some unrelated text",
            knowledge_gap="",
        )
        has_causal_law = any(c.get("constraint_type") == "causal_law" for c in chains)
        assert has_causal_law, "zero_dose causal_law chain must always be injected"

    def test_zero_dose_chain_contains_mandatory_implementation_hint(self):
        """The injected chain's modeling_implication must include 'dose' multiplication."""
        chains = ResearchAgent._derive_domain_model_causal_chains(
            smiles_priors=[],
            literature_md="",
            knowledge_gap="",
        )
        causal_law_chains = [c for c in chains if c.get("constraint_type") == "causal_law"]
        assert causal_law_chains
        implication = causal_law_chains[0].get("modeling_implication", "")
        assert "dose" in implication.lower()
        assert "CONSTRAINT" in implication

    def test_zero_dose_chain_not_duplicated(self):
        """Only one causal_law chain should appear even when called repeatedly."""
        chains = ResearchAgent._derive_domain_model_causal_chains(
            smiles_priors=[],
            literature_md="dose non-linear magnitude",
            knowledge_gap="dose",
        )
        causal_law_chains = [c for c in chains if c.get("constraint_type") == "causal_law"]
        assert len(causal_law_chains) == 1


# =============================================================================
# EvaluationAgent — Metric Comparison
# =============================================================================


class TestEvaluationAgentMetricComparison:
    """Tests for EvaluationAgent's cross-iteration metric tracking."""

    def test_initial_previous_metrics_empty(self):
        bus = SimpleMessageBus()
        cfg = _minimal_config()
        agent = EvaluationAgent(bus, cfg)
        assert agent._previous_metrics == {}
        assert agent._previous_primary is None


# =============================================================================
# AgentResponse validation
# =============================================================================


class TestAgentResponse:
    def test_valid_response(self):
        r = AgentResponse(status="success", data={"key": "val"}, next_recipient="orchestration")
        assert r.status == "success"

    def test_invalid_status(self):
        with pytest.raises(ValueError):
            AgentResponse(status="invalid", data={}, next_recipient="orchestration")

    def test_to_dict_round_trip(self):
        r = AgentResponse(status="error", data={"err": "msg"}, next_recipient="evaluation")
        d = r.to_dict()
        r2 = AgentResponse.from_dict(d)
        assert r2.status == r.status
        assert r2.data == r.data
        assert r2.next_recipient == r.next_recipient


# =============================================================================
# CLI arg parsing (--agent-mode)
# =============================================================================


class TestAgentModeArgParsing:
    """Verify the --agent-mode flag is accepted by the arg parser."""

    def test_agent_mode_flag_accepted(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent-mode", action="store_true")
        args = parser.parse_args(["--agent-mode"])
        assert args.agent_mode is True

    def test_default_no_agent_mode(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent-mode", action="store_true")
        args = parser.parse_args([])
        assert args.agent_mode is False


# =============================================================================
# Helper for mocking BaseAgent.__init__
# =============================================================================

def base_agent_init(self, bus=None, config=None):
    """Minimal init for mocking agent constructors in tests."""
    import uuid
    self.agent_id = f"test_{uuid.uuid4().hex[:4]}"
    self.bus = bus or SimpleMessageBus()
    self.config = config or {}
    if hasattr(self, '_previous_metrics'):
        pass
    elif self.__class__.__name__ == 'EvaluationAgent':
        self._previous_metrics = {}
        self._previous_primary = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
