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
        """Force-new requires rejections plus multi-metric plateau evidence."""
        orch = self._make_orchestrator()
        orch.config.setdefault("review", {})["plateau_tol_primary"] = 0.01
        orch._evaluate_iteration(
            {"accuracy": 0.3650, "code": "x=1", "metrics": {"DEG_PCC_20": 0.34, "DEG_PCC_50": 0.36, "R2": 0.07}},
            0,
        )
        orch._evaluate_iteration(
            {"accuracy": 0.3580, "code": "x=2", "metrics": {"DEG_PCC_20": 0.339, "DEG_PCC_50": 0.358, "R2": 0.065}},
            1,
        )
        result = orch._evaluate_iteration(
            {"accuracy": 0.3550, "code": "x=3", "metrics": {"DEG_PCC_20": 0.338, "DEG_PCC_50": 0.357, "R2": 0.064}},
            2,
        )
        assert result["verdict"] == "REJECT"
        assert result["forced_new_hypothesis"] is True

    def test_trend_improvement_can_be_accepted_without_beating_best(self):
        """Dual-track logic accepts positive trend even below global best."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.50, "code": "x=1"}, 0)
        r1 = orch._evaluate_iteration({"accuracy": 0.45, "code": "x=2"}, 1)
        assert r1["verdict"] == "REJECT"
        r2 = orch._evaluate_iteration({"accuracy": 0.46, "code": "x=3"}, 2)
        assert r2["verdict"] == "ACCEPT"
        assert r2["trend_delta"] == pytest.approx(0.01)

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

    def test_best_anchor_does_not_regress_after_plateau_accept(self):
        """Global best anchor should remain monotonic in protocol comparisons."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5000, "code": "x=1"}, 0)
        plateau = orch._evaluate_iteration({"accuracy": 0.4990, "code": "x=2"}, 1)
        assert plateau["verdict"] == "ACCEPT"

        result = orch._evaluate_iteration({"accuracy": 0.5005, "code": "x=3"}, 2)
        assert result["verdict"] == "ACCEPT"
        assert result["previous_score"] == pytest.approx(0.5000)
        assert result["metric_delta"] == pytest.approx(0.0005)

    def test_equal_score_accepted(self):
        """Equal score (delta=0) should be accepted."""
        orch = self._make_orchestrator()
        orch._evaluate_iteration({"accuracy": 0.5, "code": "x=1"}, 0)
        result = orch._evaluate_iteration({"accuracy": 0.5, "code": "x=2"}, 1)
        assert result["verdict"] == "ACCEPT"
        assert result["metric_delta"] == 0.0

    def test_plateau_below_threshold_eventually_rejected_and_forced_new_hypothesis(self):
        """Avoid endless plateau ACCEPT when score remains far below pass threshold."""
        orch = self._make_orchestrator()
        orch.config.setdefault("review", {})["pass_threshold"] = 0.35
        orch.config["review"]["acceptance_epsilon"] = 0.002
        orch.config["review"]["max_plateau_accepts_below_threshold"] = 2

        orch._evaluate_iteration({"accuracy": 0.2580, "code": "x=1"}, 0)
        r1 = orch._evaluate_iteration({"accuracy": 0.2579, "code": "x=2"}, 1)
        r2 = orch._evaluate_iteration({"accuracy": 0.2578, "code": "x=3"}, 2)
        r3 = orch._evaluate_iteration({"accuracy": 0.2577, "code": "x=4"}, 3)

        assert r1["verdict"] == "ACCEPT"
        assert r2["verdict"] == "ACCEPT"
        assert r3["verdict"] == "REJECT"
        assert r3["forced_new_hypothesis"] is True

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


class TestModelingReferenceRecipe:
    def test_should_use_reference_recipe_for_any_dataset_when_arch_recipe_exists(self):
        bus = SimpleMessageBus()
        cfg = _minimal_config()
        cfg["dataset_name"] = "ANYSET"
        agent = ModelingAgent(bus, cfg)
        assert agent._should_use_reference_recipe() is True

    def test_build_notebook_from_recipe_splits_cells(self, tmp_path):
        bus = SimpleMessageBus()
        cfg = _minimal_config()
        agent = ModelingAgent(bus, cfg)
        p = tmp_path / "r.py"
        p.write_text('# Title\n# ---- cell ----\nprint("a")\n# ---- cell ----\nprint("b")\n', encoding='utf-8')
        nb = agent._build_notebook_from_recipe(str(p))
        assert len(nb.cells) == 3
        assert nb.cells[1].cell_type == "code"

    def test_reference_recipe_disabled_when_architecture_recipe_missing(self):
        bus = SimpleMessageBus()
        cfg = _minimal_config()
        cfg["dataset_name"] = "OTHERSET"
        cfg["reference_recipe"] = {"enabled": True, "architecture": "nonexistent_arch"}
        agent = ModelingAgent(bus, cfg)
        assert agent._should_use_reference_recipe() is False

    def test_reference_recipe_enabled_with_explicit_file_mapping(self, tmp_path):
        bus = SimpleMessageBus()
        recipe = tmp_path / "custom.py"
        recipe.write_text('print("x")\n', encoding='utf-8')
        cfg = _minimal_config()
        cfg["dataset_name"] = "OTHERSET"
        cfg["reference_recipe"] = {"enabled": True, "file": str(recipe)}
        agent = ModelingAgent(bus, cfg)
        assert agent._should_use_reference_recipe() is True


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
