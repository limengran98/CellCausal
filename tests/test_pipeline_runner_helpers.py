import json
import os
import sys
import types
from types import SimpleNamespace

from run_pipeline import _write_pipeline_cache_manifest
from cellscientist.pipeline.metrics import print_final_scoreboard
from cellscientist.core.orchestrator import _resolve_orchestrator_backend, run_orchestrator_sync
from cellscientist.core import orchestrator_langgraph


class _Logger:
    def __init__(self):
        self.lines = []

    def full_log(self, msg):
        self.lines.append(msg)


def test_write_pipeline_cache_manifest(tmp_path):
    stage_map = {
        "Experiment": {"config": "/tmp/exp.merged.json", "module": "exp.mod"},
        "Review": {"config": "/tmp/rev.merged.json", "module": "rev.mod"},
    }
    logger = _Logger()
    out = _write_pipeline_cache_manifest(stage_map, str(tmp_path), logger)
    assert os.path.exists(out)
    data = json.loads(open(out, "r", encoding="utf-8").read())
    assert data["stages"]["Experiment"]["merged_config_path"] == "/tmp/exp.merged.json"
    assert any("Pipeline cache manifest" in x for x in logger.lines)


def test_print_final_scoreboard_includes_stability_rates(capsys):
    summary = {
        "dataset": "DS",
        "stages": {
            "Experiment": {
                "attempted": 10,
                "succeeded": 6,
                "success_rate": 0.6,
                "clean_rate": 0.4,
                "bug_rate": 0.5,
                "best_metric": "PCC",
                "best_at_budget": 0.2,
            },
            "Review": {
                "attempted": 5,
                "succeeded": 3,
                "success_rate": 0.6,
                "clean_rate": 0.2,
                "bug_rate": 0.6,
            },
            "Total": {"time_sec": 12.3},
        },
    }
    print_final_scoreboard(summary, console=None)
    out = capsys.readouterr().out
    assert "clean_rate=40.0% | bug_rate=50.0%" in out
    assert "review_clean_rate=20.0% | review_bug_rate=60.0%" in out


def test_orchestrator_backend_defaults_to_native():
    assert _resolve_orchestrator_backend({}) == "native"


def test_orchestrator_backend_langgraph_requires_dependency():
    cfg = {"orchestrator": {"backend": "langgraph"}}
    try:
        run_orchestrator_sync(cfg)
        assert False, "expected RuntimeError or NotImplementedError"
    except (RuntimeError, NotImplementedError):
        assert True


def test_orchestrator_backend_langgraph_dispatch_when_dependency_present(monkeypatch):
    fake_langgraph = SimpleNamespace()
    async def _fake_run(cfg):
        return {"status": "terminated", "data": {"ok": True}}

    fake_backend_mod = SimpleNamespace(run_orchestrator_langgraph=_fake_run)
    monkeypatch.setitem(sys.modules, "langgraph", fake_langgraph)
    monkeypatch.setitem(sys.modules, "cellscientist.core.orchestrator_langgraph", fake_backend_mod)

    cfg = {"orchestrator": {"backend": "langgraph"}}
    out = run_orchestrator_sync(cfg)
    assert out.get("data", {}).get("ok") is True


def test_langgraph_routing_uses_next_step_instead_of_stale_state(monkeypatch):
    class _Resp:
        def __init__(self, data, status="ok"):
            self.data = data
            self.status = status

    calls = {"research": 0, "modeling": 0, "execution": 0, "evaluation": 0}

    class _Research:
        def __init__(self, *_a, **_k):
            pass

        async def process(self, msg):
            calls["research"] += 1
            return _Resp({"from": "research", "iteration": msg.get("iteration", 0)})

    class _Modeling:
        def __init__(self, *_a, **_k):
            pass

        async def process(self, _msg):
            calls["modeling"] += 1
            return _Resp({"from": "modeling"})

    class _Execution:
        def __init__(self, *_a, **_k):
            pass

        async def process(self, _msg):
            calls["execution"] += 1
            return _Resp({"from": "execution"})

    class _Evaluation:
        def __init__(self, *_a, **_k):
            pass

        async def process(self, _msg):
            calls["evaluation"] += 1
            if calls["evaluation"] == 1:
                # Route to modeling explicitly for the second iteration.
                return _Resp(
                    {
                        "decision": "CONTINUE",
                        "feedback_package": {"suggested_target": "modeling", "technical_feedback": "t"},
                    }
                )
            return _Resp({"decision": "SUCCESS"}, status="success")

    monkeypatch.setattr(orchestrator_langgraph, "ResearchAgent", _Research)
    monkeypatch.setattr(orchestrator_langgraph, "ModelingAgent", _Modeling)
    monkeypatch.setattr(orchestrator_langgraph, "ExecutionAgent", _Execution)
    monkeypatch.setattr(orchestrator_langgraph, "EvaluationAgent", _Evaluation)

    class _CompiledGraph:
        def __init__(self, nodes, edges, routes):
            self._nodes = nodes
            self._edges = edges
            self._routes = routes

        async def ainvoke(self, state):
            cur = "research"
            while True:
                update = await self._nodes[cur](state)
                state = {**state, **(update or {})}
                if cur in self._routes:
                    fn, mapping = self._routes[cur]
                    nxt = mapping[fn(state)]
                else:
                    nxt = self._edges[cur]
                if nxt == "__END__":
                    return state
                cur = nxt

    class _FakeStateGraph:
        def __init__(self, _state_type):
            self.nodes = {}
            self.edges = {}
            self.routes = {}

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def add_edge(self, src, dst):
            if src == "__START__":
                return
            self.edges[src] = dst

        def add_conditional_edges(self, src, fn, mapping):
            self.routes[src] = (fn, mapping)

        def compile(self):
            return _CompiledGraph(self.nodes, self.edges, self.routes)

    fake_graph_mod = types.ModuleType("langgraph.graph")
    fake_graph_mod.START = "__START__"
    fake_graph_mod.END = "__END__"
    fake_graph_mod.StateGraph = _FakeStateGraph
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_graph_mod)
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))

    out = run_orchestrator_sync({"orchestrator": {"backend": "langgraph"}, "review": {"max_iterations": 3}})
    assert out.get("decision") == "SUCCESS"
    # First loop: research->modeling->execution->evaluation; second loop should
    # jump directly to modeling (not stale research_input branch).
    assert calls["research"] == 1
    assert calls["modeling"] == 2
