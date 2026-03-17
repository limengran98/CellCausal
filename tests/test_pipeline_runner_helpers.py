import json
import os

from run_pipeline import _write_pipeline_cache_manifest
from cellscientist.pipeline.metrics import print_final_scoreboard


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
