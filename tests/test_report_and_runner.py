"""Tests for the unified report and the benchmark runner (mock path only)."""

from __future__ import annotations

import json

from src.benchmark.report import build_report
from src.data_loader import get_task_set
from src.interpreter import MockInterpreter
from src.run_benchmark import run_benchmark


def _pairs(accuracy=None):
    task = get_task_set(use_synthetic=True)
    mock = MockInterpreter(accuracy=accuracy)
    return [(v, mock.interpret(v)) for v in task]


class TestBuildReport:
    def test_report_has_both_layers(self) -> None:
        report = build_report("mock", _pairs(accuracy=0.7))
        d = report.to_dict()
        assert "oracle_agreement" in d
        assert "biological_validators" in d

    def test_perfect_mock_scores_one(self) -> None:
        report = build_report("mock", _pairs(accuracy=1.0))
        assert report.score_report.accuracy == 1.0

    def test_report_json_serialisable(self) -> None:
        report = build_report("mock", _pairs(accuracy=0.5))
        json.dumps(report.to_dict())  # must not raise

    def test_report_writes_file(self, tmp_path) -> None:
        report = build_report("mock", _pairs(accuracy=0.6))
        out = tmp_path / "r.json"
        report.write(out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["model_name"] == "mock"


class TestRunner:
    def test_runner_end_to_end(self) -> None:
        task = get_task_set(use_synthetic=True)
        report = run_benchmark(MockInterpreter(), "mock", task)
        assert report.n_variants == len(task)

    def test_runner_perfect_accuracy(self) -> None:
        task = get_task_set(use_synthetic=True)
        report = run_benchmark(MockInterpreter(accuracy=1.0), "mock", task)
        assert report.score_report.accuracy == 1.0
        # Perfect classification also means gene grounding passes (mock echoes gene)
        assert report.validator_summary.pass_rates["gene_grounding"] == 1.0
