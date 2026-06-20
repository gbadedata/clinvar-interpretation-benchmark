"""Tests for the scoring engine.

The key idea: use the controlled-accuracy MockInterpreter to drive the
scorer with a KNOWN expected outcome, so we are testing that the evaluator
itself computes correct metrics, not just that it runs.
"""

from __future__ import annotations

from src.benchmark.scorer import score
from src.data_loader import get_task_set
from src.data_models import Classification, InterpretationResult, Tier, Variant
from src.interpreter import MockInterpreter


def _pairs(accuracy=None, seed=42):
    task = get_task_set(use_synthetic=True)
    mock = MockInterpreter(seed=seed, accuracy=accuracy)
    return [(v, mock.interpret(v)) for v in task]


def _make_pair(oracle: Classification, pred: Classification, tier=Tier.EASY):
    v = Variant(
        variant_id="t", gene="G", hgvs_c="c.1A>T", hgvs_p="p.X",
        consequence="snv", condition="cond",
        oracle_classification=oracle, tier=tier,
    )
    r = InterpretationResult(variant_id="t", classification=pred)
    return (v, r)


class TestPerfectAndZeroAccuracy:
    def test_perfect_accuracy_scores_one(self) -> None:
        report = score(_pairs(accuracy=1.0))
        assert report.accuracy == 1.0

    def test_perfect_accuracy_kappa_one(self) -> None:
        report = score(_pairs(accuracy=1.0))
        assert report.cohen_kappa == 1.0

    def test_zero_accuracy_scores_zero(self) -> None:
        report = score(_pairs(accuracy=0.0))
        assert report.accuracy == 0.0


class TestConfusionMatrix:
    def test_diagonal_counts_correct(self) -> None:
        pairs = [
            _make_pair(Classification.PATHOGENIC, Classification.PATHOGENIC),
            _make_pair(Classification.BENIGN, Classification.BENIGN),
            _make_pair(Classification.VUS, Classification.VUS),
        ]
        report = score(pairs)
        assert report.confusion["pathogenic"]["pathogenic"] == 1
        assert report.confusion["benign"]["benign"] == 1
        assert report.confusion["vus"]["vus"] == 1
        assert report.accuracy == 1.0

    def test_off_diagonal_counts_correct(self) -> None:
        pairs = [
            _make_pair(Classification.PATHOGENIC, Classification.BENIGN),
            _make_pair(Classification.BENIGN, Classification.PATHOGENIC),
        ]
        report = score(pairs)
        assert report.confusion["pathogenic"]["benign"] == 1
        assert report.confusion["benign"]["pathogenic"] == 1
        assert report.accuracy == 0.0


class TestPerClassMetrics:
    def test_precision_recall_f1_known_case(self) -> None:
        # 2 pathogenic truth, both predicted pathogenic -> recall 1.0
        # 1 benign truth predicted pathogenic -> precision 2/3
        pairs = [
            _make_pair(Classification.PATHOGENIC, Classification.PATHOGENIC),
            _make_pair(Classification.PATHOGENIC, Classification.PATHOGENIC),
            _make_pair(Classification.BENIGN, Classification.PATHOGENIC),
        ]
        report = score(pairs)
        path = next(m for m in report.per_class if m.label == "pathogenic")
        assert path.recall == 1.0
        assert abs(path.precision - 2 / 3) < 1e-9
        assert path.support == 2

    def test_support_sums_to_n(self) -> None:
        report = score(_pairs(accuracy=0.5))
        total_support = sum(m.support for m in report.per_class)
        assert total_support == report.n


class TestTierBreakdown:
    def test_tier_accuracy_computed(self) -> None:
        pairs = [
            _make_pair(Classification.PATHOGENIC, Classification.PATHOGENIC, tier=Tier.EASY),
            _make_pair(Classification.BENIGN, Classification.VUS, tier=Tier.HARD),
        ]
        report = score(pairs)
        assert report.tier_accuracy["easy"] == 1.0
        assert report.tier_accuracy["hard"] == 0.0

    def test_tier_support_counts(self) -> None:
        report = score(_pairs(accuracy=0.5))
        assert sum(report.tier_support.values()) == report.n


class TestReportSerialisation:
    def test_to_dict_is_json_safe(self) -> None:
        import json
        report = score(_pairs(accuracy=0.7))
        d = report.to_dict()
        json.dumps(d)  # must not raise
        assert "accuracy" in d
        assert "confusion" in d
        assert "per_class" in d
