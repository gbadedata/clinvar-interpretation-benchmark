"""Tests for abstention-aware scoring."""

from __future__ import annotations

from src.benchmark.abstention import score_abstention
from src.data_models import Classification, InterpretationResult, Tier, Variant


def _pair(oracle: Classification, pred: Classification):
    v = Variant(
        variant_id="t", gene="G", hgvs_c="c.1A>T", hgvs_p="p.X",
        consequence="snv", condition="cond",
        oracle_classification=oracle, tier=Tier.EASY,
    )
    r = InterpretationResult(variant_id="t", classification=pred)
    return (v, r)


class TestAbstentionScoring:
    def test_correct_call_counts(self) -> None:
        pairs = [_pair(Classification.PATHOGENIC, Classification.PATHOGENIC)]
        rep = score_abstention(pairs)
        assert rep.correct_calls == 1
        assert rep.confident_errors == 0
        assert rep.abstentions == 0

    def test_abstention_is_safe_not_error(self) -> None:
        # Benign truth, model says VUS -> abstention, NOT a confident error
        pairs = [_pair(Classification.BENIGN, Classification.VUS)]
        rep = score_abstention(pairs)
        assert rep.abstentions == 1
        assert rep.confident_errors == 0
        assert rep.safe_rate == 1.0   # no dangerous call made

    def test_confident_error_is_dangerous(self) -> None:
        # Benign truth, model says PATHOGENIC -> confident error
        pairs = [_pair(Classification.BENIGN, Classification.PATHOGENIC)]
        rep = score_abstention(pairs)
        assert rep.confident_errors == 1
        assert rep.safe_rate == 0.0

    def test_vus_truth_excluded_from_confident_analysis(self) -> None:
        # VUS truth variants are not part of the confident-truth denominator
        pairs = [_pair(Classification.VUS, Classification.VUS)]
        rep = score_abstention(pairs)
        assert rep.n_confident_truth == 0
        assert rep.safe_rate == 1.0

    def test_mixed_set_metrics(self) -> None:
        pairs = [
            _pair(Classification.PATHOGENIC, Classification.PATHOGENIC),  # correct
            _pair(Classification.BENIGN, Classification.VUS),             # abstain
            _pair(Classification.BENIGN, Classification.PATHOGENIC),      # error
            _pair(Classification.VUS, Classification.VUS),               # vus truth
        ]
        rep = score_abstention(pairs)
        assert rep.n_confident_truth == 3
        assert rep.correct_calls == 1
        assert rep.abstentions == 1
        assert rep.confident_errors == 1
        assert abs(rep.safe_rate - 2 / 3) < 1e-9       # 1 error in 3
        assert abs(rep.abstention_rate - 1 / 3) < 1e-9
        assert abs(rep.decisiveness - 2 / 3) < 1e-9     # 2 confident calls in 3

    def test_to_dict_serialisable(self) -> None:
        import json
        rep = score_abstention([_pair(Classification.PATHOGENIC, Classification.VUS)])
        json.dumps(rep.to_dict())
