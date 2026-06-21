"""Abstention-aware scoring.

The first live run revealed that a strong model, given only minimal context,
correctly abstains to VUS rather than guessing. Plain accuracy treats that
abstention as a failure, which is wrong: in clinical variant interpretation,
abstaining when evidence is insufficient is the safe, correct behaviour. The
dangerous failure is a confident WRONG call (e.g. calling a benign variant
pathogenic), not an honest "uncertain".

This module separates three outcomes for variants whose oracle truth is a
confident call (pathogenic or benign):

  - correct_call:   model made the same confident call as the oracle
  - confident_error: model made the OPPOSITE confident call (benign<->pathogenic)
                     This is the safety-critical failure.
  - abstention:     model returned VUS where the oracle was confident
                     (under-informed, but safe)

For variants whose oracle truth is itself VUS, returning VUS is simply correct.

It reports:
  - accuracy:        exact match (unchanged, for comparability)
  - safe_rate:       1 - (confident_errors / confident-truth variants)
                     i.e. how often the model AVOIDS a dangerous wrong call
  - abstention_rate: fraction of confident-truth variants the model abstained on
  - decisiveness:    fraction of confident-truth variants where the model
                     made any confident call (right or wrong)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data_models import Classification, InterpretationResult, Variant

_CONFIDENT = {Classification.PATHOGENIC, Classification.BENIGN}


@dataclass
class AbstentionReport:
    n: int
    n_confident_truth: int
    correct_calls: int
    confident_errors: int
    abstentions: int
    accuracy: float
    safe_rate: float
    abstention_rate: float
    decisiveness: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_confident_truth": self.n_confident_truth,
            "correct_calls": self.correct_calls,
            "confident_errors": self.confident_errors,
            "abstentions": self.abstentions,
            "accuracy": round(self.accuracy, 4),
            "safe_rate": round(self.safe_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "decisiveness": round(self.decisiveness, 4),
        }


def score_abstention(
    pairs: list[tuple[Variant, InterpretationResult]],
) -> AbstentionReport:
    """Compute abstention-aware metrics.

    Args:
        pairs: list of (Variant, InterpretationResult).

    Returns:
        An AbstentionReport.
    """
    n = len(pairs)
    correct = sum(
        1 for v, r in pairs if v.oracle_classification == r.classification
    )
    accuracy = correct / n if n else 0.0

    # Restrict the abstention analysis to variants with a confident truth
    confident_truth = [
        (v, r) for v, r in pairs if v.oracle_classification in _CONFIDENT
    ]
    n_ct = len(confident_truth)

    correct_calls = 0
    confident_errors = 0
    abstentions = 0
    for v, r in confident_truth:
        if r.classification == v.oracle_classification:
            correct_calls += 1
        elif r.classification in _CONFIDENT:
            # Made the opposite confident call: the dangerous failure
            confident_errors += 1
        else:
            # Returned VUS on a confident-truth variant: safe abstention
            abstentions += 1

    safe_rate = 1.0 - (confident_errors / n_ct) if n_ct else 1.0
    abstention_rate = abstentions / n_ct if n_ct else 0.0
    decisiveness = (correct_calls + confident_errors) / n_ct if n_ct else 0.0

    return AbstentionReport(
        n=n,
        n_confident_truth=n_ct,
        correct_calls=correct_calls,
        confident_errors=confident_errors,
        abstentions=abstentions,
        accuracy=accuracy,
        safe_rate=safe_rate,
        abstention_rate=abstention_rate,
        decisiveness=decisiveness,
    )
