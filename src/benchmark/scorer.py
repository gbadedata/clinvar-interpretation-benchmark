"""Scoring engine: oracle agreement and per-tier, per-class metrics.

This is the first of the two scoring layers (the second is the biological
validators in validators.py). It compares a model's classifications against
the ClinVar oracle and computes:

  - overall accuracy
  - per-difficulty-tier accuracy (easy / medium / hard)
  - per-class precision, recall, F1 (pathogenic / benign / vus)
  - Cohen's kappa (agreement corrected for chance)
  - a 3x3 confusion matrix

The engine is deliberately model-agnostic: it consumes a list of
(Variant, InterpretationResult) pairs and knows nothing about how the
results were produced. It is unit-tested with the controlled-accuracy mock,
so a model that copies the oracle exactly must score 1.0 and the metrics
are verified against known values.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from src.data_models import Classification, InterpretationResult, Tier, Variant

logger = logging.getLogger(__name__)

_CLASSES = [Classification.PATHOGENIC, Classification.BENIGN, Classification.VUS]


@dataclass
class ClassMetrics:
    """Precision, recall, F1, and support for one class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ScoreReport:
    """Full oracle-agreement scoring result."""

    n: int
    accuracy: float
    cohen_kappa: float
    tier_accuracy: dict[str, float]
    tier_support: dict[str, int]
    per_class: list[ClassMetrics]
    confusion: dict[str, dict[str, int]]
    macro_f1: float = field(default=0.0)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "macro_f1": round(self.macro_f1, 4),
            "tier_accuracy": {k: round(v, 4) for k, v in self.tier_accuracy.items()},
            "tier_support": self.tier_support,
            "per_class": [
                {
                    "label": m.label,
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                    "support": m.support,
                }
                for m in self.per_class
            ],
            "confusion": self.confusion,
        }


def _confusion_matrix(
    pairs: list[tuple[Variant, InterpretationResult]],
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix keyed [oracle][predicted] -> count."""
    matrix: dict[str, dict[str, int]] = {
        t.value: {p.value: 0 for p in _CLASSES} for t in _CLASSES
    }
    for variant, result in pairs:
        truth = variant.oracle_classification.value
        pred = result.classification.value
        matrix[truth][pred] += 1
    return matrix


def _accuracy(pairs: list[tuple[Variant, InterpretationResult]]) -> float:
    if not pairs:
        return 0.0
    correct = sum(
        1 for v, r in pairs if v.oracle_classification == r.classification
    )
    return correct / len(pairs)


def _cohen_kappa(matrix: dict[str, dict[str, int]], n: int) -> float:
    """Cohen's kappa from a confusion matrix.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and
    p_e is agreement expected by chance from the marginals.
    """
    if n == 0:
        return 0.0
    labels = [c.value for c in _CLASSES]
    p_o = sum(matrix[lab][lab] for lab in labels) / n

    row_tot = {lab: sum(matrix[lab].values()) for lab in labels}
    col_tot = {lab: sum(matrix[t][lab] for t in labels) for lab in labels}
    p_e = sum((row_tot[lab] / n) * (col_tot[lab] / n) for lab in labels)

    if p_e == 1.0:
        return 1.0  # degenerate: all in one cell and agreeing
    return (p_o - p_e) / (1 - p_e)


def _per_class_metrics(
    matrix: dict[str, dict[str, int]],
) -> list[ClassMetrics]:
    """Precision, recall, F1 for each class from the confusion matrix."""
    labels = [c.value for c in _CLASSES]
    out: list[ClassMetrics] = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[t][label] for t in labels if t != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        support = sum(matrix[label].values())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        out.append(ClassMetrics(label, precision, recall, f1, support))
    return out


def score(pairs: list[tuple[Variant, InterpretationResult]]) -> ScoreReport:
    """Compute the full oracle-agreement score report.

    Args:
        pairs: list of (Variant, InterpretationResult). The variant carries
            the oracle truth; the result carries the model's prediction.

    Returns:
        A ScoreReport with overall, per-tier, and per-class metrics.
    """
    n = len(pairs)
    matrix = _confusion_matrix(pairs)
    accuracy = _accuracy(pairs)
    kappa = _cohen_kappa(matrix, n)
    per_class = _per_class_metrics(matrix)
    macro_f1 = sum(m.f1 for m in per_class) / len(per_class) if per_class else 0.0

    # Per-tier accuracy
    by_tier: dict[Tier, list[tuple[Variant, InterpretationResult]]] = defaultdict(list)
    for v, r in pairs:
        by_tier[v.tier].append((v, r))
    tier_accuracy = {t.value: _accuracy(items) for t, items in by_tier.items()}
    tier_support = {t.value: len(items) for t, items in by_tier.items()}

    report = ScoreReport(
        n=n,
        accuracy=accuracy,
        cohen_kappa=kappa,
        tier_accuracy=tier_accuracy,
        tier_support=tier_support,
        per_class=per_class,
        confusion=matrix,
        macro_f1=macro_f1,
    )
    logger.info(
        "scored: n=%d accuracy=%.3f kappa=%.3f macro_f1=%.3f",
        n, accuracy, kappa, macro_f1,
    )
    return report
