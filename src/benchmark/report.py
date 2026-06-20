"""Unified benchmark report: combines oracle scoring and validators.

This is the top-level result object. It holds the oracle-agreement score
(scorer.py) and the biological-validator summary (validators.py) together,
plus metadata about the run, and serialises to a single JSON artefact.

The report is what answers Latch's question in one place: how accurately
did the model classify variants (per tier and per class), and how often did
its reasoning stand up to biological scrutiny (gene grounding, consequence
consistency, mechanism, no fabrication)?
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.benchmark.scorer import ScoreReport, score
from src.benchmark.validators import ValidatorSummary, summarise_validators
from src.data_models import InterpretationResult, Variant

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Full benchmark result: oracle scoring + validator summary."""

    model_name: str
    n_variants: int
    score_report: ScoreReport
    validator_summary: ValidatorSummary

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_variants": self.n_variants,
            "oracle_agreement": self.score_report.to_dict(),
            "biological_validators": self.validator_summary.to_dict(),
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("benchmark_report_written: %s", path)
        return path


def build_report(
    model_name: str,
    pairs: list[tuple[Variant, InterpretationResult]],
) -> BenchmarkReport:
    """Combine scoring and validation into one report.

    Args:
        model_name: identifier for the model being evaluated.
        pairs: list of (Variant, InterpretationResult).

    Returns:
        A BenchmarkReport.
    """
    score_report = score(pairs)
    validator_summary = summarise_validators(pairs)
    report = BenchmarkReport(
        model_name=model_name,
        n_variants=len(pairs),
        score_report=score_report,
        validator_summary=validator_summary,
    )
    logger.info(
        "report_built: model=%s n=%d accuracy=%.3f",
        model_name, len(pairs), score_report.accuracy,
    )
    return report


def print_summary(report: BenchmarkReport) -> None:
    """Print a human-readable benchmark summary to stdout."""
    s = report.score_report
    v = report.validator_summary
    print("\n" + "=" * 66)
    print("CLINVAR VARIANT INTERPRETATION BENCHMARK")
    print("=" * 66)
    print(f"  Model:     {report.model_name}")
    print(f"  Variants:  {report.n_variants}")
    print()
    print("  Oracle agreement (vs ClinVar expert consensus)")
    print(f"    Overall accuracy:  {s.accuracy:.3f}")
    print(f"    Cohen's kappa:     {s.cohen_kappa:.3f}")
    print(f"    Macro F1:          {s.macro_f1:.3f}")
    print()
    print("    By difficulty tier:")
    for tier in ("easy", "medium", "hard"):
        if tier in s.tier_accuracy:
            print(f"      {tier:8s} acc={s.tier_accuracy[tier]:.3f} "
                  f"(n={s.tier_support[tier]})")
    print()
    print("    By class (precision / recall / F1):")
    for m in s.per_class:
        print(f"      {m.label:11s} {m.precision:.3f} / {m.recall:.3f} / "
              f"{m.f1:.3f}  (n={m.support})")
    print()
    print("  Biological validators (reasoning quality)")
    for name, rate in v.pass_rates.items():
        applicable = v.applicable[name]
        print(f"    {name:26s} {rate:.3f}  (n={applicable})")
    print("=" * 66)
