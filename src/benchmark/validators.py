"""Biological validators: checking the model's reasoning, not just its label.

The scorer (scorer.py) asks "did the model get the right classification?".
These validators ask the deeper question Latch's verification loop requires:
"did the model get the right answer for the right biological reason, and did
it avoid fabricating evidence?"

A model can output the correct label for entirely wrong reasons. In a
clinical or target-discovery loop that is dangerous, because the label will
not generalise and the reasoning cannot be trusted. Only reasoning-level
checks catch this. Each validator runs independently of the oracle and
returns a pass/fail with a reason, so failures are auditable.

The four validators:

  1. gene_grounding        -- did the model name the gene actually carrying
                              the variant? A model that loses track of the
                              gene is not grounded in the input.
  2. consequence_consistency -- does the model's stated molecular consequence
                              match the variant's actual consequence? Calling
                              a frameshift a "missense" is a biology error
                              even if the final label is right.
  3. mechanism_plausibility -- for a loss-of-function consequence (frameshift,
                              nonsense, canonical splice) classified as
                              pathogenic, does the reasoning reflect a
                              loss-of-function mechanism? Catches right-label
                              wrong-mechanism reasoning.
  4. no_fabrication        -- did the model avoid inventing specific evidence
                              (named studies, exact allele frequencies,
                              database counts) that was never provided to it?
                              Fabricated evidence is the most dangerous
                              failure mode in a clinical loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.data_models import Classification, InterpretationResult, Variant

logger = logging.getLogger(__name__)


@dataclass
class ValidatorResult:
    """Outcome of one validator on one variant."""

    name: str
    passed: bool
    reason: str


# Consequence vocabulary: map many surface forms to a canonical category.
_LOF_CONSEQUENCES = {"frameshift", "nonsense", "stop gained", "stop_gained",
                     "splice", "splice donor", "splice acceptor", "start lost"}
_CANONICAL = {
    "frameshift": "frameshift",
    "fs": "frameshift",
    "nonsense": "nonsense",
    "stop gained": "nonsense",
    "stop_gained": "nonsense",
    "missense": "missense",
    "synonymous": "synonymous",
    "silent": "synonymous",
    "splice": "splice",
    "in-frame": "inframe",
    "inframe": "inframe",
    "deletion": "deletion",
}


def _canonical_consequence(text: str) -> str:
    """Reduce a free-text consequence to a canonical token."""
    t = text.strip().lower()
    for key, val in _CANONICAL.items():
        if key in t:
            return val
    return t


def _variant_true_consequence(variant: Variant) -> str:
    """Infer the variant's actual consequence from its HGVS and type."""
    blob = f"{variant.hgvs_p} {variant.hgvs_c} {variant.consequence}".lower()
    if "fs" in blob or "frameshift" in blob:
        return "frameshift"
    if "ter" in blob or "*" in blob or "stop" in blob:
        return "nonsense"
    if "=" in variant.hgvs_p or "synonymous" in blob or "silent" in blob:
        return "synonymous"
    if "del" in blob and "fs" not in blob:
        return "deletion"
    if "+" in variant.hgvs_c or "-" in variant.hgvs_c or "splice" in blob:
        return "splice"
    return "missense"


def validate_gene_grounding(
    variant: Variant, result: InterpretationResult
) -> ValidatorResult:
    """Did the model name the gene actually carrying the variant?"""
    expected = variant.gene.strip().upper()
    stated = result.stated_gene.strip().upper()
    passed = bool(stated) and stated == expected
    reason = (
        "gene matches" if passed
        else f"expected {expected!r}, model stated {stated!r}"
    )
    return ValidatorResult("gene_grounding", passed, reason)


def validate_consequence_consistency(
    variant: Variant, result: InterpretationResult
) -> ValidatorResult:
    """Does the model's stated consequence match the variant's actual one?"""
    truth = _variant_true_consequence(variant)
    stated = _canonical_consequence(result.stated_consequence)
    # frameshift and deletion both describe a del; accept either direction
    compatible = {
        ("frameshift", "deletion"), ("deletion", "frameshift"),
        ("nonsense", "frameshift"), ("frameshift", "nonsense"),
    }
    passed = stated == truth or (truth, stated) in compatible
    reason = (
        f"consequence consistent ({truth})" if passed
        else f"variant is {truth!r}, model stated {stated!r}"
    )
    return ValidatorResult("consequence_consistency", passed, reason)


def validate_mechanism_plausibility(
    variant: Variant, result: InterpretationResult
) -> ValidatorResult:
    """For a LoF variant called pathogenic, is the mechanism LoF?

    Only applies when (a) the variant's true consequence is loss-of-function
    and (b) the model called it pathogenic. Otherwise the validator is not
    applicable and passes by default (recorded as such).
    """
    truth = _variant_true_consequence(variant)
    is_lof = truth in {"frameshift", "nonsense", "splice"}
    if not (is_lof and result.classification == Classification.PATHOGENIC):
        return ValidatorResult(
            "mechanism_plausibility", True, "not applicable"
        )

    mech = result.stated_mechanism.lower()
    reasoning = result.reasoning.lower()
    signals = ["loss-of-function", "loss of function", "lof", "haploinsufficien",
               "truncat", "nonsense", "frameshift", "premature stop"]
    passed = any(s in mech or s in reasoning for s in signals)
    reason = (
        "LoF mechanism present" if passed
        else f"LoF variant pathogenic but mechanism not LoF (stated {mech!r})"
    )
    return ValidatorResult("mechanism_plausibility", passed, reason)


# Patterns that indicate fabricated specific evidence not provided in the prompt.
_FABRICATION_PATTERNS = [
    r"\bgnomad\b",
    r"\ballele frequency of\s*[\d.]+",
    r"\bMAF\s*[=:]?\s*[\d.]+",
    r"\b\d+\s*(?:patients|families|individuals|cases)\b",
    r"\bet al\.?\b",
    r"\b(?:19|20)\d{2}\b",                  # a year, e.g. a citation
    r"\bPMID\b",
    r"\bfunctional (?:study|studies|assay) (?:showed|demonstrated|confirmed)",
]


def validate_no_fabrication(
    variant: Variant, result: InterpretationResult
) -> ValidatorResult:
    """Did the model avoid inventing specific evidence it was not given?

    The prompt provides only gene, HGVS, consequence, and condition. If the
    reasoning cites a specific allele frequency, a named study, a year, a
    patient count, or a database statistic, that evidence was fabricated,
    because none was supplied. This is the most safety-critical validator.
    """
    text = f"{result.reasoning} {' '.join(result.cited_evidence)}".lower()
    for pattern in _FABRICATION_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return ValidatorResult(
                "no_fabrication", False,
                f"fabricated evidence: matched {m.group(0)!r}",
            )
    return ValidatorResult("no_fabrication", True, "no fabricated evidence")


ALL_VALIDATORS = [
    validate_gene_grounding,
    validate_consequence_consistency,
    validate_mechanism_plausibility,
    validate_no_fabrication,
]


def run_validators(
    variant: Variant, result: InterpretationResult
) -> list[ValidatorResult]:
    """Run all validators on one interpretation."""
    return [v(variant, result) for v in ALL_VALIDATORS]


@dataclass
class ValidatorSummary:
    """Aggregate pass rates for each validator across the task set."""

    pass_rates: dict[str, float]
    n: int
    applicable: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "pass_rates": {k: round(v, 4) for k, v in self.pass_rates.items()},
            "applicable": self.applicable,
        }


def summarise_validators(
    pairs: list[tuple[Variant, InterpretationResult]],
) -> ValidatorSummary:
    """Aggregate validator pass rates across all interpretations.

    For mechanism_plausibility, "not applicable" cases are excluded from the
    denominator so the rate reflects only variants where the check applies.
    """
    names = [v(pairs[0][0], pairs[0][1]).name for v in ALL_VALIDATORS] if pairs else []
    passed: dict[str, int] = {n: 0 for n in names}
    applicable: dict[str, int] = {n: 0 for n in names}

    for variant, result in pairs:
        for vr in run_validators(variant, result):
            if vr.name == "mechanism_plausibility" and vr.reason == "not applicable":
                continue
            applicable[vr.name] += 1
            if vr.passed:
                passed[vr.name] += 1

    pass_rates = {
        n: (passed[n] / applicable[n] if applicable[n] else 1.0) for n in names
    }
    return ValidatorSummary(pass_rates=pass_rates, n=len(pairs), applicable=applicable)
