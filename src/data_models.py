"""Core data models for the benchmark.

These typed structures are the contract between the data loader, the model
interface, and the evaluator. Keeping them in one place means every
component agrees on the shape of a variant, a model's interpretation, and
the oracle truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Classification(str, Enum):
    """Collapsed three-class variant classification.

    ClinVar uses finer labels (e.g. Likely pathogenic vs Pathogenic), but
    those boundaries are split even among experts. Collapsing to three
    classes scores the clinically meaningful distinction without penalising
    the likely-vs-definite call.
    """

    PATHOGENIC = "pathogenic"        # Pathogenic or Likely pathogenic
    BENIGN = "benign"                # Benign or Likely benign
    VUS = "vus"                      # Uncertain significance

    @classmethod
    def from_clinvar(cls, raw: str) -> "Classification | None":
        """Map a raw ClinVar clinical-significance string to a class.

        Returns None for strings that do not map cleanly (e.g. drug
        response, risk factor, other), which are excluded from the task.
        """
        s = raw.strip().lower()
        # Pathogenic family
        if "pathogenic" in s and "non-pathogenic" not in s:
            # Excludes "conflicting" which is handled separately upstream
            if "conflicting" in s:
                return None
            return cls.PATHOGENIC
        # Benign family
        if "benign" in s:
            if "conflicting" in s:
                return None
            return cls.BENIGN
        # Uncertain
        if "uncertain significance" in s or s == "vus":
            return cls.VUS
        return None


class Tier(str, Enum):
    """Difficulty tier, derived from ClinVar review status and history."""

    EASY = "easy"        # 3-4 star, clear consensus
    MEDIUM = "medium"    # 2 star, multiple submitters
    HARD = "hard"        # conflicting history or borderline VUS


@dataclass
class Variant:
    """A single variant interpretation task.

    Holds the context given to the model AND the oracle truth, but the
    model interface only ever receives the context fields (see
    `to_prompt_context`). The oracle fields are withheld until scoring.
    """

    # ── Context (shown to the model) ───────────────────────────────────
    variant_id: str               # ClinVar VariationID
    gene: str
    hgvs_c: str                   # nucleotide change, e.g. c.1521_1523del
    hgvs_p: str                   # protein change, e.g. p.Phe508del
    consequence: str              # molecular consequence
    condition: str                # associated phenotype/disease

    # ── Oracle truth (withheld until scoring) ──────────────────────────
    oracle_classification: Classification = Classification.VUS
    review_stars: int = 0
    tier: Tier = Tier.MEDIUM
    raw_clinical_significance: str = ""

    def to_prompt_context(self) -> dict[str, str]:
        """Return ONLY the fields the model is allowed to see."""
        return {
            "gene": self.gene,
            "hgvs_c": self.hgvs_c,
            "hgvs_p": self.hgvs_p,
            "consequence": self.consequence,
            "condition": self.condition,
        }


@dataclass
class InterpretationResult:
    """A model's interpretation of one variant.

    Returned by any VariantInterpreter implementation. Captures both the
    final classification and the structured reasoning the validators check.
    """

    variant_id: str
    classification: Classification
    stated_gene: str = ""             # gene the model says is involved
    stated_consequence: str = ""      # consequence the model states
    stated_mechanism: str = ""        # e.g. loss-of-function, gain-of-function
    reasoning: str = ""               # free-text rationale
    cited_evidence: list[str] = field(default_factory=list)
    raw_response: str = ""            # unparsed model output, for audit
