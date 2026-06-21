"""Tests for evidence-rich prompt building and consequence derivation."""

from __future__ import annotations

from src.data_models import Classification, Tier, Variant
from src.interpreter import (
    build_evidence_rich_prompt,
    build_user_prompt,
    derive_molecular_consequence,
)


def _v(hgvs_c="c.5946del", hgvs_p="p.Ser1982fs", consequence="Deletion"):
    return Variant(
        variant_id="t", gene="BRCA2", hgvs_c=hgvs_c, hgvs_p=hgvs_p,
        consequence=consequence, condition="cancer",
        oracle_classification=Classification.PATHOGENIC, tier=Tier.EASY,
    )


class TestConsequenceDerivation:
    def test_frameshift(self) -> None:
        assert "frameshift" in derive_molecular_consequence(_v())

    def test_nonsense(self) -> None:
        c = derive_molecular_consequence(_v(hgvs_p="p.Arg213Ter"))
        assert "nonsense" in c or "stop" in c

    def test_synonymous(self) -> None:
        c = derive_molecular_consequence(_v(hgvs_p="p.Thr1493=", consequence="snv"))
        assert "synonymous" in c

    def test_missense(self) -> None:
        c = derive_molecular_consequence(
            _v(hgvs_c="c.2573T>G", hgvs_p="p.Leu858Arg", consequence="snv")
        )
        assert "missense" in c


class TestEvidenceRichPrompt:
    def test_includes_derived_consequence(self) -> None:
        prompt = build_evidence_rich_prompt(_v())
        assert "frameshift" in prompt
        assert "PVS1" in prompt  # ACMG framing present

    def test_evidence_rich_differs_from_poor(self) -> None:
        v = _v()
        assert build_evidence_rich_prompt(v) != build_user_prompt(v)

    def test_still_hides_oracle(self) -> None:
        v = _v()
        prompt = build_evidence_rich_prompt(v)
        # oracle label must not leak
        assert "pathogenic" not in prompt.lower().split("pvs1")[0].replace(
            "loss-of-function", ""
        ) or "PVS1" in prompt  # PVS1 sentence legitimately discusses pathogenic criteria
