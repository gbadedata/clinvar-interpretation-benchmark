"""Tests for the biological validators.

Each validator is tested on both a passing and a failing case. The
fabrication detector gets extra coverage because it is the safety-critical
check and the most prone to false positives/negatives.
"""

from __future__ import annotations

from src.data_models import Classification, InterpretationResult, Tier, Variant
from src.benchmark.validators import (
    run_validators,
    summarise_validators,
    validate_consequence_consistency,
    validate_gene_grounding,
    validate_mechanism_plausibility,
    validate_no_fabrication,
)


def _variant(gene="BRCA2", hgvs_c="c.5946del", hgvs_p="p.Ser1982fs",
             consequence="Deletion", oracle=Classification.PATHOGENIC):
    return Variant(
        variant_id="t", gene=gene, hgvs_c=hgvs_c, hgvs_p=hgvs_p,
        consequence=consequence, condition="cancer",
        oracle_classification=oracle, tier=Tier.EASY,
    )


def _result(gene="BRCA2", consequence="frameshift", mechanism="loss-of-function",
            cls=Classification.PATHOGENIC, reasoning="truncating variant",
            evidence=None):
    return InterpretationResult(
        variant_id="t", classification=cls, stated_gene=gene,
        stated_consequence=consequence, stated_mechanism=mechanism,
        reasoning=reasoning, cited_evidence=evidence or ["consequence"],
    )


class TestGeneGrounding:
    def test_pass_when_gene_matches(self) -> None:
        vr = validate_gene_grounding(_variant(), _result(gene="BRCA2"))
        assert vr.passed

    def test_fail_when_gene_wrong(self) -> None:
        vr = validate_gene_grounding(_variant(), _result(gene="BRCA1"))
        assert not vr.passed

    def test_fail_when_gene_empty(self) -> None:
        vr = validate_gene_grounding(_variant(), _result(gene=""))
        assert not vr.passed

    def test_case_insensitive(self) -> None:
        vr = validate_gene_grounding(_variant(gene="BRCA2"), _result(gene="brca2"))
        assert vr.passed


class TestConsequenceConsistency:
    def test_pass_frameshift_match(self) -> None:
        vr = validate_consequence_consistency(_variant(), _result(consequence="frameshift"))
        assert vr.passed

    def test_fail_missense_for_frameshift(self) -> None:
        vr = validate_consequence_consistency(_variant(), _result(consequence="missense"))
        assert not vr.passed

    def test_synonymous_match(self) -> None:
        v = _variant(hgvs_p="p.Thr1493=", consequence="single nucleotide variant",
                     oracle=Classification.BENIGN)
        vr = validate_consequence_consistency(v, _result(consequence="synonymous"))
        assert vr.passed


class TestMechanismPlausibility:
    def test_pass_lof_mechanism(self) -> None:
        vr = validate_mechanism_plausibility(
            _variant(), _result(mechanism="loss-of-function")
        )
        assert vr.passed

    def test_pass_lof_in_reasoning(self) -> None:
        vr = validate_mechanism_plausibility(
            _variant(), _result(mechanism="", reasoning="this truncating frameshift causes LoF")
        )
        assert vr.passed

    def test_fail_lof_variant_no_mechanism(self) -> None:
        vr = validate_mechanism_plausibility(
            _variant(), _result(mechanism="unknown", reasoning="this is bad")
        )
        assert not vr.passed

    def test_not_applicable_for_missense(self) -> None:
        v = _variant(hgvs_p="p.Leu2307Phe", consequence="single nucleotide variant",
                     oracle=Classification.VUS)
        vr = validate_mechanism_plausibility(
            v, _result(consequence="missense", cls=Classification.VUS)
        )
        assert vr.passed
        assert vr.reason == "not applicable"

    def test_not_applicable_when_benign(self) -> None:
        vr = validate_mechanism_plausibility(
            _variant(), _result(cls=Classification.BENIGN)
        )
        assert vr.reason == "not applicable"


class TestNoFabrication:
    def test_pass_clean_reasoning(self) -> None:
        vr = validate_no_fabrication(
            _variant(), _result(reasoning="A frameshift in this gene is loss-of-function.")
        )
        assert vr.passed

    def test_fail_invented_frequency(self) -> None:
        vr = validate_no_fabrication(
            _variant(), _result(reasoning="The allele frequency of 0.0001 supports pathogenicity.")
        )
        assert not vr.passed

    def test_fail_invented_gnomad(self) -> None:
        vr = validate_no_fabrication(
            _variant(), _result(reasoning="Absent from gnomAD population databases.")
        )
        assert not vr.passed

    def test_fail_invented_citation_year(self) -> None:
        vr = validate_no_fabrication(
            _variant(), _result(reasoning="As shown by Smith et al. 2019 this is pathogenic.")
        )
        assert not vr.passed

    def test_fail_invented_patient_count(self) -> None:
        vr = validate_no_fabrication(
            _variant(), _result(reasoning="Reported in 42 families with the disease.")
        )
        assert not vr.passed

    def test_fail_invented_functional_study(self) -> None:
        vr = validate_no_fabrication(
            _variant(), _result(reasoning="Functional studies showed loss of protein activity.")
        )
        assert not vr.passed

    def test_pass_generic_evidence_types(self) -> None:
        vr = validate_no_fabrication(
            _variant(),
            _result(reasoning="The molecular consequence and gene-disease association support this.",
                    evidence=["consequence", "gene-disease association"])
        )
        assert vr.passed


class TestRunAndSummarise:
    def test_run_returns_all_four(self) -> None:
        results = run_validators(_variant(), _result())
        assert len(results) == 4
        names = {r.name for r in results}
        assert names == {"gene_grounding", "consequence_consistency",
                         "mechanism_plausibility", "no_fabrication"}

    def test_summary_perfect_reasoning(self) -> None:
        pairs = [(_variant(), _result())]
        summary = summarise_validators(pairs)
        assert summary.pass_rates["gene_grounding"] == 1.0
        assert summary.pass_rates["no_fabrication"] == 1.0

    def test_summary_excludes_na_from_mechanism_denominator(self) -> None:
        # A benign missense -> mechanism not applicable -> excluded
        v = _variant(hgvs_p="p.Pro72Arg", consequence="single nucleotide variant",
                     oracle=Classification.BENIGN)
        r = _result(consequence="missense", cls=Classification.BENIGN)
        summary = summarise_validators([(v, r)])
        assert summary.applicable["mechanism_plausibility"] == 0
        assert summary.pass_rates["mechanism_plausibility"] == 1.0
