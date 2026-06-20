"""Tests for the ClinVar data loader, run entirely on the synthetic fixture."""

from __future__ import annotations

from config.settings import settings
from src.data_loader import (
    _assign_tier,
    _parse_hgvs,
    _stars_for,
    _synthetic_table,
    build_task_set,
    get_task_set,
    parse_clinvar_table,
)
from src.data_models import Classification, Tier


class TestStarMapping:
    def test_expert_panel_three_stars(self) -> None:
        assert _stars_for("reviewed by expert panel") == 3

    def test_practice_guideline_four_stars(self) -> None:
        assert _stars_for("practice guideline") == 4

    def test_multiple_submitters_two_stars(self) -> None:
        assert _stars_for("criteria provided, multiple submitters, no conflicts") == 2

    def test_single_submitter_one_star(self) -> None:
        assert _stars_for("criteria provided, single submitter") == 1

    def test_unknown_zero_stars(self) -> None:
        assert _stars_for("something unrecognised") == 0

    def test_case_insensitive(self) -> None:
        assert _stars_for("PRACTICE GUIDELINE") == 4


class TestClassificationMapping:
    def test_pathogenic(self) -> None:
        assert Classification.from_clinvar("Pathogenic") == Classification.PATHOGENIC

    def test_likely_pathogenic_collapses(self) -> None:
        assert Classification.from_clinvar("Likely pathogenic") == Classification.PATHOGENIC

    def test_benign(self) -> None:
        assert Classification.from_clinvar("Benign") == Classification.BENIGN

    def test_likely_benign_collapses(self) -> None:
        assert Classification.from_clinvar("Likely benign") == Classification.BENIGN

    def test_vus(self) -> None:
        assert Classification.from_clinvar("Uncertain significance") == Classification.VUS

    def test_conflicting_is_none(self) -> None:
        assert Classification.from_clinvar("Conflicting classifications of pathogenicity") is None

    def test_drug_response_is_none(self) -> None:
        assert Classification.from_clinvar("drug response") is None


class TestHgvsParsing:
    def test_extracts_c_and_p(self) -> None:
        c, p = _parse_hgvs("NM_000492.4(CFTR):c.1521_1523del (p.Phe508del)")
        assert c == "c.1521_1523del"
        assert p == "p.Phe508del"

    def test_handles_missing_protein(self) -> None:
        c, p = _parse_hgvs("NM_000251.3(MSH2):c.1661+12G>A")
        assert c == "c.1661+12G>A"
        assert p == ""


class TestTierAssignment:
    def test_conflicting_is_hard(self) -> None:
        assert _assign_tier(1, "Conflicting classifications") == Tier.HARD

    def test_vus_is_hard(self) -> None:
        assert _assign_tier(2, "Uncertain significance") == Tier.HARD

    def test_high_star_clear_is_easy(self) -> None:
        assert _assign_tier(3, "Pathogenic") == Tier.EASY

    def test_two_star_clear_is_medium(self) -> None:
        assert _assign_tier(2, "Pathogenic") == Tier.MEDIUM


class TestParseTable:
    def test_filters_below_star_threshold(self) -> None:
        variants = parse_clinvar_table(_synthetic_table())
        # EGFR is 1-star, must be excluded
        assert all(v.gene != "EGFR" for v in variants)

    def test_filters_wrong_assembly(self) -> None:
        variants = parse_clinvar_table(_synthetic_table())
        # KRAS is GRCh37, must be excluded
        assert all(v.gene != "KRAS" for v in variants)

    def test_drops_unmappable_classification(self) -> None:
        variants = parse_clinvar_table(_synthetic_table())
        # DPYD is drug response, must be dropped
        assert all(v.gene != "DPYD" for v in variants)

    def test_keeps_valid_variants(self) -> None:
        variants = parse_clinvar_table(_synthetic_table())
        genes = {v.gene for v in variants}
        assert "BRCA2" in genes
        assert "CFTR" in genes
        assert "ATM" in genes

    def test_oracle_classification_correct(self) -> None:
        variants = parse_clinvar_table(_synthetic_table())
        cftr = next(v for v in variants if v.gene == "CFTR")
        assert cftr.oracle_classification == Classification.PATHOGENIC
        assert cftr.review_stars == 4
        assert cftr.tier == Tier.EASY

    def test_context_hides_oracle(self) -> None:
        variants = parse_clinvar_table(_synthetic_table())
        ctx = variants[0].to_prompt_context()
        assert "oracle_classification" not in ctx
        assert "review_stars" not in ctx
        assert "gene" in ctx and "hgvs_c" in ctx


class TestBuildTaskSet:
    def test_deterministic_with_seed(self) -> None:
        variants = parse_clinvar_table(_synthetic_table())
        a = build_task_set(variants, seed=42)
        b = build_task_set(variants, seed=42)
        assert [v.variant_id for v in a] == [v.variant_id for v in b]

    def test_get_task_set_synthetic(self) -> None:
        task = get_task_set(use_synthetic=True)
        assert len(task) > 0
        assert all(v.review_stars >= settings.min_review_stars for v in task)
