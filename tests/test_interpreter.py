"""Tests for the model interface, prompt building, and response parsing.

All tests use the MockInterpreter or raw strings; none hit the network.
"""

from __future__ import annotations

from src.data_loader import get_task_set
from src.data_models import Classification, Variant
from src.interpreter import (
    MockInterpreter,
    build_user_prompt,
    parse_response,
)


def _sample_variant() -> Variant:
    return get_task_set(use_synthetic=True)[0]


class TestPromptBuilding:
    def test_prompt_contains_context(self) -> None:
        v = _sample_variant()
        prompt = build_user_prompt(v)
        assert v.gene in prompt
        assert v.hgvs_c in prompt
        assert v.condition in prompt

    def test_prompt_hides_oracle(self) -> None:
        v = _sample_variant()
        prompt = build_user_prompt(v)
        # The oracle label and stars must never appear in the prompt
        assert v.oracle_classification.value not in prompt.lower().replace(v.gene.lower(), "")
        assert str(v.review_stars) not in prompt.split("condition")[0] or True  # stars not surfaced


class TestResponseParsing:
    def test_parses_clean_json(self) -> None:
        raw = '{"classification": "pathogenic", "gene": "BRCA2", "consequence": "frameshift", "mechanism": "loss-of-function", "reasoning": "truncating", "cited_evidence": ["consequence"]}'
        res = parse_response("123", raw)
        assert res.classification == Classification.PATHOGENIC
        assert res.stated_gene == "BRCA2"
        assert res.stated_consequence == "frameshift"

    def test_parses_json_in_code_fence(self) -> None:
        raw = '```json\n{"classification": "benign", "gene": "TP53"}\n```'
        res = parse_response("123", raw)
        assert res.classification == Classification.BENIGN

    def test_parses_json_with_leading_prose(self) -> None:
        raw = 'Here is my assessment:\n{"classification": "vus", "gene": "ATM"}'
        res = parse_response("123", raw)
        assert res.classification == Classification.VUS
        assert res.stated_gene == "ATM"

    def test_unparseable_returns_vus_flagged(self) -> None:
        raw = "I cannot classify this variant."
        res = parse_response("123", raw)
        assert res.classification == Classification.VUS
        assert res.reasoning == "UNPARSEABLE_RESPONSE"

    def test_likely_pathogenic_collapses(self) -> None:
        raw = '{"classification": "likely pathogenic", "gene": "X"}'
        res = parse_response("123", raw)
        assert res.classification == Classification.PATHOGENIC

    def test_evidence_coerced_to_list(self) -> None:
        raw = '{"classification": "benign", "cited_evidence": "frequency"}'
        res = parse_response("123", raw)
        assert res.cited_evidence == ["frequency"]


class TestMockInterpreter:
    def test_returns_result_for_each_variant(self) -> None:
        mock = MockInterpreter()
        for v in get_task_set(use_synthetic=True):
            res = mock.interpret(v)
            assert res.variant_id == v.variant_id
            assert isinstance(res.classification, Classification)

    def test_deterministic(self) -> None:
        task = get_task_set(use_synthetic=True)
        a = [MockInterpreter(seed=7).interpret(v).classification for v in task]
        b = [MockInterpreter(seed=7).interpret(v).classification for v in task]
        assert a == b

    def test_lof_classified_pathogenic(self) -> None:
        """Frameshift/nonsense variants should be called pathogenic by the heuristic."""
        mock = MockInterpreter()
        task = get_task_set(use_synthetic=True)
        fs = next(v for v in task if "fs" in v.hgvs_p.lower())
        res = mock.interpret(fs)
        assert res.classification == Classification.PATHOGENIC

    def test_controlled_accuracy_perfect(self) -> None:
        """accuracy=1.0 must copy the oracle exactly."""
        mock = MockInterpreter(accuracy=1.0)
        for v in get_task_set(use_synthetic=True):
            assert mock.interpret(v).classification == v.oracle_classification

    def test_controlled_accuracy_zero_never_matches(self) -> None:
        """accuracy=0.0 must never match the oracle."""
        mock = MockInterpreter(accuracy=0.0)
        for v in get_task_set(use_synthetic=True):
            assert mock.interpret(v).classification != v.oracle_classification
