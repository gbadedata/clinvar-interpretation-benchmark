"""Model interface for variant interpretation.

The benchmark scores any model that implements the VariantInterpreter
protocol. The scoring logic never depends on which model is used; this
separation is itself an evaluation-design principle.

Two implementations live here:
  - MockInterpreter: deterministic, seeded, no network. Used in all unit
    tests and CI. Lets the entire scoring framework be validated without
    an API key.
  - ClaudeInterpreter: wires in the real Anthropic API. Used only for live
    evaluation runs (see src/run_live.py), never in CI.

Both return a strict, validated InterpretationResult. The model is asked
to reply in strict JSON, which is parsed and validated; malformed output
is handled explicitly rather than silently mis-scored.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Protocol

from src.data_models import Classification, InterpretationResult, Variant

logger = logging.getLogger(__name__)


# ── The prompt ─────────────────────────────────────────────────────────
# The model receives ONLY the context fields (gene, HGVS, consequence,
# condition). It is explicitly told to return strict JSON with both a
# classification and the structured reasoning the validators check.

SYSTEM_PROMPT = """You are a clinical variant interpretation assistant. \
You classify germline genetic variants using ACMG/AMP-style reasoning. \
You are given structured variant information and must return your \
assessment as strict JSON only, with no surrounding prose or markdown.

You must respond with a single JSON object with exactly these keys:
  "classification": one of "pathogenic", "benign", "vus"
        (use "pathogenic" for pathogenic or likely pathogenic,
         "benign" for benign or likely benign,
         "vus" for uncertain significance)
  "gene": the gene symbol you believe carries this variant
  "consequence": the molecular consequence (e.g. missense, nonsense,
         frameshift, splice, synonymous, in-frame deletion)
  "mechanism": the disease mechanism if pathogenic
         (e.g. "loss-of-function", "gain-of-function", "none")
  "reasoning": a brief explanation of your classification (2-4 sentences)
  "cited_evidence": a list of evidence types you relied on
         (e.g. ["consequence", "gene-disease association"]). Only list
         evidence that was provided to you; do not invent specific
         studies, frequencies, or database entries you were not given.

Return ONLY the JSON object."""


def build_user_prompt(variant: Variant) -> str:
    """Build the evidence-poor user message (minimal context only)."""
    ctx = variant.to_prompt_context()
    return (
        "Classify the following germline variant.\n\n"
        f"Gene: {ctx['gene']}\n"
        f"HGVS (coding): {ctx['hgvs_c']}\n"
        f"HGVS (protein): {ctx['hgvs_p']}\n"
        f"Variant type: {ctx['consequence']}\n"
        f"Associated condition: {ctx['condition']}\n\n"
        "Respond with the JSON object only."
    )


def derive_molecular_consequence(variant: Variant) -> str:
    """Derive the real molecular consequence from the HGVS strings.

    This is genuine evidence, not fabricated: it is read directly from the
    variant's own HGVS nomenclature, which ClinVar provides. The molecular
    consequence is the highest-value ACMG criterion (PVS1 for loss of
    function), so supplying it is the single most informative honest piece
    of evidence we can add.
    """
    blob = f"{variant.hgvs_p} {variant.hgvs_c} {variant.consequence}".lower()
    if "fs" in blob or "frameshift" in blob:
        return "frameshift (predicted loss of function)"
    if "ter" in blob or "*" in blob or "stop" in blob:
        return "nonsense / stop-gain (predicted loss of function)"
    if "=" in variant.hgvs_p or "synonymous" in blob or "silent" in blob:
        return "synonymous (no amino acid change)"
    if ("+" in variant.hgvs_c or "-" in variant.hgvs_c) and "del" not in blob:
        return "splice-region (potential splicing effect)"
    if "del" in blob and "fs" not in blob:
        return "in-frame deletion"
    if "dup" in blob and "fs" not in blob:
        return "in-frame duplication"
    if variant.hgvs_p and variant.hgvs_p != "p.":
        return "missense (single amino acid substitution)"
    return "unknown consequence"


def build_evidence_rich_prompt(variant: Variant) -> str:
    """Build the evidence-rich user message.

    Adds the derived molecular consequence and an explicit note on its ACMG
    relevance. All evidence is read from the variant's own HGVS, so nothing
    is fabricated; the model is simply given the functional interpretation
    of the nomenclature it was already shown.
    """
    ctx = variant.to_prompt_context()
    consequence = derive_molecular_consequence(variant)
    return (
        "Classify the following germline variant. Additional molecular "
        "evidence is provided to support ACMG/AMP-style assessment.\n\n"
        f"Gene: {ctx['gene']}\n"
        f"HGVS (coding): {ctx['hgvs_c']}\n"
        f"HGVS (protein): {ctx['hgvs_p']}\n"
        f"Molecular consequence: {consequence}\n"
        f"Associated condition: {ctx['condition']}\n\n"
        "Consider the molecular consequence as ACMG evidence: predicted "
        "loss-of-function variants (frameshift, nonsense, canonical splice) "
        "in genes where loss of function is an established disease mechanism "
        "meet strong pathogenic criteria (PVS1). Synonymous variants outside "
        "splice regions typically lack a protein-level effect.\n\n"
        "Respond with the JSON object only."
    )


# ── Response parsing ───────────────────────────────────────────────────

def parse_response(variant_id: str, raw: str) -> InterpretationResult:
    """Parse a model's raw text into a validated InterpretationResult.

    Robust to common formatting noise: code fences, leading prose, or a
    JSON object embedded in surrounding text. If no valid classification
    can be recovered, returns a VUS result flagged in its reasoning, so a
    malformed response scores as an (incorrect) uncertain call rather than
    crashing the run.
    """
    obj = _extract_json(raw)
    if obj is None:
        logger.warning("unparseable_response: %s", variant_id)
        return InterpretationResult(
            variant_id=variant_id,
            classification=Classification.VUS,
            reasoning="UNPARSEABLE_RESPONSE",
            raw_response=raw,
        )

    cls = _coerce_classification(obj.get("classification", ""))
    evidence = obj.get("cited_evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    return InterpretationResult(
        variant_id=variant_id,
        classification=cls,
        stated_gene=str(obj.get("gene", "")).strip(),
        stated_consequence=str(obj.get("consequence", "")).strip().lower(),
        stated_mechanism=str(obj.get("mechanism", "")).strip().lower(),
        reasoning=str(obj.get("reasoning", "")).strip(),
        cited_evidence=[str(e).strip() for e in evidence],
        raw_response=raw,
    )


def _extract_json(raw: str) -> dict | None:
    """Pull the first valid JSON object out of a raw model response."""
    # Strip code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Fast path: whole string is JSON
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _coerce_classification(value: str) -> Classification:
    """Map a model's classification string to the Classification enum."""
    s = str(value).strip().lower()
    if s in ("pathogenic", "likely pathogenic", "p", "lp"):
        return Classification.PATHOGENIC
    if s in ("benign", "likely benign", "b", "lb"):
        return Classification.BENIGN
    return Classification.VUS


# ── Interpreter protocol and implementations ───────────────────────────

class VariantInterpreter(Protocol):
    """Any model that can interpret a variant implements this."""

    def interpret(self, variant: Variant) -> InterpretationResult:
        ...


class MockInterpreter:
    """Deterministic interpreter for tests and CI, no network required.

    Its behaviour is a simple, transparent heuristic, NOT a real model: it
    classifies loss-of-function consequences as pathogenic, synonymous as
    benign, and otherwise echoes a seeded pseudo-random class. The point is
    not accuracy; it is to exercise the full scoring path deterministically.

    An optional `accuracy` parameter lets tests request a mock that copies
    the oracle a fixed fraction of the time, so scoring metrics can be
    asserted against a known expected value.
    """

    def __init__(self, seed: int = 42, accuracy: float | None = None) -> None:
        self.rng = random.Random(seed)
        self.accuracy = accuracy

    def interpret(self, variant: Variant) -> InterpretationResult:
        if self.accuracy is not None:
            # Controlled-accuracy mock: copy oracle `accuracy` of the time
            if self.rng.random() < self.accuracy:
                cls = variant.oracle_classification
            else:
                others = [c for c in Classification if c != variant.oracle_classification]
                cls = self.rng.choice(others)
        else:
            cls = self._heuristic(variant)

        return InterpretationResult(
            variant_id=variant.variant_id,
            classification=cls,
            stated_gene=variant.gene,
            stated_consequence=self._infer_consequence(variant),
            stated_mechanism="loss-of-function" if cls == Classification.PATHOGENIC else "none",
            reasoning="Mock heuristic interpretation for testing.",
            cited_evidence=["consequence"],
            raw_response="<mock>",
        )

    @staticmethod
    def _infer_consequence(variant: Variant) -> str:
        name = (variant.hgvs_p + " " + variant.consequence).lower()
        if "fs" in name or "deletion" in name:
            return "frameshift"
        if "ter" in name or "*" in name:
            return "nonsense"
        if "=" in variant.hgvs_p:
            return "synonymous"
        return "missense"

    def _heuristic(self, variant: Variant) -> Classification:
        cons = self._infer_consequence(variant)
        if cons in ("frameshift", "nonsense"):
            return Classification.PATHOGENIC
        if cons == "synonymous":
            return Classification.BENIGN
        return self.rng.choice(list(Classification))


class ClaudeInterpreter:
    """Real Anthropic API interpreter. Used only for live runs, not CI.

    Requires the ANTHROPIC_API_KEY environment variable. Kept deliberately
    thin: it sends the system prompt plus the per-variant user prompt and
    delegates all parsing to parse_response, so the live path and the test
    path share identical scoring.
    """

    def __init__(self, model: str | None = None, max_tokens: int | None = None,
                 evidence_rich: bool = False) -> None:
        from anthropic import Anthropic

        from config.settings import settings

        self.client = Anthropic()
        self.model = model or settings.model
        self.max_tokens = max_tokens or settings.max_tokens
        self.evidence_rich = evidence_rich

    def interpret(self, variant: Variant) -> InterpretationResult:
        prompt = (
            build_evidence_rich_prompt(variant)
            if self.evidence_rich
            else build_user_prompt(variant)
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in message.content if hasattr(block, "text"))
        return parse_response(variant.variant_id, raw)
