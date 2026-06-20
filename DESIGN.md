# ClinVar Variant Interpretation Benchmark: Design

## The question this benchmark answers

Frontier models are increasingly used to interpret genetic variants in target
discovery and clinical decision support. Before trusting them, we need to know:
**when a model classifies a variant as pathogenic or benign, is it right, where
does it fail, and can we measure that against expert consensus?**

This benchmark builds that verification loop. It scores a model's variant
interpretation against ClinVar's expert-reviewed classifications, with calibrated
difficulty tiers, biological validators that check the model's reasoning (not just
its final label), and honest reporting of where the model breaks down.

## Why ClinVar is the right oracle

ClinVar assigns every variant a review status on a four-star expert-consensus
scale:

| Stars | Review status | Use in this benchmark |
|---|---|---|
| 4 | Practice guideline | Oracle (highest confidence) |
| 3 | Reviewed by expert panel | Oracle (high confidence) |
| 2 | Criteria provided, multiple submitters, no conflicts | Oracle (consensus) |
| 1 | Criteria provided, single submitter | Excluded from oracle (low confidence) |
| 0 | No assertion criteria | Excluded |

By restricting the oracle to 2-star-and-above, the ground truth is genuine
multi-laboratory expert consensus, not a single opinion. This is exactly the
"expert consensus loop" that high-stakes genetic interpretation requires.

## The task

For each variant, the model receives structured context:
- Gene symbol
- HGVS nucleotide and protein change
- Molecular consequence (missense, nonsense, frameshift, splice, etc.)
- Associated condition / phenotype
- (Optionally) population allele frequency

The model must return:
1. A classification: Pathogenic/Likely pathogenic, Benign/Likely benign, or VUS
2. Structured reasoning: the gene, the predicted consequence, the inheritance
   pattern, and the key evidence it relied on

## Scoring (three layers, mirroring the portfolio pattern)

### Layer 1 -- Oracle agreement
Does the model's classification match ClinVar's aggregate classification?
Collapsed to three classes (P/LP, B/LB, VUS) to avoid penalising the
likely-vs-definite boundary, which even experts split on. Metrics: accuracy,
per-class F1, Cohen kappa, and a confusion matrix.

### Layer 2 -- Difficulty tiers
- **Easy**: 3-4 star, clear P or B, strong consensus. Any capable model should get these.
- **Medium**: 2 star, P or B with multiple submitters. Separates good from average.
- **Hard**: variants with a history of conflicting interpretations, or VUS with
  borderline evidence. Where models (and humans) genuinely struggle.

Reporting per tier prevents a high easy-tier score from masking hard-tier failure.

### Layer 3 -- Biological validators (independent of the oracle)
These check the model's *reasoning*, not its label. They can pass or fail
regardless of whether the final classification was correct:
- **Gene grounding**: did the model name the gene actually carrying the variant?
- **Consequence consistency**: does the model's stated molecular consequence
  match the variant's actual consequence annotation?
- **Mechanism plausibility**: for a loss-of-function consequence (nonsense,
  frameshift, canonical splice) in a known haploinsufficient gene, does the
  model's reasoning reflect a loss-of-function mechanism?
- **No-fabrication check**: does the model avoid inventing evidence (e.g. citing
  a specific functional study or frequency that was not provided to it)?

The validators are the heart of the "verify the biology" contribution. A model
that gets the right label for the wrong reason is not trustworthy in a clinical
loop, and only reasoning-level validators can catch that.

## Model interface (framework-first, then real API)

The benchmark is built against an abstract `VariantInterpreter` interface with a
single method: `interpret(variant_context) -> InterpretationResult`. Two
implementations:
1. `MockInterpreter` -- deterministic, seeded, used in all unit tests and CI.
   Requires no API key and no network. Lets the entire scoring framework be
   validated independently of any model.
2. `ClaudeInterpreter` -- wires in the real Anthropic API. Used for live
   evaluation runs, never in CI.

This separation is deliberate and is itself an evaluation-design principle: the
scoring logic must not depend on which model is being scored.

## Reproducibility

- ClinVar `variant_summary.txt` downloaded from the NCBI FTP (no key required),
  pinned by release date and checksummed.
- Deterministic stratified sampling (seed=42) into the task set.
- Hidden-answer design: the oracle labels are withheld from the model and only
  revealed to the evaluator after interpretation.
- Full pytest suite on synthetic fixtures, ruff, CI.

## What this demonstrates for evaluation-design work

- A real expert-consensus oracle with a principled confidence threshold
- Difficulty calibration grounded in the data's own review hierarchy
- Reasoning-level validators that catch right-answer-wrong-reason failures
- A clean model-agnostic scoring seam
- Honest per-tier reporting including documented failure modes
