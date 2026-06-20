"""Configuration for the ClinVar variant interpretation benchmark.

All parameters are environment-overridable via the CLINVAR_ prefix, so
thresholds and paths can be changed without editing source.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLINVAR_", extra="ignore")

    # ── Paths ──────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    evidence_dir: Path = Path("evidence")

    # ── ClinVar source ─────────────────────────────────────────────────
    # The NCBI FTP variant_summary is a tab-delimited table updated weekly.
    # No API key required. Pinned filename keeps the build reproducible.
    clinvar_summary_url: str = (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    )
    clinvar_assembly: str = "GRCh38"

    # ── Oracle confidence filter ───────────────────────────────────────
    # Review-status star ratings accepted into the oracle. Restricting to
    # 2-star-and-above means the ground truth is genuine multi-submitter
    # expert consensus, not a single opinion.
    min_review_stars: int = 2

    # Review-status strings mapped to their star ratings (ClinVar scale).
    review_status_stars: dict[str, int] = {
        "practice guideline": 4,
        "reviewed by expert panel": 3,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, conflicting classifications": 1,
        "criteria provided, conflicting interpretations": 1,
        "criteria provided, single submitter": 1,
        "no assertion criteria provided": 0,
        "no assertion provided": 0,
        "no classification provided": 0,
    }

    # ── Task sampling ──────────────────────────────────────────────────
    n_variants_per_tier: int = 100
    random_seed: int = 42

    # ── Scoring ────────────────────────────────────────────────────────
    # Minimum oracle agreement (accuracy) for the benchmark to "pass" as a
    # whole. Calibrated, not aspirational; documented in the README.
    min_overall_accuracy: float = 0.60
    # Minimum gene-grounding validator pass rate.
    min_gene_grounding: float = 0.90

    @property
    def reports_dir(self) -> Path:
        return self.evidence_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.evidence_dir / "figures"


settings = Settings()
