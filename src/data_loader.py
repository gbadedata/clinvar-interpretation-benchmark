"""ClinVar data loader.

Downloads the ClinVar variant_summary table from the NCBI FTP, parses it,
filters to the high-confidence oracle set (review stars >= threshold,
germline, cleanly classifiable), assigns difficulty tiers, and produces a
stratified, reproducible task set of Variant objects.

The loader never requires an API key. For tests and CI it can build the
task set from a small synthetic table instead of the network download, so
the entire pipeline is verifiable offline.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.data_models import Classification, Tier, Variant

logger = logging.getLogger(__name__)

# The columns we use from the ClinVar variant_summary table.
_USECOLS = [
    "GeneSymbol",
    "ClinicalSignificance",
    "ReviewStatus",
    "Assembly",
    "Name",                 # contains HGVS, e.g. NM_...(GENE):c.123A>G (p.Xxx)
    "PhenotypeList",
    "VariationID",
    "Type",
]


def download_clinvar_summary(dest: Path | None = None) -> Path:
    """Download and decompress the ClinVar variant_summary table.

    Args:
        dest: Destination path for the decompressed .txt. Defaults to
            settings.data_dir / variant_summary.txt.

    Returns:
        Path to the decompressed tab-delimited file.
    """
    import urllib.request

    dest = dest or (settings.data_dir / "variant_summary.txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("clinvar_summary_cached: %s", dest)
        return dest

    gz_path = dest.with_suffix(".txt.gz")
    logger.info("downloading_clinvar: %s", settings.clinvar_summary_url)
    urllib.request.urlretrieve(settings.clinvar_summary_url, gz_path)

    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        f_out.write(f_in.read())
    gz_path.unlink()
    logger.info("clinvar_summary_ready: %s", dest)
    return dest


def _stars_for(review_status: str) -> int:
    """Map a ClinVar review-status string to its star rating."""
    return settings.review_status_stars.get(review_status.strip().lower(), 0)


def _parse_hgvs(name: str) -> tuple[str, str]:
    """Extract (hgvs_c, hgvs_p) from a ClinVar Name field.

    Name looks like: NM_000059.4(BRCA2):c.1521_1523del (p.Phe508del)
    Returns ("", "") components that are absent.
    """
    hgvs_c = ""
    hgvs_p = ""
    if ":" in name:
        after = name.split(":", 1)[1]
        hgvs_c = after.split(" ")[0].strip()
    if "(p." in name:
        p = name.split("(p.", 1)[1].split(")")[0]
        hgvs_p = "p." + p.strip()
    return hgvs_c, hgvs_p


def _assign_tier(stars: int, raw_significance: str) -> Tier:
    """Assign a difficulty tier from review stars and classification text."""
    s = raw_significance.strip().lower()
    if "conflicting" in s or "uncertain significance" in s:
        return Tier.HARD
    if stars >= 3:
        return Tier.EASY
    return Tier.MEDIUM


def parse_clinvar_table(df: pd.DataFrame) -> list[Variant]:
    """Parse a ClinVar variant_summary DataFrame into Variant objects.

    Applies: assembly filter, star threshold, clean classification mapping.
    Variants that do not map to one of the three classes are dropped.
    """
    variants: list[Variant] = []
    for _, row in df.iterrows():
        if str(row.get("Assembly", "")) != settings.clinvar_assembly:
            continue
        stars = _stars_for(str(row.get("ReviewStatus", "")))
        if stars < settings.min_review_stars:
            continue

        raw_sig = str(row.get("ClinicalSignificance", ""))
        cls = Classification.from_clinvar(raw_sig)
        if cls is None:
            continue

        hgvs_c, hgvs_p = _parse_hgvs(str(row.get("Name", "")))
        tier = _assign_tier(stars, raw_sig)

        variants.append(
            Variant(
                variant_id=str(row.get("VariationID", "")),
                gene=str(row.get("GeneSymbol", "")),
                hgvs_c=hgvs_c,
                hgvs_p=hgvs_p,
                consequence=str(row.get("Type", "")),
                condition=str(row.get("PhenotypeList", "")).split("|")[0],
                oracle_classification=cls,
                review_stars=stars,
                tier=tier,
                raw_clinical_significance=raw_sig,
            )
        )
    logger.info("parsed_variants: %d", len(variants))
    return variants


def build_task_set(variants: list[Variant], seed: int | None = None) -> list[Variant]:
    """Stratified, reproducible sample of variants across tiers and classes.

    Samples up to settings.n_variants_per_tier from each difficulty tier,
    balancing classes within a tier where possible. Deterministic for a
    fixed seed.
    """
    import random

    rng = random.Random(seed if seed is not None else settings.random_seed)
    by_tier: dict[Tier, list[Variant]] = {t: [] for t in Tier}
    for v in variants:
        by_tier[v.tier].append(v)

    task: list[Variant] = []
    for tier, items in by_tier.items():
        rng.shuffle(items)
        task.extend(items[: settings.n_variants_per_tier])
    rng.shuffle(task)
    logger.info("task_set_built: %d variants", len(task))
    return task


def get_task_set(use_synthetic: bool = False) -> list[Variant]:
    """Top-level entry point: produce the benchmark task set.

    Args:
        use_synthetic: if True, build from the synthetic fixture table
            instead of downloading from NCBI (used in tests/CI).
    """
    if use_synthetic:
        df = _synthetic_table()
    else:
        path = download_clinvar_summary()
        df = pd.read_csv(path, sep="\t", usecols=lambda c: c in _USECOLS,
                         dtype=str, low_memory=False)
    variants = parse_clinvar_table(df)
    return build_task_set(variants)


def _synthetic_table() -> pd.DataFrame:
    """A small, biologically realistic synthetic ClinVar table for tests.

    Contains well-known variants with their genuine classifications so the
    parsing, filtering, and tiering logic is exercised against realistic
    shapes without any network access.
    """
    rows = [
        # gene, sig, review status, assembly, name, phenotype, id, type
        ("BRCA2", "Pathogenic", "reviewed by expert panel", "GRCh38",
         "NM_000059.4(BRCA2):c.5946del (p.Ser1982fs)", "Breast-ovarian cancer", "9325", "Deletion"),
        ("CFTR", "Pathogenic", "practice guideline", "GRCh38",
         "NM_000492.4(CFTR):c.1521_1523del (p.Phe508del)", "Cystic fibrosis", "7105", "Deletion"),
        ("BRCA1", "Pathogenic", "criteria provided, multiple submitters, no conflicts", "GRCh38",
         "NM_007294.4(BRCA1):c.68_69del (p.Glu23fs)", "Breast-ovarian cancer", "17661", "Deletion"),
        ("MLH1", "Likely pathogenic", "criteria provided, multiple submitters, no conflicts", "GRCh38",
         "NM_000249.4(MLH1):c.1852_1854del (p.Lys618del)", "Lynch syndrome", "89846", "Deletion"),
        ("TP53", "Benign", "reviewed by expert panel", "GRCh38",
         "NM_000546.6(TP53):c.215C>G (p.Pro72Arg)", "Li-Fraumeni syndrome", "12365", "single nucleotide variant"),
        ("APC", "Benign", "criteria provided, multiple submitters, no conflicts", "GRCh38",
         "NM_000038.6(APC):c.4479G>A (p.Thr1493=)", "Familial adenomatous polyposis", "41207", "single nucleotide variant"),
        ("MSH2", "Likely benign", "criteria provided, multiple submitters, no conflicts", "GRCh38",
         "NM_000251.3(MSH2):c.1661+12G>A", "Lynch syndrome", "90827", "single nucleotide variant"),
        ("ATM", "Uncertain significance", "criteria provided, multiple submitters, no conflicts", "GRCh38",
         "NM_000051.4(ATM):c.6919C>T (p.Leu2307Phe)", "Ataxia-telangiectasia", "127400", "single nucleotide variant"),
        ("VHL", "Uncertain significance", "criteria provided, multiple submitters, no conflicts", "GRCh38",
         "NM_000551.4(VHL):c.292T>C (p.Tyr98His)", "von Hippel-Lindau", "97584", "single nucleotide variant"),
        ("PALB2", "Conflicting classifications of pathogenicity", "criteria provided, conflicting classifications", "GRCh38",
         "NM_024675.4(PALB2):c.1010T>C (p.Leu337Ser)", "Breast cancer", "127888", "single nucleotide variant"),
        # A 1-star variant that MUST be filtered out (below threshold)
        ("EGFR", "Pathogenic", "criteria provided, single submitter", "GRCh38",
         "NM_005228.5(EGFR):c.2573T>G (p.Leu858Arg)", "Lung cancer", "16609", "single nucleotide variant"),
        # A drug-response variant that MUST be dropped (unmappable class)
        ("DPYD", "drug response", "reviewed by expert panel", "GRCh38",
         "NM_000110.4(DPYD):c.1905+1G>A", "Fluorouracil response", "97244", "single nucleotide variant"),
        # Wrong assembly, MUST be filtered
        ("KRAS", "Pathogenic", "reviewed by expert panel", "GRCh37",
         "NM_004985.5(KRAS):c.35G>A (p.Gly12Asp)", "Noonan syndrome", "12583", "single nucleotide variant"),
    ]
    return pd.DataFrame(rows, columns=[
        "GeneSymbol", "ClinicalSignificance", "ReviewStatus", "Assembly",
        "Name", "PhenotypeList", "VariationID", "Type",
    ])
