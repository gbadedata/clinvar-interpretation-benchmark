"""Benchmark runner: load data, interpret every variant, build the report.

Two entry points:

  run_benchmark(interpreter, ...) -- the core loop, model-agnostic. Takes any
      VariantInterpreter, runs it over the task set, and returns the report.

  main() -- CLI. Defaults to the MockInterpreter (offline, no key). Pass
      --live to use the real Claude API, and --synthetic to use the bundled
      fixture instead of downloading ClinVar.

Usage:
    python3 -m src.run_benchmark                 # mock + synthetic (CI-safe)
    python3 -m src.run_benchmark --real-data     # mock + real ClinVar
    python3 -m src.run_benchmark --live          # Claude + real ClinVar
    python3 -m src.run_benchmark --live --synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import structlog

from config.settings import settings
from src.benchmark.report import BenchmarkReport, build_report, print_summary
from src.data_loader import get_task_set
from src.data_models import Variant
from src.interpreter import MockInterpreter, VariantInterpreter

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger()


def run_benchmark(
    interpreter: VariantInterpreter,
    model_name: str,
    task: list[Variant],
    progress: bool = False,
) -> BenchmarkReport:
    """Run an interpreter over the task set and build the report.

    Args:
        interpreter: any object implementing interpret(variant).
        model_name: identifier recorded in the report.
        task: the list of variants to interpret.
        progress: if True, log every Nth variant (useful for live runs).

    Returns:
        The combined BenchmarkReport.
    """
    pairs = []
    t0 = time.time()
    for i, variant in enumerate(task, 1):
        result = interpreter.interpret(variant)
        pairs.append((variant, result))
        if progress and i % 10 == 0:
            log.info("interpreted", done=i, total=len(task))
    runtime = round(time.time() - t0, 1)

    report = build_report(model_name, pairs)
    log.info(
        "benchmark_complete",
        model=model_name,
        n=len(task),
        accuracy=round(report.score_report.accuracy, 4),
        runtime_seconds=runtime,
    )
    return report


def _run_comparison(task: list[Variant], model: str | None) -> None:
    """Run both evidence modes on the same variants and print the contrast.

    This is the headline experiment: the same model, the same variants,
    evaluated with and without the molecular-consequence evidence. The
    difference in safe decisive accuracy isolates the effect of evidence on
    model reliability, which is the core verification-loop question.
    """
    from src.interpreter import ClaudeInterpreter

    print("\nRunning evidence-POOR mode (minimal context)...")
    poor = ClaudeInterpreter(model=model, evidence_rich=False)
    poor_report = run_benchmark(poor, poor.model + " (poor)", task, progress=True)

    print("\nRunning evidence-RICH mode (with molecular consequence)...")
    rich = ClaudeInterpreter(model=model, evidence_rich=True)
    rich_report = run_benchmark(rich, rich.model + " (rich)", task, progress=True)

    print_summary(poor_report)
    print_summary(rich_report)

    # The contrast
    pa, ra = poor_report.abstention_report, rich_report.abstention_report
    print("\n" + "=" * 66)
    print("EVIDENCE EFFECT (the headline contrast)")
    print("=" * 66)
    print(f"  Overall accuracy:    {poor_report.score_report.accuracy:.3f}"
          f"  ->  {rich_report.score_report.accuracy:.3f}")
    print(f"  Decisiveness:        {pa.decisiveness:.3f}  ->  {ra.decisiveness:.3f}")
    print(f"  Safe rate:           {pa.safe_rate:.3f}  ->  {ra.safe_rate:.3f}")
    print(f"  Confident errors:    {pa.confident_errors}  ->  {ra.confident_errors}")
    print("=" * 66)

    for rep in (poor_report, rich_report):
        out = settings.reports_dir / f"benchmark_{rep.model_name.replace('/', '_').replace(' ', '_')}.json"
        rep.write(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClinVar interpretation benchmark")
    parser.add_argument("--live", action="store_true",
                        help="use the real Claude API (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the bundled fixture instead of downloading ClinVar")
    parser.add_argument("--real-data", action="store_true",
                        help="download real ClinVar (with the mock interpreter)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the task set to the first N variants (for small test runs)")
    parser.add_argument("--model", type=str, default=None,
                        help="override the model string for live runs")
    parser.add_argument("--evidence-rich", action="store_true",
                        help="supply derived molecular consequence as ACMG evidence")
    parser.add_argument("--compare-modes", action="store_true",
                        help="run BOTH evidence-poor and evidence-rich and compare")
    args = parser.parse_args()

    use_synthetic = args.synthetic or not (args.live or args.real_data)
    log.info("loading_task_set", synthetic=use_synthetic)
    task = get_task_set(use_synthetic=use_synthetic)
    if args.limit is not None:
        task = task[: args.limit]
        log.info("task_limited", n=len(task))

    if args.compare_modes:
        _run_comparison(task, args.model)
        return

    if args.live:
        from src.interpreter import ClaudeInterpreter
        interpreter: VariantInterpreter = ClaudeInterpreter(
            model=args.model, evidence_rich=args.evidence_rich
        )
        model_name = interpreter.model + ("+evidence" if args.evidence_rich else "")
        progress = True
    else:
        interpreter = MockInterpreter()
        model_name = "mock-heuristic"
        progress = False

    report = run_benchmark(interpreter, model_name, task, progress=progress)

    out = settings.reports_dir / f"benchmark_{model_name.replace('/', '_')}.json"
    report.write(out)
    print_summary(report)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    sys.exit(main())
