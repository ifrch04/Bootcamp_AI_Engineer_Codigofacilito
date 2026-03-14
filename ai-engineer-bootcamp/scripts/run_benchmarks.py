#!/usr/bin/env python3
"""CLI script to run inference benchmarks across backends.

Usage examples:
    python scripts/run_benchmarks.py --backends groq ollama --runs 5 --format table
    python scripts/run_benchmarks.py --runs 3 --output resultados_clase8 --format json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from inference.benchmark import (
    BenchmarkConfig,
    export_results_csv,
    export_results_json,
    format_results,
    measure_vram_usage,
    run_full_benchmark,
)
from inference.local_adapter import InferenceBackend
from inference.model_registry import check_backend_health, get_model_info

console = Console()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference backends (Clase 8 — Cloud vs Local)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[b.value for b in InferenceBackend],
        default=None,
        help="Backends to benchmark (default: all available)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measured iterations per prompt (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per response (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output file path without extension (default: benchmark_results)",
    )
    parser.add_argument(
        "--format",
        dest="out_format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        default=True,
        help="Skip backends that are not available (default: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def discover_backends(requested: list[str] | None) -> list[InferenceBackend]:
    """Check which backends are healthy and return the active ones."""
    if requested:
        candidates = [InferenceBackend(b) for b in requested]
    else:
        candidates = list(InferenceBackend)

    active: list[InferenceBackend] = []

    console.print("\n[bold]Verificando backends disponibles...[/bold]")

    for backend in candidates:
        info = get_model_info(backend)
        is_local = backend != InferenceBackend.GROQ
        label = "local" if is_local else "cloud"

        healthy = check_backend_health(backend)
        if healthy:
            console.print(f"  [green]OK[/green] {backend.value:<10} -> {info.model_id} ({label})")
            active.append(backend)
        else:
            console.print(
                f"  [red]X[/red]  {backend.value:<10} -> No disponible ({label})"
            )

    if not active:
        console.print("\n[bold red]No hay backends disponibles. Abortando.[/bold red]")
        sys.exit(1)

    return active


def print_rich_results(results: list, config: BenchmarkConfig) -> None:
    """Display results using rich tables."""
    import statistics

    # Group results by prompt label
    prompts_seen: list[str] = []
    for r in results:
        if r.prompt_label not in prompts_seen:
            prompts_seen.append(r.prompt_label)

    for label in prompts_seen:
        table = Table(
            title=f"Prompt: {label}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Backend", style="bold")
        table.add_column("Modelo")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("Tks/s", justify="right")
        table.add_column("Total (ms)", justify="right")
        table.add_column("VRAM (MB)", justify="right")

        for r in results:
            if r.prompt_label != label:
                continue

            def fmt(vals: list[float]) -> str:
                if not vals:
                    return "N/A"
                avg = statistics.mean(vals)
                if len(vals) >= 2:
                    sd = statistics.stdev(vals)
                    return f"{avg:.0f} +/- {sd:.0f}"
                return f"{avg:.0f}"

            vram = f"{r.vram_mb:,.0f}" if r.vram_mb is not None else "N/A"

            table.add_row(
                r.backend,
                r.model,
                fmt(r.ttft_ms),
                fmt(r.tokens_per_second),
                fmt(r.total_time_ms),
                vram,
            )

        console.print(table)
        console.print()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    console.print(
        Panel(
            "[bold]Benchmark: Clase 8 — Ejecución Local vs Cloud[/bold]",
            style="bold blue",
        )
    )

    # 1. Discover healthy backends
    active_backends = discover_backends(args.backends)

    # 2. Build config
    config = BenchmarkConfig(
        n_runs=args.runs,
        warmup_runs=args.warmup,
        max_tokens=args.max_tokens,
        temperature=0.3,
    )

    console.print(
        f"\n[bold]Ejecutando benchmarks ({config.n_runs} runs + "
        f"{config.warmup_runs} warmup por prompt)...[/bold]\n"
    )

    # 3. Baseline VRAM
    baseline_vram = measure_vram_usage()
    if baseline_vram is not None:
        console.print(f"[dim]VRAM baseline: {baseline_vram:,.0f} MB[/dim]\n")

    # 4. Run benchmarks
    results = run_full_benchmark(active_backends, config)

    # 5. Display results
    console.print()
    print_rich_results(results, config)

    # 6. Export
    if args.out_format == "json":
        export_results_json(results, args.output)
        console.print(f"\n[bold green]Resultados guardados en: {args.output}.json[/bold green]")
    elif args.out_format == "csv":
        export_results_csv(results, args.output)
        console.print(f"\n[bold green]Resultados guardados en: {args.output}.csv[/bold green]")
    else:
        # Table to stdout (already printed above), also print plain text
        console.print(format_results(results))


if __name__ == "__main__":
    main()
