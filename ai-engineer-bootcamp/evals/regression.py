"""Nivel 4 de testing: regresión entre dos versiones del sistema."""

from __future__ import annotations

from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from evals.golden_test import cosine_similarity


def run_regression(
    dataset: list[dict[str, Any]],
    fn_old: Callable[[str], dict[str, Any]],
    fn_new: Callable[[str], dict[str, Any]],
    fail_threshold: float = -0.05,
) -> dict[str, Any]:
    """Compara dos versiones del sistema sobre el mismo dataset.

    Args:
        dataset: Lista de casos con `id`, `query` y `golden_answer`.
        fn_old: Versión "estable" actualmente en producción.
        fn_new: Versión candidata.
        fail_threshold: Si el delta promedio cae por debajo de este valor,
            el reporte marca `regressed=True`.

    Returns:
        Dict con `rows`, `mean_delta`, `worst_case`, `best_case`,
        `pct_degraded`, `regressed`.
    """
    rows: list[dict[str, Any]] = []

    for case in dataset:
        query = case["query"]
        golden = case["golden_answer"]

        old_answer = fn_old(query)["answer"]
        new_answer = fn_new(query)["answer"]

        sim_old = cosine_similarity(old_answer, golden)
        sim_new = cosine_similarity(new_answer, golden)
        delta = sim_new - sim_old

        rows.append(
            {
                "id": case["id"],
                "query": query,
                "sim_old": round(sim_old, 4),
                "sim_new": round(sim_new, 4),
                "delta": round(delta, 4),
            }
        )

    deltas = [r["delta"] for r in rows]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    worst_case = min(rows, key=lambda r: r["delta"]) if rows else None
    best_case = max(rows, key=lambda r: r["delta"]) if rows else None
    degraded_count = sum(1 for d in deltas if d < 0)
    pct_degraded = degraded_count / len(deltas) if deltas else 0.0

    summary = {
        "rows": rows,
        "mean_delta": round(mean_delta, 4),
        "worst_case": worst_case,
        "best_case": best_case,
        "pct_degraded": round(pct_degraded, 4),
        "regressed": mean_delta < fail_threshold,
        "fail_threshold": fail_threshold,
    }

    _print_regression_report(summary)
    return summary


def _delta_color(delta: float) -> str:
    """Asigna un color rich según el signo y magnitud del delta."""
    if delta >= 0:
        return "green"
    if delta >= -0.05:
        return "yellow"
    return "red"


def _print_regression_report(summary: dict[str, Any]) -> None:
    """Imprime tabla rich con la comparación caso por caso + métricas globales."""
    console = Console()

    table = Table(
        title="Reporte de regresión: old vs new",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="bold")
    table.add_column("Query", overflow="fold", max_width=42)
    table.add_column("sim_old", justify="right")
    table.add_column("sim_new", justify="right")
    table.add_column("Δ", justify="right")

    for row in summary["rows"]:
        color = _delta_color(row["delta"])
        delta_str = f"[{color}]{row['delta']:+.4f}[/{color}]"
        table.add_row(
            row["id"],
            row["query"],
            f"{row['sim_old']:.4f}",
            f"{row['sim_new']:.4f}",
            delta_str,
        )

    console.print(table)

    console.print(
        f"\n[bold]Δ promedio:[/bold] {summary['mean_delta']:+.4f}    "
        f"[bold]% degradados:[/bold] {summary['pct_degraded'] * 100:.1f}%"
    )
    if summary["worst_case"]:
        wc = summary["worst_case"]
        console.print(
            f"[bold red]Peor caso:[/bold red] {wc['id']} (Δ={wc['delta']:+.4f})"
        )
    if summary["best_case"]:
        bc = summary["best_case"]
        console.print(
            f"[bold green]Mejor caso:[/bold green] {bc['id']} (Δ={bc['delta']:+.4f})"
        )

    if summary["regressed"]:
        console.print(
            f"\n[bold red]REGRESIÓN DETECTADA[/bold red] "
            f"(Δ promedio < {summary['fail_threshold']})"
        )
    else:
        console.print(
            f"\n[bold green]Sin regresión significativa[/bold green] "
            f"(umbral {summary['fail_threshold']})"
        )
