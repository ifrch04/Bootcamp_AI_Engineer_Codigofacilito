"""Nivel 5 de testing: orquestador de evaluación offline.

Corre un sistema bajo prueba contra un dataset completo, agrega resultados
por categoría y guarda un reporte JSON reproducible.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from evals.datasets import CURATED_DATASET
from evals.golden_test import check_golden


def run_eval(
    system_fn: Callable[[str], dict[str, Any]],
    dataset: list[dict[str, Any]] | None = None,
    output_path: str = "evals/results/results.json",
) -> dict[str, Any]:
    """Ejecuta el sistema sobre el dataset y agrega resultados.

    Args:
        system_fn: Función a evaluar (acepta str, devuelve dict con `answer`).
        dataset: Lista de casos. Si es None, usa `CURATED_DATASET`.
        output_path: Ruta donde se guarda el JSON con los resultados.

    Returns:
        Dict con `results`, `pass_rate`, `by_category`, `total`, `passed`.
    """
    cases = dataset if dataset is not None else CURATED_DATASET

    results: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "total": 0}
    )

    for case in cases:
        query = case["query"]
        golden = case["golden_answer"]
        keywords = case.get("required_keywords", [])
        min_sim = case.get("min_similarity", 0.55)
        category = case.get("category", "uncategorized")

        try:
            output = system_fn(query)
            actual = output["answer"]
            error = None
        except Exception as exc:
            actual = ""
            error = f"{type(exc).__name__}: {exc}"

        if error is None:
            report = check_golden(actual, golden, keywords, min_sim)
            passed = report["overall_pass"]
            similarity = report["similarity"]
            keywords_missing = report["keywords_missing"]
        else:
            passed = False
            similarity = 0.0
            keywords_missing = keywords

        results.append(
            {
                "id": case["id"],
                "category": category,
                "query": query,
                "actual": actual,
                "golden": golden,
                "similarity": similarity,
                "min_sim": min_sim,
                "keywords_missing": keywords_missing,
                "passed": passed,
                "error": error,
            }
        )

        by_category[category]["total"] += 1
        if passed:
            by_category[category]["passed"] += 1

    total = len(results)
    passed_total = sum(1 for r in results if r["passed"])
    pass_rate = passed_total / total if total else 0.0

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": passed_total,
        "pass_rate": round(pass_rate, 4),
        "by_category": {
            cat: {
                "passed": stats["passed"],
                "total": stats["total"],
                "pass_rate": round(
                    stats["passed"] / stats["total"] if stats["total"] else 0.0, 4
                ),
            }
            for cat, stats in by_category.items()
        },
        "results": results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _print_summary(summary, output_path)
    return summary


def _print_summary(summary: dict[str, Any], output_path: str) -> None:
    """Imprime resumen agregado y por categoría."""
    console = Console()

    detail_table = Table(
        title="Resultados por caso",
        show_header=True,
        header_style="bold cyan",
    )
    detail_table.add_column("ID")
    detail_table.add_column("Categoría")
    detail_table.add_column("Sim", justify="right")
    detail_table.add_column("Resultado")

    for row in summary["results"]:
        status = (
            "[green]PASS[/green]" if row["passed"] else "[red]FAIL[/red]"
        )
        detail_table.add_row(
            row["id"],
            row["category"],
            f"{row['similarity']:.3f}",
            status,
        )

    console.print(detail_table)

    cat_table = Table(
        title="Resumen por categoría",
        show_header=True,
        header_style="bold magenta",
    )
    cat_table.add_column("Categoría")
    cat_table.add_column("Pasados", justify="right")
    cat_table.add_column("Total", justify="right")
    cat_table.add_column("Pass rate", justify="right")

    for cat, stats in summary["by_category"].items():
        cat_table.add_row(
            cat,
            str(stats["passed"]),
            str(stats["total"]),
            f"{stats['pass_rate'] * 100:.1f}%",
        )

    console.print(cat_table)
    console.print(
        f"\n[bold]TOTAL:[/bold] {summary['passed']}/{summary['total']} "
        f"([bold]{summary['pass_rate'] * 100:.1f}%[/bold])"
    )
    console.print(f"[dim]Reporte guardado en: {output_path}[/dim]")


if __name__ == "__main__":
    from demos.fake_rag import answer

    run_eval(answer)
