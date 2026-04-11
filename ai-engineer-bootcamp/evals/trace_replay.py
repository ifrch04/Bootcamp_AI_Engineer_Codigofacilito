"""Replay de trazas: re-ejecuta un sistema sobre una traza grabada y compara."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from evals.golden_test import check_golden


def replay_trace(
    trace_path: str,
    system_fn: Callable[[str], dict[str, Any]],
    min_sim: float = 0.55,
) -> dict[str, Any]:
    """Re-ejecuta `system_fn` con la query de la traza y compara las respuestas.

    Args:
        trace_path: Ruta a un archivo JSON producido por `capture_trace`.
        system_fn: Función a invocar (firma idéntica a la usada en captura).
        min_sim: Umbral de similitud para considerar el replay exitoso.

    Returns:
        Dict con `passed`, `similarity`, `original`, `replayed`, `query`.
    """
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    query = trace["query"]
    original_answer = trace["output"]["answer"]

    replayed = system_fn(query)
    replayed_answer = replayed["answer"]

    report = check_golden(
        actual=replayed_answer,
        golden=original_answer,
        keywords=[],
        min_sim=min_sim,
    )

    result = {
        "trace_id": trace.get("trace_id", "unknown"),
        "query": query,
        "original": original_answer,
        "replayed": replayed_answer,
        "similarity": report["similarity"],
        "min_sim": min_sim,
        "passed": report["sim_pass"],
    }

    _print_replay_report(result)
    return result


def _print_replay_report(result: dict[str, Any]) -> None:
    """Imprime una tabla con los resultados del replay usando rich."""
    console = Console()
    table = Table(
        title=f"Replay de traza: {result['trace_id']}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Campo", style="bold")
    table.add_column("Valor")

    table.add_row("Query", result["query"])
    table.add_row("Original", result["original"])
    table.add_row("Replay", result["replayed"])
    table.add_row(
        "Similitud",
        f"{result['similarity']} (mín {result['min_sim']})",
    )
    status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
    table.add_row("Resultado", status)

    console.print(table)
