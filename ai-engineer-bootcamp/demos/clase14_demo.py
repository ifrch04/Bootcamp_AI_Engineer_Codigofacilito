"""Demo orquestador de la Clase 14: ejecuta los 5 niveles de testing en serie.

Pensado para correrse en vivo frente a estudiantes con un único comando:
    `make demo-clase14`

Cada nivel imprime un banner, ejecuta su contenido y deja un segundo de
respiro antes del siguiente. Al final muestra un resumen global.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from rich.console import Console

from demos import fake_rag
from evals.datasets import CURATED_DATASET
from evals.regression import run_regression
from evals.runner import run_eval
from evals.trace_replay import replay_trace


_console = Console()


def _banner(level: int, title: str) -> None:
    """Imprime un banner llamativo entre niveles."""
    bar = "=" * 70
    _console.print()
    _console.print(f"[bold cyan]{bar}[/bold cyan]")
    _console.print(f"[bold cyan]   NIVEL {level} — {title}[/bold cyan]")
    _console.print(f"[bold cyan]{bar}[/bold cyan]")
    _console.print()


def _final_summary(
    eval_summary: dict[str, Any],
    regression_summary: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    """Imprime el resumen final del demo."""
    bar = "=" * 70
    _console.print()
    _console.print(f"[bold green]{bar}[/bold green]")
    _console.print("[bold green]   RESUMEN FINAL — CLASE 14[/bold green]")
    _console.print(f"[bold green]{bar}[/bold green]")
    _console.print()
    _console.print("[bold]5/5 niveles ejecutados.[/bold]")
    _console.print(
        f"  Eval offline: [bold]{eval_summary['passed']}/{eval_summary['total']}[/bold] "
        f"([bold]{eval_summary['pass_rate'] * 100:.1f}%[/bold])"
    )
    regression_status = (
        "[red]REGRESIÓN DETECTADA[/red]"
        if regression_summary["regressed"]
        else "[green]sin regresión[/green]"
    )
    _console.print(
        f"  Regresión:    Δ promedio = [bold]{regression_summary['mean_delta']:+.4f}[/bold]  "
        f"({regression_status})"
    )
    _console.print(f"  Tiempo total: [bold]{elapsed_seconds:.1f}s[/bold]")
    _console.print()


def main() -> int:
    """Punto de entrada del demo orquestador."""
    started = time.perf_counter()

    _banner(1, "CONTRATOS")
    pytest.main(["tests/test_contracts.py", "-v"])
    time.sleep(1)

    _banner(2, "GOLDEN TESTS")
    pytest.main(["tests/test_goldens.py", "-v", "--tb=short"])
    time.sleep(1)

    _banner(3, "REPLAY DE TRAZAS")
    replay_trace("evals/traces/trace_001.json", fake_rag.answer)
    time.sleep(1)

    _banner(4, "REGRESIÓN")
    regression_summary = run_regression(
        CURATED_DATASET, fake_rag.answer, fake_rag.answer_v2
    )
    time.sleep(1)

    _banner(5, "EVAL OFFLINE")
    eval_summary = run_eval(fake_rag.answer)
    time.sleep(1)

    elapsed = time.perf_counter() - started
    _final_summary(eval_summary, regression_summary, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
