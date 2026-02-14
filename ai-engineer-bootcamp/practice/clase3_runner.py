#!/usr/bin/env python3
"""Clase 3 runner — Prompt Engineering evaluation across 4 template versions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.llm_client import LLMClient  # noqa: E402
from prompting.promptkit import evaluate_prompt  # noqa: E402
from prompting.templates.ticket_classifier import (  # noqa: E402
    registry,
    v4_chain,
)


def _load_golden_set() -> list[dict]:
    path = Path(__file__).resolve().parent / "golden_set_tickets.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_table(rows: list[tuple[str, str, object]]) -> None:
    """Print a comparison table to stdout."""
    header = (
        f"| {'Versión':<18}| {'Backend':<9}| {'Accuracy':<10}"
        f"| {'JSON válido %':<15}| {'Tokens avg':<12}| {'Latencia avg':<13}|"
    )
    sep = "|" + "-" * 19 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 14 + "|"

    print()
    print(header)
    print(sep)
    for label, backend, m in rows:
        print(
            f"| {label:<18}"
            f"| {backend:<9}"
            f"| {m.accuracy:>7.0%}   "
            f"| {m.json_parse_rate:>10.0%}     "
            f"| {m.tokens_promedio:>9.0f}   "
            f"| {m.latencia_promedio:>9.1f}s    |"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clase 3: Prompt Engineering — Evaluación comparativa",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Mostrar detalle ticket por ticket",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Segundos de pausa entre llamadas (default: 1.0)",
    )
    args = parser.parse_args()

    golden_set = _load_golden_set()
    print(f"Golden set cargado: {len(golden_set)} tickets\n")

    # -- Gemma evaluations -------------------------------------------------
    gemma_client = LLMClient(provider="gemini")
    gemma_label = f"Gemma ({gemma_client.model})"

    evaluations: list[tuple[str, object]] = [
        ("v1 (base)", registry.get("ticket_classifier_v1")),
        ("v2 (few-shot)", registry.get("ticket_classifier_v2")),
        ("v3 (restricc.)", registry.get("ticket_classifier_v3")),
        ("v4 (chain)", v4_chain),
    ]

    results: list[tuple[str, str, object]] = []

    for label, prompt_obj in evaluations:
        print(f"Evaluando: {label} con {gemma_label} ...")
        metrics = evaluate_prompt(
            prompt_obj, gemma_client, golden_set,
            verbose=args.verbose, delay=args.delay,
        )
        results.append((label, "Gemma", metrics))
        print(
            f"  -> Accuracy: {metrics.accuracy:.0%} | "
            f"JSON válido: {metrics.json_parse_rate:.0%} | "
            f"Tokens avg: {metrics.tokens_promedio:.0f} | "
            f"Latencia avg: {metrics.latencia_promedio:.1f}s\n"
        )

    # -- Groq evaluation (best version) -----------------------------------
    best_label, best_prompt = max(
        zip(
            [r[0] for r in results],
            [e[1] for e in evaluations],
        ),
        key=lambda pair: next(
            r[2].accuracy for r in results if r[0] == pair[0]
        ),
    )

    try:
        groq_client = LLMClient(provider="groq")
        groq_label = f"Groq ({groq_client.model})"
        print(f"Evaluando: {best_label} con {groq_label} (mejor versión) ...")
        groq_metrics = evaluate_prompt(
            best_prompt, groq_client, golden_set,
            verbose=args.verbose, delay=args.delay,
        )
        results.append((best_label, "Groq", groq_metrics))
        print(
            f"  -> Accuracy: {groq_metrics.accuracy:.0%} | "
            f"JSON válido: {groq_metrics.json_parse_rate:.0%} | "
            f"Tokens avg: {groq_metrics.tokens_promedio:.0f} | "
            f"Latencia avg: {groq_metrics.latencia_promedio:.1f}s\n"
        )
    except Exception as exc:
        print(f"Groq no disponible: {exc}\n")

    # -- Summary table -----------------------------------------------------
    print("=" * 88)
    print("  RESULTADOS COMPARATIVOS — Clase 3: Prompt Engineering")
    print("=" * 88)
    _print_table(results)


if __name__ == "__main__":
    main()
