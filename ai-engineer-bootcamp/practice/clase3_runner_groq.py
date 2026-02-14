#!/usr/bin/env python3
"""Clase 3 runner (Groq only) — imprime cada respuesta en tiempo real."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.llm_client import LLMClient  # noqa: E402
from prompting.promptkit import (  # noqa: E402
    EvalMetrics,
    PromptChain,
    PromptTemplate,
    _extract_json,
    _normalize,
)
from prompting.templates.ticket_classifier import (  # noqa: E402
    registry,
    v4_chain,
)


def _load_golden_set() -> list[dict]:
    path = Path(__file__).resolve().parent / "golden_set_tickets.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_and_print(
    label: str,
    prompt_or_chain: PromptTemplate | PromptChain,
    client: LLMClient,
    golden_set: list[dict],
    delay: float,
) -> EvalMetrics:
    """Evaluate a prompt against the golden set, printing each response live."""
    total = len(golden_set)
    json_ok = 0
    correct = 0
    campos_correctos = 0
    campos_total = 0
    total_tokens = 0
    total_latency = 0.0
    details: list[dict] = []

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    for idx, item in enumerate(golden_set):
        input_text = item["input"]
        expected = item["expected"]

        print(f"\n  [{idx + 1}/{total}] {input_text}")

        try:
            if isinstance(prompt_or_chain, PromptChain):
                chain_result = prompt_or_chain.run(
                    client, {"ticket": input_text}
                )
                response_text = chain_result.final_response
                tokens = chain_result.total_tokens
                latency_s = chain_result.total_latency_s
            else:
                rendered = prompt_or_chain.render(ticket=input_text)
                result = client.chat(rendered)
                response_text = result["response"]
                tokens = result["metadata"]["usage"]["total_tokens"]
                latency_s = result["metadata"]["latency_ms"] / 1000
        except Exception as exc:
            print(f"    ERROR: {exc}")
            details.append({
                "input": input_text, "expected": expected,
                "response": str(exc), "parsed": None,
                "correct": False, "json_valid": False,
            })
            campos_total += 2
            if delay > 0 and idx < total - 1:
                time.sleep(delay)
            continue

        total_tokens += tokens
        total_latency += latency_s

        parsed = _extract_json(response_text)
        json_valid = parsed is not None
        if json_valid:
            json_ok += 1

        cat_match = False
        pri_match = False

        if parsed:
            cat_match = (
                _normalize(str(parsed.get("categoria", "")))
                == _normalize(str(expected.get("categoria", "")))
            )
            pri_match = (
                _normalize(str(parsed.get("prioridad", "")))
                == _normalize(str(expected.get("prioridad", "")))
            )
            if cat_match:
                campos_correctos += 1
            if pri_match:
                campos_correctos += 1

        campos_total += 2
        if cat_match and pri_match:
            correct += 1

        # -- Print result immediately ------------------------------------
        status = "OK" if (cat_match and pri_match) else "FAIL"
        if parsed:
            print(f"    Respuesta: {parsed}")
            print(f"    Esperado : {expected}")
            cat_icon = "ok" if cat_match else "x"
            pri_icon = "ok" if pri_match else "x"
            print(f"    [{status}] categoria: {cat_icon} | prioridad: {pri_icon}  ({latency_s:.2f}s, {tokens} tokens)")
        else:
            print(f"    Respuesta (no JSON): {response_text[:150]}")
            print(f"    [{status}] JSON invalido  ({latency_s:.2f}s, {tokens} tokens)")

        details.append({
            "input": input_text, "expected": expected,
            "response": response_text, "parsed": parsed,
            "correct": cat_match and pri_match, "json_valid": json_valid,
            "cat_match": cat_match, "pri_match": pri_match,
            "tokens": tokens, "latency_s": latency_s,
        })

        if delay > 0 and idx < total - 1:
            time.sleep(delay)

    metrics = EvalMetrics(
        accuracy=correct / total if total else 0,
        json_parse_rate=json_ok / total if total else 0,
        campos_correctos_rate=campos_correctos / campos_total if campos_total else 0,
        tokens_promedio=total_tokens / total if total else 0,
        latencia_promedio=total_latency / total if total else 0,
        details=details,
    )

    print(f"\n  Resumen {label}:")
    print(
        f"    Accuracy: {metrics.accuracy:.0%} | "
        f"JSON valido: {metrics.json_parse_rate:.0%} | "
        f"Tokens avg: {metrics.tokens_promedio:.0f} | "
        f"Latencia avg: {metrics.latencia_promedio:.1f}s"
    )
    return metrics


def _print_table(rows: list[tuple[str, str, EvalMetrics]]) -> None:
    header = (
        f"| {'Version':<18}| {'Accuracy':<10}"
        f"| {'JSON valido %':<15}| {'Tokens avg':<12}| {'Latencia avg':<13}|"
    )
    sep = "|" + "-" * 19 + "|" + "-" * 11 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 14 + "|"

    print()
    print(header)
    print(sep)
    for label, _, m in rows:
        print(
            f"| {label:<18}"
            f"| {m.accuracy:>7.0%}   "
            f"| {m.json_parse_rate:>10.0%}     "
            f"| {m.tokens_promedio:>9.0f}   "
            f"| {m.latencia_promedio:>9.1f}s    |"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clase 3: Prompt Engineering — Groq only",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Segundos de pausa entre llamadas (default: 1.0)",
    )
    args = parser.parse_args()

    golden_set = _load_golden_set()

    client = LLMClient(provider="groq")
    model = client.model
    print(f"Backend: Groq ({model})")
    print(f"Golden set: {len(golden_set)} tickets")

    evaluations: list[tuple[str, PromptTemplate | PromptChain]] = [
        ("v1 (base)", registry.get("ticket_classifier_v1")),
        ("v2 (few-shot)", registry.get("ticket_classifier_v2")),
        ("v3 (restricc.)", registry.get("ticket_classifier_v3")),
        ("v4 (chain)", v4_chain),
    ]

    results: list[tuple[str, str, EvalMetrics]] = []

    for label, prompt_obj in evaluations:
        metrics = run_and_print(label, prompt_obj, client, golden_set, args.delay)
        results.append((label, "Groq", metrics))

    # -- Final summary table -----------------------------------------------
    print(f"\n{'=' * 75}")
    print(f"  RESULTADOS COMPARATIVOS — Groq ({model})")
    print(f"{'=' * 75}")
    _print_table(results)


if __name__ == "__main__":
    main()
