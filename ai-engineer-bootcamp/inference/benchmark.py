"""Benchmarking module — compare inference backends on TTFT, throughput, and VRAM."""

from __future__ import annotations

import csv
import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from tabulate import tabulate

from inference.local_adapter import InferenceBackend, chat_with_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default benchmark prompts
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS: list[dict] = [
    {
        "label": "simple_qa",
        "messages": [
            {"role": "system", "content": "Responde en español de forma concisa."},
            {"role": "user", "content": "¿Qué es Docker y para qué se usa?"},
        ],
    },
    {
        "label": "structured_output",
        "messages": [
            {
                "role": "system",
                "content": "Responde SOLO con JSON válido, sin texto adicional.",
            },
            {
                "role": "user",
                "content": (
                    "Genera un JSON con 3 tareas de un proyecto de software. "
                    "Cada tarea tiene: id (int), titulo (str), prioridad "
                    "(alta/media/baja), estimacion_horas (int)."
                ),
            },
        ],
    },
    {
        "label": "rag_synthesis",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente de documentación empresarial. "
                    "Sintetiza la información proporcionada."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Contexto del documento:\n\n"
                    "La empresa RavanTech fue fundada en enero 2025 en Tuxtla "
                    "Gutiérrez, Chiapas. Se especializa en IA aplicada, VR/AR y "
                    "desarrollo de software. Opera con un modelo mixto: fundadores "
                    "lideran I+D y proyectos premium, red 40/60 con colaboradores "
                    "para servicios. Sus proyectos actuales incluyen Kotoba "
                    "(plataforma de idiomas con IA), Project Mist (survival horror "
                    "en UE5) y servicios de digitalización para PyMEs.\n\n"
                    "Pregunta: ¿Cuáles son las principales líneas de negocio de "
                    "RavanTech y cómo se estructura su equipo?"
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    n_runs: int = 5
    warmup_runs: int = 1
    max_tokens: int = 256
    temperature: float = 0.3
    prompts: list[dict] = field(default_factory=lambda: list(BENCHMARK_PROMPTS))


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single backend + prompt combination."""

    backend: str
    model: str
    prompt_label: str
    ttft_ms: list[float] = field(default_factory=list)
    tokens_per_second: list[float] = field(default_factory=list)
    total_time_ms: list[float] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    vram_mb: float | None = None


# ---------------------------------------------------------------------------
# VRAM measurement
# ---------------------------------------------------------------------------


def measure_vram_usage() -> float | None:
    """Return current GPU VRAM usage in MB, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / 1024 / 1024
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def run_single_benchmark(
    backend: InferenceBackend,
    messages: list[dict],
    config: BenchmarkConfig,
    prompt_label: str = "unknown",
) -> BenchmarkResult:
    """Run warmup + measured iterations for one backend/prompt pair.

    Args:
        backend: The backend to benchmark.
        messages: Chat messages for the prompt.
        config: Benchmark configuration.
        prompt_label: Human-readable label for the prompt.

    Returns:
        A ``BenchmarkResult`` with per-run metrics.
    """
    from inference.model_registry import get_model_info

    model_info = get_model_info(backend)
    result = BenchmarkResult(
        backend=backend.value,
        model=model_info.model_id,
        prompt_label=prompt_label,
    )

    # Warmup runs (discarded)
    for i in range(config.warmup_runs):
        try:
            logger.debug("Warmup %d/%d for %s", i + 1, config.warmup_runs, backend.value)
            chat_with_metrics(
                backend=backend,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        except Exception as exc:
            logger.warning("Warmup run failed for %s: %s", backend.value, exc)

    # Measured runs
    for i in range(config.n_runs):
        try:
            logger.debug("Run %d/%d for %s", i + 1, config.n_runs, backend.value)
            metrics = chat_with_metrics(
                backend=backend,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            result.ttft_ms.append(metrics["ttft_ms"])
            result.tokens_per_second.append(metrics["tokens_per_second"])
            result.total_time_ms.append(metrics["total_time_ms"])
            result.completion_tokens.append(metrics["completion_tokens"])
        except Exception as exc:
            logger.warning("Run %d failed for %s: %s", i + 1, backend.value, exc)

    # Measure VRAM (only meaningful for local backends)
    if backend != InferenceBackend.GROQ:
        result.vram_mb = measure_vram_usage()

    return result


def run_full_benchmark(
    backends: list[InferenceBackend],
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Run benchmarks across multiple backends and prompts.

    Args:
        backends: List of backends to benchmark.
        config: Benchmark configuration. Uses defaults if None.

    Returns:
        List of ``BenchmarkResult`` objects (one per backend × prompt).
    """
    config = config or BenchmarkConfig()
    results: list[BenchmarkResult] = []

    for prompt_data in config.prompts:
        label = prompt_data["label"]
        messages = prompt_data["messages"]

        for backend in backends:
            logger.info("Benchmarking %s on prompt '%s'...", backend.value, label)
            result = run_single_benchmark(backend, messages, config, label)
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Formatting & export
# ---------------------------------------------------------------------------


def _stat_summary(values: list[float]) -> str:
    """Format a list of values as 'mean +/- stdev'."""
    if not values:
        return "N/A"
    avg = statistics.mean(values)
    if len(values) >= 2:
        sd = statistics.stdev(values)
        return f"{avg:.0f} +/- {sd:.0f}"
    return f"{avg:.0f}"


def format_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results into a readable table.

    Args:
        results: List of benchmark results.

    Returns:
        Formatted table string.
    """
    # Group by prompt label
    prompts_seen: list[str] = []
    for r in results:
        if r.prompt_label not in prompts_seen:
            prompts_seen.append(r.prompt_label)

    sections: list[str] = []

    for label in prompts_seen:
        rows = []
        for r in results:
            if r.prompt_label != label:
                continue
            tks = _stat_summary(r.tokens_per_second)
            vram = f"{r.vram_mb:,.0f}" if r.vram_mb is not None else "N/A"
            rows.append([
                r.backend,
                r.model,
                _stat_summary(r.ttft_ms),
                tks,
                _stat_summary(r.total_time_ms),
                vram,
            ])

        headers = ["Backend", "Modelo", "TTFT (ms)", "Tks/s", "Total (ms)", "VRAM (MB)"]
        table = tabulate(rows, headers=headers, tablefmt="simple")
        sections.append(f"\nPrompt: {label}\n{'─' * 66}\n{table}\n")

    return "\n".join(sections)


def _result_to_dict(r: BenchmarkResult) -> dict:
    """Convert a BenchmarkResult to a serializable dict with summary stats."""
    def _safe_mean(vals: list) -> float | None:
        return round(statistics.mean(vals), 2) if vals else None

    def _safe_stdev(vals: list) -> float | None:
        return round(statistics.stdev(vals), 2) if len(vals) >= 2 else None

    return {
        "backend": r.backend,
        "model": r.model,
        "prompt_label": r.prompt_label,
        "ttft_ms_mean": _safe_mean(r.ttft_ms),
        "ttft_ms_stdev": _safe_stdev(r.ttft_ms),
        "tokens_per_second_mean": _safe_mean(r.tokens_per_second),
        "tokens_per_second_stdev": _safe_stdev(r.tokens_per_second),
        "total_time_ms_mean": _safe_mean(r.total_time_ms),
        "total_time_ms_stdev": _safe_stdev(r.total_time_ms),
        "completion_tokens_mean": _safe_mean([float(t) for t in r.completion_tokens]),
        "vram_mb": round(r.vram_mb, 1) if r.vram_mb is not None else None,
        "n_runs": len(r.ttft_ms),
        "raw": {
            "ttft_ms": r.ttft_ms,
            "tokens_per_second": r.tokens_per_second,
            "total_time_ms": r.total_time_ms,
            "completion_tokens": r.completion_tokens,
        },
    }


def export_results_json(results: list[BenchmarkResult], filepath: str | Path) -> None:
    """Save benchmark results to a JSON file.

    Args:
        results: List of benchmark results.
        filepath: Output file path (without extension).
    """
    filepath = Path(filepath).with_suffix(".json")
    data = [_result_to_dict(r) for r in results]
    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved to %s", filepath)


def export_results_csv(results: list[BenchmarkResult], filepath: str | Path) -> None:
    """Save benchmark results to a CSV file.

    Args:
        results: List of benchmark results.
        filepath: Output file path (without extension).
    """
    filepath = Path(filepath).with_suffix(".csv")
    rows = [_result_to_dict(r) for r in results]
    # Flatten — exclude raw data for CSV
    fieldnames = [k for k in rows[0] if k != "raw"] if rows else []

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "raw"})

    logger.info("Results saved to %s", filepath)
