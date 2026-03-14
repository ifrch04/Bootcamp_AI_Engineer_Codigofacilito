"""Tests de integración para el módulo de inferencia (Clase 8).

Ejecutan inferencia REAL contra los backends disponibles y muestran
las respuestas del modelo junto con métricas de rendimiento.

Uso:
    python -m pytest tests/test_inference.py -v -s
"""

from __future__ import annotations

import pytest

from inference.local_adapter import InferenceBackend, chat, chat_with_metrics, get_model_name
from inference.model_registry import check_backend_health, get_model_info, list_available_models
from inference.benchmark import BenchmarkConfig, run_single_benchmark, format_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_backends() -> list[InferenceBackend]:
    """Detecta qué backends están activos ahora mismo."""
    active = []
    for b in InferenceBackend:
        if check_backend_health(b):
            active.append(b)
    return active


# Skip si no hay ningún backend disponible
_BACKENDS = _available_backends()
_skip_no_backends = pytest.mark.skipif(not _BACKENDS, reason="No hay backends disponibles")


def _backend_available(backend: InferenceBackend) -> bool:
    return backend in _BACKENDS


# ---------------------------------------------------------------------------
# Test 1: Health check — mostrar qué backends están vivos
# ---------------------------------------------------------------------------

class TestBackendDiscovery:
    def test_discover_backends(self):
        """Muestra el estado de cada backend."""
        print("\n" + "=" * 60)
        print("DISCOVERY: Estado de backends")
        print("=" * 60)
        for b in InferenceBackend:
            info = get_model_info(b)
            healthy = b in _BACKENDS
            status = "OK" if healthy else "NO DISPONIBLE"
            print(f"  [{status:>14}] {b.value:<10} -> {info.model_id}")
        print()
        # Al menos un backend debe estar activo para que los tests tengan sentido
        assert len(_BACKENDS) >= 1, "Necesitas al menos un backend activo"


# ---------------------------------------------------------------------------
# Test 2: Chat real — el modelo contesta
# ---------------------------------------------------------------------------

@_skip_no_backends
class TestChatReal:
    def test_chat_simple(self):
        """Envía un mensaje simple y muestra la respuesta del modelo."""
        backend = _BACKENDS[0]
        model = get_model_name(backend)

        print("\n" + "=" * 60)
        print(f"CHAT SIMPLE: {backend.value} ({model})")
        print("=" * 60)

        response = chat(
            backend=backend,
            messages=[
                {"role": "system", "content": "Responde en español, máximo 2 oraciones."},
                {"role": "user", "content": "¿Qué es Docker y para qué se usa?"},
            ],
            max_tokens=150,
            temperature=0.3,
        )

        print(f"\n  Pregunta: ¿Qué es Docker y para qué se usa?")
        print(f"  Respuesta: {response}")
        print()

        assert isinstance(response, str)
        assert len(response) > 10, "La respuesta es demasiado corta"

    def test_chat_structured_output(self):
        """Pide JSON estructurado al modelo."""
        backend = _BACKENDS[0]
        model = get_model_name(backend)

        print("\n" + "=" * 60)
        print(f"STRUCTURED OUTPUT: {backend.value} ({model})")
        print("=" * 60)

        response = chat(
            backend=backend,
            messages=[
                {"role": "system", "content": "Responde SOLO con JSON válido, sin texto adicional."},
                {"role": "user", "content": "Genera un JSON con 2 tareas de software. Cada tarea: id (int), titulo (str), prioridad (alta/media/baja)."},
            ],
            max_tokens=200,
            temperature=0.1,
        )

        print(f"\n  Respuesta JSON:\n  {response}")
        print()

        assert isinstance(response, str)
        assert "{" in response, "La respuesta no contiene JSON"


# ---------------------------------------------------------------------------
# Test 3: Chat con métricas — TTFT, tokens/s, tiempo total
# ---------------------------------------------------------------------------

@_skip_no_backends
class TestChatWithMetricsReal:
    def test_metrics_single_backend(self):
        """Mide TTFT, throughput y tiempo total con streaming real."""
        backend = _BACKENDS[0]
        model = get_model_name(backend)

        print("\n" + "=" * 60)
        print(f"METRICAS: {backend.value} ({model})")
        print("=" * 60)

        result = chat_with_metrics(
            backend=backend,
            messages=[
                {"role": "system", "content": "Responde en español de forma concisa."},
                {"role": "user", "content": "Explica qué es Kubernetes en 3 oraciones."},
            ],
            max_tokens=200,
            temperature=0.3,
        )

        print(f"\n  Respuesta: {result['content']}")
        print(f"\n  --- Métricas ---")
        print(f"  TTFT:              {result['ttft_ms']:.1f} ms")
        print(f"  Tiempo total:      {result['total_time_ms']:.1f} ms")
        print(f"  Tokens generados:  {result['completion_tokens']}")
        print(f"  Throughput:        {result['tokens_per_second']:.1f} tokens/s")
        print()

        assert result["content"], "El modelo no generó respuesta"
        assert result["ttft_ms"] > 0, "TTFT debe ser positivo"
        assert result["total_time_ms"] > 0, "Tiempo total debe ser positivo"
        assert result["completion_tokens"] > 0, "Debe generar al menos un token"

    @pytest.mark.skipif(len(_BACKENDS) < 2, reason="Se necesitan 2+ backends para comparar")
    def test_metrics_compare_backends(self):
        """Compara métricas entre múltiples backends disponibles."""
        print("\n" + "=" * 60)
        print("COMPARACION DE BACKENDS")
        print("=" * 60)

        messages = [
            {"role": "system", "content": "Responde en español, máximo 2 oraciones."},
            {"role": "user", "content": "¿Qué ventajas tiene Python sobre otros lenguajes?"},
        ]

        for backend in _BACKENDS:
            model = get_model_name(backend)
            result = chat_with_metrics(
                backend=backend,
                messages=messages,
                max_tokens=150,
                temperature=0.3,
            )

            print(f"\n  [{backend.value}] {model}")
            print(f"    Respuesta: {result['content'][:100]}...")
            print(f"    TTFT: {result['ttft_ms']:.1f} ms | "
                  f"Total: {result['total_time_ms']:.1f} ms | "
                  f"Throughput: {result['tokens_per_second']:.1f} tks/s")

        print()


# ---------------------------------------------------------------------------
# Test 4: Benchmark real — tabla comparativa
# ---------------------------------------------------------------------------

@_skip_no_backends
class TestBenchmarkReal:
    def test_run_benchmark(self):
        """Ejecuta un benchmark corto (2 runs) y muestra la tabla de resultados."""
        config = BenchmarkConfig(
            n_runs=2,
            warmup_runs=1,
            max_tokens=128,
            temperature=0.3,
            prompts=[
                {
                    "label": "simple_qa",
                    "messages": [
                        {"role": "system", "content": "Responde en español de forma concisa."},
                        {"role": "user", "content": "¿Qué es Docker y para qué se usa?"},
                    ],
                },
                {
                    "label": "rag_synthesis",
                    "messages": [
                        {"role": "system", "content": "Sintetiza la información proporcionada."},
                        {"role": "user", "content": (
                            "Contexto: RavanTech fue fundada en 2025 en Chiapas. "
                            "Se especializa en IA aplicada y desarrollo de software. "
                            "Pregunta: ¿A qué se dedica RavanTech?"
                        )},
                    ],
                },
            ],
        )

        print("\n" + "=" * 60)
        print(f"BENCHMARK: {len(_BACKENDS)} backend(s), {config.n_runs} runs + {config.warmup_runs} warmup")
        print("=" * 60)

        results = []
        for backend in _BACKENDS:
            for prompt in config.prompts:
                result = run_single_benchmark(
                    backend=backend,
                    messages=prompt["messages"],
                    config=config,
                    prompt_label=prompt["label"],
                )
                results.append(result)

        table = format_results(results)
        print(table)

        assert len(results) > 0, "No se generaron resultados"
        for r in results:
            assert len(r.ttft_ms) > 0, f"Sin métricas para {r.backend}/{r.prompt_label}"


# ---------------------------------------------------------------------------
# Test 5: Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_list_models(self):
        """Muestra todos los modelos registrados."""
        models = list_available_models()

        print("\n" + "=" * 60)
        print("REGISTRY: Modelos registrados")
        print("=" * 60)
        for m in models:
            q = m.quantization or "full"
            print(f"  {m.backend.value:<10} {m.model_id:<25} {m.parameters_billions}B ({q})")
        print()

        assert len(models) == len(InferenceBackend)
