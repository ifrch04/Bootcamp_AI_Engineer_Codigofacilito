"""Pipelines declarativos para encadenar pasos de procesamiento.

Permite definir pipelines como secuencias de funciones decoradas con
``@pipeline_step``, con reintentos automáticos, timeouts y reportes de
ejecución.

Nota: los timeouts se implementan con ``ThreadPoolExecutor``. Los hilos
que exceden el timeout siguen ejecutándose en segundo plano (limitación
de Python) — el timeout solo controla cuánto espera el pipeline.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("orchestration.pipelines")

_current_pipeline_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_pipeline_name"
)


class StepTimeoutError(Exception):
    """Se lanza cuando un paso excede su tiempo límite."""

    def __init__(self, step_name: str, timeout: float) -> None:
        self.step_name = step_name
        self.timeout = timeout
        super().__init__(f"Step '{step_name}' timed out after {timeout}s")


@dataclass
class StepResult:
    """Resultado de la ejecución de un paso individual."""

    output: Any
    duration_seconds: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "metadata": self.metadata,
            "success": self.success,
        }


@dataclass
class PipelineResult:
    """Resultado agregado de la ejecución de un pipeline completo."""

    steps: list[StepResult] = field(default_factory=list)
    final_output: Any = None
    total_duration: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0

    @property
    def success(self) -> bool:
        return all(s.success for s in self.steps)

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Pipeline Status: {status}",
            f"Steps: {len(self.steps)}",
            f"Total Duration: {self.total_duration:.2f}s",
            f"Total Tokens: {self.total_tokens}",
            f"Total Cost: ${self.total_cost:.4f}",
            "",
            "Step Breakdown:",
        ]
        for i, step in enumerate(self.steps):
            mark = "OK" if step.success else "FAIL"
            lines.append(
                f"  [{i + 1}] {mark} — {step.duration_seconds:.2f}s"
                f" | tokens={step.tokens_used} | cost=${step.cost_usd:.4f}"
            )
            if step.error:
                lines.append(f"       Error: {step.error}")
        return "\n".join(lines)


def pipeline_step(
    name: str, max_retries: int = 2, timeout_seconds: float = 30.0
) -> Callable:
    """Decorador que convierte una función en un paso de pipeline.

    Args:
        name: Nombre del paso para logs y reportes.
        max_retries: Número máximo de reintentos tras fallo.
        timeout_seconds: Tiempo máximo de ejecución por intento.

    Returns:
        Decorador que envuelve la función y retorna un ``StepResult``.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> StepResult:
            try:
                pipeline_name = _current_pipeline_name.get()
                prefix = f"[{pipeline_name}.{name}]"
            except LookupError:
                prefix = f"[{name}]"

            last_error: str | None = None
            last_duration: float = 0.0

            for attempt in range(max_retries + 1):
                start = time.time()
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        result = future.result(timeout=timeout_seconds)
                    elapsed = time.time() - start
                    logger.info(
                        "%s completed in %.2fs (attempt %d)",
                        prefix,
                        elapsed,
                        attempt + 1,
                    )
                    return StepResult(output=result, duration_seconds=elapsed)
                except TimeoutError:
                    elapsed = time.time() - start
                    last_duration = elapsed
                    last_error = (
                        f"Step '{name}' timed out after {timeout_seconds}s"
                    )
                    logger.warning(
                        "%s timeout on attempt %d: %s",
                        prefix,
                        attempt + 1,
                        last_error,
                    )
                except Exception as exc:
                    elapsed = time.time() - start
                    last_duration = elapsed
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "%s error on attempt %d: %s",
                        prefix,
                        attempt + 1,
                        last_error,
                    )

                if attempt < max_retries:
                    delay = min(2**attempt + random.uniform(0, 1), 30)
                    logger.info("%s retrying in %.1fs...", prefix, delay)
                    time.sleep(delay)

            return StepResult(
                output=None, error=last_error, duration_seconds=last_duration
            )

        wrapper.step_name = name  # type: ignore[attr-defined]
        return wrapper

    return decorator


class Pipeline:
    """Pipeline declarativo que encadena pasos secuencialmente.

    Cada paso recibe la salida del paso anterior. Si un paso falla,
    el pipeline se detiene inmediatamente.

    Args:
        name: Nombre del pipeline para logs.
        steps: Lista de funciones decoradas con ``@pipeline_step``.
    """

    def __init__(self, name: str, steps: list[Callable]) -> None:
        self.name = name
        self.steps = steps

    @property
    def step_names(self) -> list[str]:
        return [
            getattr(s, "step_name", s.__name__) for s in self.steps
        ]

    def run(self, initial_input: Any) -> PipelineResult:
        """Ejecuta todos los pasos del pipeline secuencialmente.

        Args:
            initial_input: Entrada para el primer paso.

        Returns:
            ``PipelineResult`` con resultados de cada paso ejecutado.
        """
        return self._run_steps(self.steps, initial_input)

    def run_from(self, step_index: int, input_data: Any) -> PipelineResult:
        """Ejecuta el pipeline desde un paso específico.

        Args:
            step_index: Índice del paso desde el cual iniciar (0-based).
            input_data: Entrada para el paso inicial.

        Returns:
            ``PipelineResult`` con resultados de los pasos ejecutados.

        Raises:
            ValueError: Si el índice está fuera de rango.
        """
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(
                f"step_index {step_index} out of range "
                f"[0, {len(self.steps) - 1}]"
            )
        return self._run_steps(self.steps[step_index:], input_data)

    def _run_steps(
        self, steps: list[Callable], initial_input: Any
    ) -> PipelineResult:
        token = _current_pipeline_name.set(self.name)
        try:
            result = PipelineResult()
            current_input = initial_input

            for step_fn in steps:
                step_result: StepResult = step_fn(current_input)
                result.steps.append(step_result)

                if not step_result.success:
                    result.final_output = None
                    break

                current_input = step_result.output
            else:
                result.final_output = current_input

            result.total_duration = sum(
                s.duration_seconds for s in result.steps
            )
            result.total_tokens = sum(s.tokens_used for s in result.steps)
            result.total_cost = sum(s.cost_usd for s in result.steps)
            return result
        finally:
            _current_pipeline_name.reset(token)
