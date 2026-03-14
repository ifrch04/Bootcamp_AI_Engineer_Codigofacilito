"""Módulo de orquestación para pipelines declarativos y registro de herramientas.

Provee las bases para construir agentes ReAct, multi-agentes y
Human-in-the-Loop usando solo la biblioteca estándar de Python.
"""

from orchestration.pipelines import (
    Pipeline,
    PipelineResult,
    StepResult,
    StepTimeoutError,
    pipeline_step,
)
from orchestration.tools import (
    ToolDefinition,
    ToolRegistry,
)

__all__ = [
    "Pipeline",
    "PipelineResult",
    "StepResult",
    "StepTimeoutError",
    "pipeline_step",
    "ToolDefinition",
    "ToolRegistry",
]
