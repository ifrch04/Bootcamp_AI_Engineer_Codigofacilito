"""Esquemas Pydantic v2 para la Clase 14 (Testing & Reliability)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocAnswer(BaseModel):
    """Contrato de respuesta para sistemas RAG documentales.

    Garantiza que toda salida del pipeline tenga texto, al menos una fuente
    citada y una confianza acotada en [0, 1]. Es el "Nivel 1" de testing:
    si el LLM se sale de este contrato, falla en validación antes de llegar
    al usuario.
    """

    answer: str = Field(
        min_length=5,
        description="Respuesta textual al usuario.",
    )
    sources: list[str] = Field(
        min_length=1,
        description="IDs de los documentos citados como evidencia.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confianza del modelo en la respuesta, en el rango [0, 1].",
    )
