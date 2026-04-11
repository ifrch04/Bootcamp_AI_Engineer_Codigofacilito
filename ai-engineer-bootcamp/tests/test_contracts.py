"""Nivel 1 de testing: validación de contratos Pydantic.

El contrato `DocAnswer` actúa como puerta de entrada al sistema. Cualquier
respuesta del LLM que no cumpla la forma (texto suficiente, al menos una
fuente, confianza acotada) es bloqueada antes de llegar al usuario.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.schemas import DocAnswer
from demos import fake_rag


def test_doc_answer_happy_path() -> None:
    """Una respuesta válida pasa el contrato sin levantar excepción."""
    doc = DocAnswer(
        answer="Tienes 20 días hábiles de vacaciones al año.",
        sources=["hr_vacaciones"],
        confidence=0.95,
    )
    assert doc.answer.startswith("Tienes")
    assert doc.sources == ["hr_vacaciones"]
    assert 0.0 <= doc.confidence <= 1.0


def test_doc_answer_missing_sources_raises() -> None:
    """Una respuesta sin fuentes citadas debe ser rechazada."""
    with pytest.raises(ValidationError) as exc_info:
        DocAnswer(
            answer="Tienes 20 días de vacaciones.",
            sources=[],
            confidence=0.9,
        )
    assert "sources" in str(exc_info.value)


def test_doc_answer_confidence_out_of_range_raises() -> None:
    """Una confianza fuera de [0, 1] es violación de contrato."""
    with pytest.raises(ValidationError) as exc_info:
        DocAnswer(
            answer="Tienes 20 días de vacaciones.",
            sources=["hr_vacaciones"],
            confidence=1.5,
        )
    assert "confidence" in str(exc_info.value)


def test_doc_answer_answer_too_short_raises() -> None:
    """Una respuesta de menos de 5 caracteres es rechazada."""
    with pytest.raises(ValidationError):
        DocAnswer(
            answer="ok",
            sources=["hr_vacaciones"],
            confidence=0.5,
        )


def test_fake_rag_returns_valid_contract() -> None:
    """El sistema bajo prueba debe devolver siempre una respuesta válida.

    Este es el test estrella del Nivel 1: invocamos al LLM real y validamos
    que su salida pase el contrato sin coerción manual.
    """
    result = fake_rag.answer("¿Cuántos días de vacaciones tengo al año?")
    validated = DocAnswer.model_validate(result)
    assert validated.answer
    assert len(validated.sources) >= 1
    assert 0.0 <= validated.confidence <= 1.0
