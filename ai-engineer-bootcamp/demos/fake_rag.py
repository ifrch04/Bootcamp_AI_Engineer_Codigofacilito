"""Mini-RAG ficticio sobre políticas internas de "Acme Corp".

Sistema autocontenido bajo prueba para la Clase 14. El corpus vive en memoria
(sin Chroma, sin ingestión) y la función `answer()` ataca a Groq con todo el
corpus como contexto, devolviendo una respuesta validada por el contrato
`DocAnswer`. Existe también `answer_v2()` para demostrar regresiones entre
versiones de prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from contracts.schemas import DocAnswer
from core.llm_client import LLMClient


CORPUS: list[dict[str, str]] = [
    {
        "id": "hr_vacaciones",
        "title": "Política de vacaciones",
        "content": (
            "Los empleados de Acme Corp tienen derecho a 20 días hábiles de "
            "vacaciones al año. Los días no usados pueden acumularse hasta un "
            "máximo de 5 al año siguiente. Las solicitudes deben enviarse al "
            "menos 15 días antes de la fecha de inicio."
        ),
    },
    {
        "id": "hr_horario",
        "title": "Horario laboral",
        "content": (
            "El horario estándar en Acme Corp es de 9:00 a 18:00 de lunes a "
            "viernes, con una hora libre para almorzar. Existe flexibilidad de "
            "entrada de 30 minutos antes o después, siempre que se cumplan las "
            "8 horas diarias."
        ),
    },
    {
        "id": "hr_beneficios",
        "title": "Beneficios para empleados",
        "content": (
            "Acme Corp ofrece seguro médico privado, vales de comida por valor "
            "de 11 EUR por día trabajado y un plan de pensiones con aportación "
            "del 3% del salario bruto. También hay descuentos en gimnasios "
            "asociados."
        ),
    },
    {
        "id": "hr_vestimenta",
        "title": "Código de vestimenta",
        "content": (
            "El código de vestimenta en Acme Corp es business casual de lunes a "
            "jueves. Los viernes son de vestimenta libre. Para reuniones con "
            "clientes externos se exige traje formal."
        ),
    },
    {
        "id": "hr_home_office",
        "title": "Política de home office",
        "content": (
            "Los empleados de Acme Corp pueden trabajar desde casa hasta 3 días "
            "por semana, previa coordinación con su responsable directo. Los "
            "martes y jueves son obligatoriamente presenciales para fomentar la "
            "colaboración entre equipos."
        ),
    },
    {
        "id": "hr_capacitacion",
        "title": "Plan de capacitación",
        "content": (
            "Acme Corp asigna un presupuesto anual de 1500 EUR por empleado para "
            "formación continua, incluyendo cursos online, certificaciones y "
            "conferencias del sector. Las solicitudes se aprueban por el manager "
            "directo."
        ),
    },
]


_CLIENT: LLMClient | None = None


def _get_client() -> LLMClient:
    """Devuelve un singleton lazy de LLMClient forzado a Groq con temperature=0."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient(provider="groq", temperature=0.0)
    return _CLIENT


def _build_context() -> str:
    """Concatena el corpus en un único bloque etiquetado por id."""
    chunks = []
    for doc in CORPUS:
        chunks.append(
            f"[{doc['id']}] {doc['title']}\n{doc['content']}"
        )
    return "\n\n".join(chunks)


_JSON_INSTRUCTIONS = (
    "Devuelve SIEMPRE un objeto JSON válido con exactamente estas claves:\n"
    '  - "answer": string con la respuesta (mínimo 5 caracteres)\n'
    '  - "sources": SIEMPRE una lista JSON (nunca un string), conteniendo los '
    "ids de los documentos usados como strings, SIN corchetes alrededor del id.\n"
    '  - "confidence": número entre 0.0 y 1.0\n\n'
    "Ejemplo de salida correcta:\n"
    '{"answer": "Tienes 20 días de vacaciones al año.", '
    '"sources": ["hr_vacaciones"], "confidence": 0.95}\n\n'
    "No incluyas texto fuera del JSON. No uses bloques de código markdown."
)

_OUT_OF_SCOPE_RULE = (
    "Si la pregunta NO se puede responder con el CONTEXTO, devuelve "
    'answer="No tengo información sobre eso en mis fuentes.", '
    'sources=["none"] y confidence baja (<= 0.2). NUNCA dejes sources vacío.\n\n'
)

_SYSTEM_V1 = (
    "Eres un asistente de RRHH de Acme Corp. Responde preguntas de empleados "
    "usando ÚNICAMENTE la información del CONTEXTO proporcionado.\n\n"
    + _OUT_OF_SCOPE_RULE
    + _JSON_INSTRUCTIONS
)

_SYSTEM_V2 = (
    "Eres un asistente de RRHH de Acme Corp. Responde de forma MUY breve "
    "(máximo una oración) usando ÚNICAMENTE el CONTEXTO. Sé extremadamente "
    "conciso y omite detalles secundarios.\n\n"
    + _OUT_OF_SCOPE_RULE
    + _JSON_INSTRUCTIONS
)


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Intenta parsear el JSON, con un único fallback de extracción por regex."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"El LLM no devolvió JSON parseable: {raw[:200]}")
        return json.loads(match.group(0))


def _ask(query: str, system_prompt: str) -> dict[str, Any]:
    """Llama al LLM con el contexto y devuelve un dict validado por DocAnswer."""
    context = _build_context()
    user_content = f"CONTEXTO:\n{context}\n\nPREGUNTA: {query}"

    client = _get_client()
    response = client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    parsed = _parse_json_response(response["response"])
    validated = DocAnswer.model_validate(parsed)
    return validated.model_dump()


def answer(query: str) -> dict[str, Any]:
    """Versión 1 del sistema RAG: respuesta detallada y conservadora."""
    return _ask(query, _SYSTEM_V1)


def answer_v2(query: str) -> dict[str, Any]:
    """Versión 2 del sistema RAG: respuesta ultra-concisa (regresión)."""
    return _ask(query, _SYSTEM_V2)


if __name__ == "__main__":
    import pprint

    print("=" * 70)
    print("  Smoke test de fake_rag.answer()")
    print("=" * 70)
    result = answer("¿Cuántos días de vacaciones tengo al año?")
    pprint.pp(result)
