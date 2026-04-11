"""Datasets curados y generador sintético para evaluación offline."""

from __future__ import annotations

import json
import re
from typing import Any

from core.llm_client import LLMClient


CURATED_DATASET: list[dict[str, Any]] = [
    {
        "id": "c_001",
        "category": "factual",
        "query": "¿Cuántos días de vacaciones tengo al año?",
        "golden_answer": "Tienes derecho a 20 días hábiles de vacaciones al año.",
        "required_keywords": ["20", "días"],
        "min_similarity": 0.55,
    },
    {
        "id": "c_002",
        "category": "factual",
        "query": "¿De qué hora a qué hora trabajo?",
        "golden_answer": "El horario estándar es de 9:00 a 18:00 de lunes a viernes.",
        "required_keywords": ["9", "18"],
        "min_similarity": 0.55,
    },
    {
        "id": "c_003",
        "category": "factual",
        "query": "¿Cuánto dinero recibo en vales de comida por día?",
        "golden_answer": "Acme Corp ofrece vales de comida por valor de 11 EUR por día trabajado.",
        "required_keywords": ["11"],
        "min_similarity": 0.55,
    },
    {
        "id": "c_004",
        "category": "factual",
        "query": "¿Cuántos días puedo trabajar desde casa?",
        "golden_answer": "Puedes trabajar desde casa hasta 3 días por semana.",
        "required_keywords": ["3"],
        "min_similarity": 0.55,
    },
    {
        "id": "c_005",
        "category": "factual",
        "query": "¿Cuánto presupuesto anual tengo para formación?",
        "golden_answer": "Tienes 1500 EUR anuales para formación continua.",
        "required_keywords": ["1500"],
        "min_similarity": 0.55,
    },
    {
        "id": "c_006",
        "category": "inferential",
        "query": "Si quiero salir 30 minutos antes, ¿es posible?",
        "golden_answer": "Sí, hay flexibilidad de entrada y salida de 30 minutos siempre que cumplas las 8 horas diarias.",
        "required_keywords": ["30"],
        "min_similarity": 0.50,
    },
    {
        "id": "c_007",
        "category": "inferential",
        "query": "Si tengo una reunión con un cliente externo, ¿qué debo ponerme?",
        "golden_answer": "Para reuniones con clientes externos se exige traje formal.",
        "required_keywords": ["formal"],
        "min_similarity": 0.50,
    },
    {
        "id": "c_008",
        "category": "inferential",
        "query": "¿Puedo trabajar desde casa los martes?",
        "golden_answer": "No, los martes son obligatoriamente presenciales.",
        "required_keywords": ["martes"],
        "min_similarity": 0.50,
    },
    {
        "id": "c_009",
        "category": "out_of_scope",
        "query": "¿Cuál es la capital de Australia?",
        "golden_answer": "No tengo información sobre eso en mis fuentes.",
        "required_keywords": ["no", "información"],
        "min_similarity": 0.45,
    },
    {
        "id": "c_010",
        "category": "out_of_scope",
        "query": "¿Cuánto cuesta un Tesla Model 3?",
        "golden_answer": "No tengo información sobre eso en mis fuentes.",
        "required_keywords": ["no", "información"],
        "min_similarity": 0.45,
    },
]


_SYNTH_PROMPT = (
    "Eres un generador de pares pregunta/respuesta para evaluar sistemas RAG. "
    "A partir del CHUNK proporcionado, genera exactamente {n} pares distintos. "
    "Cada par debe ser una pregunta natural en español y una respuesta breve "
    "extraída literalmente del chunk.\n\n"
    "Devuelve SIEMPRE un JSON válido con esta forma exacta:\n"
    '{{"pairs": [{{"query": "...", "answer": "..."}}, ...]}}\n\n'
    "No incluyas texto fuera del JSON. No uses bloques markdown."
)


def generate_synthetic(
    doc_chunk: str, n: int = 3, client: LLMClient | None = None
) -> list[dict[str, Any]]:
    """Genera n pares query/answer sintéticos a partir de un chunk de texto.

    Cada caso generado lleva `verified=False` por defecto. Se espera que el
    instructor o un revisor humano valide y promueva los casos al dataset
    curado.
    """
    client = client or LLMClient(provider="groq", temperature=0.0)
    response = client.chat(
        messages=[
            {"role": "system", "content": _SYNTH_PROMPT.format(n=n)},
            {"role": "user", "content": f"CHUNK:\n{doc_chunk}"},
        ],
        temperature=0.0,
    )

    raw = response["response"].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"El generador devolvió JSON inválido: {raw[:200]}")
        data = json.loads(match.group(0))

    pairs = data.get("pairs", [])
    return [
        {
            "id": f"synth_{idx:03d}",
            "category": "synthetic",
            "query": pair["query"],
            "golden_answer": pair["answer"],
            "required_keywords": [],
            "min_similarity": 0.50,
            "verified": False,
        }
        for idx, pair in enumerate(pairs, start=1)
    ]
