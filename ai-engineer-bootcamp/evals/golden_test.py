"""Núcleo del Nivel 2: comparación contra respuestas de oro (golden answers).

Combina similitud semántica vía embeddings + verificación de palabras clave
obligatorias. Mantiene un singleton del modelo `all-MiniLM-L6-v2` para que la
suite de tests no pague el costo de carga en cada caso.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer, util


_MODEL: SentenceTransformer | None = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """Devuelve un singleton lazy del modelo de embeddings."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Similitud coseno entre dos textos usando el modelo singleton."""
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(score)


def check_golden(
    actual: str,
    golden: str,
    keywords: list[str] | None = None,
    min_sim: float = 0.55,
) -> dict[str, Any]:
    """Compara una respuesta real contra una respuesta de oro.

    Args:
        actual: Respuesta producida por el sistema bajo prueba.
        golden: Respuesta de referencia curada.
        keywords: Palabras (case-insensitive) que la respuesta debe contener.
        min_sim: Umbral mínimo de similitud coseno para considerar pase.

    Returns:
        Dict con `similarity`, `sim_pass`, `keywords_found`, `keywords_missing`,
        `kw_pass` y `overall_pass`.
    """
    similarity = cosine_similarity(actual, golden)
    sim_pass = similarity >= min_sim

    keywords = keywords or []
    actual_lower = actual.lower()
    keywords_found = [kw for kw in keywords if kw.lower() in actual_lower]
    keywords_missing = [kw for kw in keywords if kw.lower() not in actual_lower]
    kw_pass = len(keywords_missing) == 0

    return {
        "similarity": round(similarity, 4),
        "sim_pass": sim_pass,
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
        "kw_pass": kw_pass,
        "overall_pass": sim_pass and kw_pass,
        "min_sim": min_sim,
    }
