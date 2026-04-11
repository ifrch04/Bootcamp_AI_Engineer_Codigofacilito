"""Nivel 2 de testing: golden tests sobre el corpus Acme Corp."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demos import fake_rag
from evals.golden_test import check_golden


_GOLDEN_PATH = Path(__file__).resolve().parent.parent / "evals" / "golden" / "rag_questions.json"
_GOLDENS = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _GOLDENS, ids=lambda c: c["id"])
def test_golden_rag_question(case: dict) -> None:
    """Compara la respuesta del fake RAG contra la respuesta de oro."""
    result = fake_rag.answer(case["query"]) #Respuesta LLM
    actual = result["answer"] # Respuesta esperada

    report = check_golden(
        actual=actual,
        golden=case["golden_answer"],
        keywords=case.get("required_keywords", []),
        min_sim=case.get("min_similarity", 0.55),
    )

    assert report["overall_pass"], (
        f"\n  query:    {case['query']}"
        f"\n  actual:   {actual}"
        f"\n  golden:   {case['golden_answer']}"
        f"\n  sim:      {report['similarity']} (min {report['min_sim']})"
        f"\n  missing:  {report['keywords_missing']}"
    )
