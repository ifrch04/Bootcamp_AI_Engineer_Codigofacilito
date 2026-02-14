"""Prompt engineering toolkit: templates, chains, registry, and evaluation."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, strip, and remove accents for comparison."""
    s = s.strip().lower()
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _extract_json(text: str) -> dict | None:
    """Try to parse JSON from LLM output, handling markdown wrappers."""
    text = text.strip()
    # 1. Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2. Markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # 3. Find first {...} (non-greedy, flat)
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------

class PromptTemplate:
    """A reusable prompt with variable substitution and metadata."""

    def __init__(
        self,
        name: str,
        template: str,
        metadata: dict | None = None,
    ) -> None:
        self.name = name
        self.template = template
        self.metadata = metadata or {}

    def render(self, **kwargs: Any) -> str:
        """Substitute variables in the template."""
        return self.template.format(**kwargs)

    def render_with_examples(
        self, examples: list[dict], **kwargs: Any
    ) -> str:
        """Format few-shot examples and inject them into the template.

        Each example dict should have ``input`` and ``output`` keys.
        The formatted block is passed as the ``{examples}`` variable.
        """
        lines: list[str] = []
        for ex in examples:
            lines.append(f"Input: {ex['input']}\nOutput: {ex['output']}")
        kwargs["examples"] = "\n\n".join(lines)
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        version = self.metadata.get("version", "?")
        return f"PromptTemplate(name={self.name!r}, version={version!r})"


# ---------------------------------------------------------------------------
# PromptRegistry
# ---------------------------------------------------------------------------

class PromptRegistry:
    """Centralized dictionary of PromptTemplates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found in registry.")
        return self._templates[name]

    def list_all(self) -> list[str]:
        return list(self._templates.keys())

    def get_version(self, name: str) -> str:
        return self.get(name).metadata.get("version", "unknown")


# ---------------------------------------------------------------------------
# PromptChain
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    """Result of running a PromptChain."""
    steps: list[dict]
    final_response: str
    total_tokens: int
    total_latency_s: float


class PromptChain:
    """Execute a sequence of PromptTemplates, piping outputs forward."""

    def __init__(self, templates: list[PromptTemplate]) -> None:
        self.templates = templates

    def run(self, llm_client: Any, initial_input: dict) -> ChainResult:
        steps: list[dict] = []
        current_vars = dict(initial_input)
        total_tokens = 0
        total_latency_s = 0.0

        for i, template in enumerate(self.templates):
            prompt = template.render(**current_vars)
            result = llm_client.chat(prompt)

            response_text = result["response"]
            metadata = result["metadata"]
            tokens = metadata["usage"]["total_tokens"]
            latency_s = metadata["latency_ms"] / 1000

            steps.append({
                "step": i + 1,
                "template": template.name,
                "prompt": prompt,
                "response": response_text,
                "tokens": tokens,
                "latency_s": latency_s,
            })

            total_tokens += tokens
            total_latency_s += latency_s

            # Make output available for subsequent steps
            current_vars[f"step_{i + 1}_output"] = response_text
            current_vars["extraction_result"] = response_text

        return ChainResult(
            steps=steps,
            final_response=steps[-1]["response"] if steps else "",
            total_tokens=total_tokens,
            total_latency_s=total_latency_s,
        )


# ---------------------------------------------------------------------------
# evaluate_prompt
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """Metrics returned by evaluate_prompt."""
    accuracy: float
    json_parse_rate: float
    campos_correctos_rate: float
    tokens_promedio: float
    latencia_promedio: float
    details: list[dict] = field(default_factory=list)


def evaluate_prompt(
    prompt_or_chain: PromptTemplate | PromptChain,
    llm_client: Any,
    golden_set: list[dict],
    *,
    input_key: str = "ticket",
    delay: float = 1.0,
    verbose: bool = False,
) -> EvalMetrics:
    """Run *prompt_or_chain* against every example in *golden_set* and score.

    Returns accuracy, JSON parse rate, per-field correctness, average tokens,
    and average latency.
    """
    total = len(golden_set)
    json_ok = 0
    correct = 0
    campos_correctos = 0
    campos_total = 0
    total_tokens = 0
    total_latency = 0.0
    details: list[dict] = []

    for idx, item in enumerate(golden_set):
        input_text = item["input"]
        expected = item["expected"]

        # -- call the LLM ------------------------------------------------
        try:
            if isinstance(prompt_or_chain, PromptChain):
                chain_result = prompt_or_chain.run(
                    llm_client, {input_key: input_text}
                )
                response_text = chain_result.final_response
                tokens = chain_result.total_tokens
                latency_s = chain_result.total_latency_s
            else:
                rendered = prompt_or_chain.render(**{input_key: input_text})
                result = llm_client.chat(rendered)
                response_text = result["response"]
                tokens = result["metadata"]["usage"]["total_tokens"]
                latency_s = result["metadata"]["latency_ms"] / 1000
        except Exception as exc:
            logger.warning("LLM error on item %d: %s", idx, exc)
            details.append({
                "input": input_text,
                "expected": expected,
                "response": str(exc),
                "parsed": None,
                "correct": False,
                "json_valid": False,
            })
            campos_total += 2
            if delay > 0 and idx < total - 1:
                time.sleep(delay)
            continue

        total_tokens += tokens
        total_latency += latency_s

        # -- parse & compare ---------------------------------------------
        parsed = _extract_json(response_text)
        json_valid = parsed is not None
        if json_valid:
            json_ok += 1

        cat_match = False
        pri_match = False

        if parsed:
            cat_match = (
                _normalize(str(parsed.get("categoria", "")))
                == _normalize(str(expected.get("categoria", "")))
            )
            pri_match = (
                _normalize(str(parsed.get("prioridad", "")))
                == _normalize(str(expected.get("prioridad", "")))
            )
            if cat_match:
                campos_correctos += 1
            if pri_match:
                campos_correctos += 1

        campos_total += 2

        if cat_match and pri_match:
            correct += 1

        detail = {
            "input": input_text,
            "expected": expected,
            "response": response_text,
            "parsed": parsed,
            "correct": cat_match and pri_match,
            "json_valid": json_valid,
            "cat_match": cat_match,
            "pri_match": pri_match,
            "tokens": tokens,
            "latency_s": latency_s,
        }
        details.append(detail)

        if verbose:
            status = "OK" if detail["correct"] else "FAIL"
            print(f"  [{idx + 1}/{total}] {status} | {input_text[:60]}")
            if parsed:
                print(f"    Esperado : {expected}")
                print(f"    Obtenido : {parsed}")
            else:
                print(f"    Respuesta (no JSON): {response_text[:120]}")

        if delay > 0 and idx < total - 1:
            time.sleep(delay)

    return EvalMetrics(
        accuracy=correct / total if total else 0,
        json_parse_rate=json_ok / total if total else 0,
        campos_correctos_rate=campos_correctos / campos_total if campos_total else 0,
        tokens_promedio=total_tokens / total if total else 0,
        latencia_promedio=total_latency / total if total else 0,
        details=details,
    )
