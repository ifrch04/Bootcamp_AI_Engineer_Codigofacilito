"""Tests para agents.docops_agent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.docops_agent import AgentResult, DocOpsAgent
from orchestration.tools import ToolDefinition, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_response(content: str, prompt_tokens: int = 20, completion_tokens: int = 30) -> MagicMock:
    """Crea un mock de respuesta de OpenAI."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    return response


def _make_test_registry() -> ToolRegistry:
    """Crea un registro con herramientas mock para tests."""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="search_documents",
        description="Busca documentos.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        function=lambda query, top_k=3: [
            {"content": f"Info sobre {query}", "source": "test.txt", "score": 0.95},
            {"content": f"Más datos de {query}", "source": "doc.txt", "score": 0.85},
        ],
    ))
    registry.register(ToolDefinition(
        name="get_current_datetime",
        description="Fecha actual.",
        parameters={"type": "object", "properties": {}},
        function=lambda: "2026-03-13T10:00:00",
    ))
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Tests para AgentResult."""

    def test_success_when_pipeline_succeeds(self):
        pr = MagicMock()
        pr.success = True
        pr.total_duration = 1.0
        pr.steps = []
        result = AgentResult(query="test", pipeline_result=pr)
        assert result.success is True

    def test_failure_when_no_pipeline(self):
        result = AgentResult(query="test")
        assert result.success is False

    def test_summary_contains_query_and_answer(self):
        pr = MagicMock()
        pr.success = True
        pr.total_duration = 0.5
        pr.steps = []
        result = AgentResult(query="¿horario?", answer="9 a 6", pipeline_result=pr)
        summary = result.summary()
        assert "horario" in summary
        assert "9 a 6" in summary


class TestDocOpsAgent:
    """Tests para DocOpsAgent."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def _make_agent(self, **kwargs) -> DocOpsAgent:
        with patch("agents.docops_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            agent = DocOpsAgent(registry=_make_test_registry(), **kwargs)
            agent._mock_client = mock_client  # guardar referencia para tests
            return agent

    def test_pipeline_runs_three_steps(self):
        """El pipeline ejecuta retrieve, process y generate."""
        agent = self._make_agent()
        agent._mock_client.chat.completions.create.return_value = (
            _mock_openai_response("La respuesta es 42.")
        )

        result = agent.run("¿Cuál es el horario?")

        assert result.success
        assert result.pipeline_result is not None
        assert len(result.pipeline_result.steps) == 3

    def test_answer_comes_from_llm(self):
        """La respuesta final viene de la generación del LLM."""
        agent = self._make_agent()
        agent._mock_client.chat.completions.create.return_value = (
            _mock_openai_response("El horario es de 9 a 18.")
        )

        result = agent.run("¿Cuál es el horario?")

        assert result.answer == "El horario es de 9 a 18."

    def test_context_includes_retrieved_docs(self):
        """El contexto pasado al LLM incluye los documentos recuperados."""
        agent = self._make_agent()
        agent._mock_client.chat.completions.create.return_value = (
            _mock_openai_response("Respuesta.")
        )

        result = agent.run("vacaciones")

        assert "Info sobre vacaciones" in result.context

    def test_context_includes_timestamp(self):
        """El contexto incluye el timestamp de la consulta."""
        agent = self._make_agent()
        agent._mock_client.chat.completions.create.return_value = (
            _mock_openai_response("Respuesta.")
        )

        result = agent.run("test")

        assert "2026-03-13" in result.context

    def test_token_usage_tracked(self):
        """Los tokens de uso del LLM se registran en el resultado."""
        agent = self._make_agent()
        agent._mock_client.chat.completions.create.return_value = (
            _mock_openai_response("Ok.", prompt_tokens=100, completion_tokens=50)
        )

        result = agent.run("test")

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    def test_llm_failure_stops_pipeline(self):
        """Si el LLM falla, el pipeline reporta error."""
        agent = self._make_agent()
        agent._mock_client.chat.completions.create.side_effect = (
            RuntimeError("API down")
        )

        result = agent.run("test")

        assert not result.success
        assert "Error" in result.answer

    def test_custom_system_prompt(self):
        """Se puede personalizar el system prompt."""
        agent = self._make_agent(system_prompt="Eres un bot de prueba.")
        assert agent.system_prompt == "Eres un bot de prueba."

    @patch.dict("os.environ", {"GROQ_API_KEY": ""})
    def test_missing_api_key_raises(self):
        """Sin GROQ_API_KEY lanza ValueError."""
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            with patch("agents.docops_agent.OpenAI"):
                DocOpsAgent(registry=_make_test_registry())


class TestRetrievedChunksTracking:
    """Tests para el tracking de chunks recuperados."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_chunks_are_tracked(self):
        """Los chunks recuperados se guardan en el resultado."""
        with patch("agents.docops_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = (
                _mock_openai_response("Ok.")
            )

            agent = DocOpsAgent(registry=_make_test_registry())
            result = agent.run("vacaciones")

            assert len(result.retrieved_chunks) == 2
            assert result.retrieved_chunks[0]["score"] == 0.95
            assert result.retrieved_chunks[0]["source"] == "test.txt"
