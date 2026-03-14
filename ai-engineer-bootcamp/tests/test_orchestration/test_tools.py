"""Tests para orchestration.tools."""

import pytest

from orchestration.tools import (
    ToolDefinition,
    ToolRegistry,
    calculate,
    get_current_datetime,
    search_documents,
)


def _make_search_tool() -> ToolDefinition:
    """Helper para crear una herramienta de búsqueda de prueba."""
    return ToolDefinition(
        name="search",
        description="Search documents",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Results count"},
            },
            "required": ["query"],
        },
        function=search_documents,
    )


def _make_calc_tool() -> ToolDefinition:
    """Helper para crear una herramienta de cálculo de prueba."""
    return ToolDefinition(
        name="calculate",
        description="Evaluate arithmetic",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Expression"},
            },
            "required": ["expression"],
        },
        function=calculate,
    )


class TestToolDefinition:
    """Tests para ToolDefinition."""

    def test_validate_params_valid(self):
        tool = _make_search_tool()
        valid, msg = tool.validate_params({"query": "test", "top_k": 3})
        assert valid is True
        assert msg == ""

    def test_validate_params_missing_required(self):
        tool = _make_search_tool()
        valid, msg = tool.validate_params({"top_k": 3})
        assert valid is False
        assert "query" in msg

    def test_validate_params_wrong_type(self):
        tool = _make_search_tool()
        valid, msg = tool.validate_params({"query": "test", "top_k": "not_int"})
        assert valid is False
        assert "top_k" in msg

    def test_execute_valid_params(self):
        tool = _make_search_tool()
        result = tool.execute({"query": "AI", "top_k": 2})
        assert "Document 1" in result
        assert "Document 2" in result

    def test_execute_invalid_params(self):
        tool = _make_search_tool()
        result = tool.execute({"top_k": 3})  # missing required 'query'
        assert "Validation error" in result

    def test_execute_captures_exception(self):
        def broken(**kwargs):
            raise RuntimeError("something broke")

        tool = ToolDefinition(
            name="broken",
            description="A broken tool",
            parameters={"type": "object", "properties": {}},
            function=broken,
        )
        result = tool.execute({})
        assert "Execution error" in result
        assert "something broke" in result


class TestToolRegistry:
    """Tests para ToolRegistry."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = _make_search_tool()
        registry.register(tool)
        assert registry.get("search") is tool

    def test_duplicate_raises_value_error(self):
        registry = ToolRegistry()
        tool = _make_search_tool()
        registry.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_missing_raises_key_error(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(_make_search_tool())
        registry.register(_make_calc_tool())
        names = registry.list_tools()
        assert "search" in names
        assert "calculate" in names

    def test_to_openai_format(self):
        registry = ToolRegistry()
        registry.register(_make_search_tool())
        fmt = registry.to_openai_format()

        assert len(fmt) == 1
        assert fmt[0]["type"] == "function"
        assert fmt[0]["function"]["name"] == "search"
        assert "parameters" in fmt[0]["function"]

    def test_to_anthropic_format(self):
        registry = ToolRegistry()
        registry.register(_make_search_tool())
        fmt = registry.to_anthropic_format()

        assert len(fmt) == 1
        assert fmt[0]["name"] == "search"
        assert "input_schema" in fmt[0]

    def test_execute_tool_end_to_end(self):
        registry = ToolRegistry()
        registry.register(_make_calc_tool())
        result = registry.execute_tool("calculate", {"expression": "2 + 3"})
        assert result == "5"

    def test_remove(self):
        registry = ToolRegistry()
        registry.register(_make_search_tool())
        registry.remove("search")
        assert "search" not in registry.list_tools()

    def test_remove_missing_raises_key_error(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.remove("ghost")
