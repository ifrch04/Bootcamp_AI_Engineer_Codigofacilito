"""Registro de herramientas para agentes de IA.

Provee un sistema de definición, validación y ejecución de herramientas
compatible con los formatos de OpenAI y Anthropic. Incluye herramientas
de ejemplo para búsqueda, fecha/hora y cálculo seguro.
"""

from __future__ import annotations

import ast
import datetime
import logging
import operator
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("orchestration.tools")

# Operadores permitidos para la calculadora segura
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class ToolDefinition:
    """Definición de una herramienta para agentes.

    Args:
        name: Nombre único de la herramienta.
        description: Descripción de lo que hace la herramienta.
        parameters: Esquema JSON Schema de los parámetros.
        function: Función que implementa la herramienta.
        requires_confirmation: Si requiere confirmación del usuario.
        timeout_seconds: Tiempo máximo de ejecución.
    """

    name: str
    description: str
    parameters: dict
    function: Callable
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0

    def validate_params(self, params: dict) -> tuple[bool, str]:
        """Valida parámetros contra el JSON Schema (validación manual).

        Args:
            params: Diccionario de parámetros a validar.

        Returns:
            Tupla ``(válido, mensaje_error)``. Si es válido, el mensaje
            es una cadena vacía.
        """
        schema = self.parameters

        # Verificar campos requeridos
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in params:
                return False, f"Missing required parameter: '{field_name}'"

        # Verificar tipos y enums
        properties = schema.get("properties", {})
        for param_name, value in params.items():
            if param_name not in properties:
                continue

            prop_schema = properties[param_name]

            # Verificar tipo
            expected_type_str = prop_schema.get("type")
            if expected_type_str and expected_type_str in _JSON_TYPE_MAP:
                expected_type = _JSON_TYPE_MAP[expected_type_str]
                # bool es subclase de int en Python, tratar como caso especial
                if expected_type_str == "integer" and isinstance(value, bool):
                    return (
                        False,
                        f"Parameter '{param_name}' expected type "
                        f"'{expected_type_str}', got 'boolean'",
                    )
                if not isinstance(value, expected_type):
                    actual = type(value).__name__
                    return (
                        False,
                        f"Parameter '{param_name}' expected type "
                        f"'{expected_type_str}', got '{actual}'",
                    )

            # Verificar enum
            enum_values = prop_schema.get("enum")
            if enum_values is not None and value not in enum_values:
                return (
                    False,
                    f"Parameter '{param_name}' must be one of {enum_values}, "
                    f"got '{value}'",
                )

        return True, ""

    def execute(self, params: dict) -> str:
        """Ejecuta la herramienta con los parámetros dados.

        Args:
            params: Diccionario de parámetros.

        Returns:
            Resultado como cadena, o mensaje de error.
        """
        valid, error_msg = self.validate_params(params)
        if not valid:
            return f"Validation error: {error_msg}"

        try:
            result = self.function(**params)
            return str(result)
        except Exception as exc:
            return f"Execution error: {type(exc).__name__}: {exc}"


class ToolRegistry:
    """Registro centralizado de herramientas disponibles para agentes.

    Permite registrar, consultar y ejecutar herramientas, y exportar
    sus definiciones en formatos compatibles con OpenAI y Anthropic.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Registra una herramienta.

        Raises:
            ValueError: Si ya existe una herramienta con el mismo nombre.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def register_function(
        self,
        name: str,
        description: str,
        parameters: dict,
        function: Callable,
        **kwargs: Any,
    ) -> None:
        """Atajo para registrar una función como herramienta.

        Args:
            name: Nombre de la herramienta.
            description: Descripción.
            parameters: JSON Schema de parámetros.
            function: Función implementadora.
            **kwargs: Parámetros extra para ``ToolDefinition``.
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            **kwargs,
        )
        self.register(tool)

    def get(self, name: str) -> ToolDefinition:
        """Obtiene una herramienta por nombre.

        Raises:
            KeyError: Si la herramienta no existe.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not found. "
                f"Available: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def remove(self, name: str) -> None:
        """Elimina una herramienta del registro.

        Raises:
            KeyError: Si la herramienta no existe.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        del self._tools[name]
        logger.info("Removed tool: %s", name)

    def list_tools(self) -> list[str]:
        """Retorna la lista de nombres de herramientas registradas."""
        return list(self._tools.keys())

    def to_openai_format(self) -> list[dict]:
        """Exporta las herramientas en formato OpenAI function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def to_anthropic_format(self) -> list[dict]:
        """Exporta las herramientas en formato Anthropic tool use."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def execute_tool(self, name: str, params: dict) -> str:
        """Ejecuta una herramienta por nombre con los parámetros dados.

        Args:
            name: Nombre de la herramienta.
            params: Diccionario de parámetros.

        Returns:
            Resultado como cadena, o mensaje de error.
        """
        try:
            tool = self.get(name)
            return tool.execute(params)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Herramientas de ejemplo
# ---------------------------------------------------------------------------


def search_documents(query: str, top_k: int = 5) -> str:
    """Busca documentos relevantes (mock).

    Args:
        query: Consulta de búsqueda.
        top_k: Número de resultados a retornar.

    Returns:
        Texto con resultados placeholder.
    """
    results = [
        f"Document {i + 1}: Result for '{query}' (relevance: {0.9 - i * 0.1:.1f})"
        for i in range(top_k)
    ]
    return "\n".join(results)


def get_current_datetime() -> str:
    """Retorna la fecha y hora actual en formato ISO."""
    return datetime.datetime.now().isoformat()


def _safe_eval_node(node: ast.AST) -> int | float:
    """Evalúa un nodo AST aritmético de forma segura."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evalúa una expresión aritmética de forma segura usando AST.

    Solo permite constantes numéricas y operadores aritméticos básicos.
    Nunca usa ``eval()`` directamente.

    Args:
        expression: Expresión aritmética (e.g., ``"2 + 3 * 4"``).

    Returns:
        Resultado como cadena.

    Raises:
        ValueError: Si la expresión contiene elementos no permitidos.
    """
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval_node(tree.body)
    return str(result)


if __name__ == "__main__":
    # Demo de herramientas
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
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
    )

    registry.register(
        ToolDefinition(
            name="datetime",
            description="Get current date and time",
            parameters={"type": "object", "properties": {}},
            function=get_current_datetime,
        )
    )

    registry.register(
        ToolDefinition(
            name="calculate",
            description="Evaluate arithmetic expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression",
                    },
                },
                "required": ["expression"],
            },
            function=calculate,
        )
    )

    print("Tools:", registry.list_tools())
    print("\nOpenAI format:", registry.to_openai_format())
    print("\nSearch result:", registry.execute_tool("search", {"query": "AI agents"}))
    print("\nDatetime:", registry.execute_tool("datetime", {}))
    print("\nCalculate:", registry.execute_tool("calculate", {"expression": "2 + 3 * 4"}))
