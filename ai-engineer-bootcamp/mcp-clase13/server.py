"""
Práctica 1 — Servidor MCP "HolaMCP"

Servidor MCP simple con dos tools: obtener la hora actual y evaluar
expresiones matemáticas de forma segura. Demuestra los fundamentos
del protocolo MCP con FastMCP.
"""

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

# Crea el servidor MCP
mcp = FastMCP("HolaMCP")

# Operadores permitidos en el evaluador seguro de expresiones
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Evalúa recursivamente un AST que solo contiene aritmética básica.

    Lanza ValueError si encuentra cualquier nodo que no sea un número,
    operador aritmético o paréntesis (agrupación implícita en el AST).
    """
    # Números literales (int o float)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # Operaciones binarias: 2 + 3, 4 * 5, etc.
    if isinstance(node, ast.BinOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Operador binario no permitido: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_func(left, right)

    # Operaciones unarias: -5, +3
    if isinstance(node, ast.UnaryOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Operador unario no permitido: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))

    # Cualquier otro nodo (Name, Call, Attribute, Import, etc.) se rechaza
    raise ValueError(
        f"Elemento no permitido en la expresión: {type(node).__name__}"
    )


@mcp.tool()
def get_current_time(timezone: str) -> str:
    """Devuelve la hora actual en formato ISO 8601 para una zona horaria dada.

    Úsala cuando el usuario pregunte qué hora es, necesite saber la hora en
    otra ciudad o zona horaria, o quiera comparar horarios entre regiones.
    No la uses para convertir entre zonas horarias ni para operaciones con fechas.

    Args:
        timezone: Zona horaria en formato IANA, por ejemplo "America/Mexico_City",
                  "US/Eastern", "Europe/London". Usa "UTC" para tiempo universal.

    Returns:
        La hora actual en formato ISO 8601, por ejemplo "2026-04-08T14:23:45-06:00".
        Si la zona horaria es inválida, devuelve un mensaje de error descriptivo.

    Ejemplo:
        get_current_time("America/Mexico_City") → "2026-04-08T14:23:45-06:00"
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.isoformat()
    except (ZoneInfoNotFoundError, KeyError):
        return (
            f"Error: Zona horaria '{timezone}' no válida. "
            f"Usa el formato IANA, por ejemplo: 'America/Mexico_City', 'US/Eastern', 'UTC'."
        )
    except Exception as e:
        return f"Error inesperado al obtener la hora: {e}"


@mcp.tool()
def calculate(expression: str) -> str:
    """Evalúa una expresión matemática de forma segura y devuelve el resultado.

    Úsala cuando el usuario pida hacer un cálculo aritmético: sumas, restas,
    multiplicaciones, divisiones, potencias o módulo. Soporta paréntesis para
    agrupar operaciones.
    No la uses para operaciones con fechas, conversiones de unidades ni
    funciones matemáticas avanzadas (sin, cos, log, etc.).

    Args:
        expression: Expresión aritmética como string.
                    Operadores permitidos: +, -, *, /, %, ** y paréntesis.
                    Ejemplos válidos: "2 * (3 + 4)", "(100 - 25) / 5", "2 ** 10".

    Returns:
        El resultado del cálculo como string. Si la expresión contiene elementos
        no permitidos (como llamadas a funciones o imports), devuelve un error.

    Ejemplo:
        calculate("2 * (3 + 4)") → "14"
    """
    try:
        # Parsear la expresión como AST de Python
        tree = ast.parse(expression, mode="eval")

        # El nodo raíz en modo "eval" es ast.Expression
        result = _safe_eval(tree.body)

        # Si el resultado es un float entero (ej: 14.0), mostrarlo como int
        if isinstance(result, float) and result == int(result) and not (
            # Preservar float si la expresión tenía división explícita
            "/" in expression and "**" not in expression.replace("/", "")
        ):
            # Dejamos el resultado como float si hubo división para ser explícitos
            pass

        return str(result)

    except (ValueError, TypeError) as e:
        return f"Expresión no válida: {e}. Solo se permiten operaciones aritméticas básicas (+, -, *, /, %, **)."
    except SyntaxError:
        return f"Error de sintaxis en la expresión: '{expression}'. Verifica que sea una expresión matemática válida."
    except ZeroDivisionError:
        return "Error: División entre cero."
    except Exception as e:
        return f"Error al evaluar la expresión: {e}"


if __name__ == "__main__":
    mcp.run()
