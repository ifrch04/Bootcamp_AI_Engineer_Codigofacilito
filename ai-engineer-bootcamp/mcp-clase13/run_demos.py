#!/usr/bin/env python3
"""
Script de demostración — Clase 13 MCP

Ejecuta las demos de las 3 prácticas de forma automatizada,
conectándose a los servidores MCP y ejecutando las tools programáticamente.
Usa colores ANSI para distinguir cada sección.

Uso:
    python run_demos.py           # Ejecuta prácticas 1, 2 y 3
    python run_demos.py 1         # Solo práctica 1
    python run_demos.py 2         # Solo práctica 2
    python run_demos.py 3         # Solo práctica 3 (requiere GROQ_API_KEY en .env)
"""

import asyncio
import json
import os
import sys

# ─── Colores ANSI ────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Colores para cada práctica
CYAN = "\033[36m"       # Práctica 1
GREEN = "\033[32m"      # Práctica 2
MAGENTA = "\033[35m"    # Práctica 3

YELLOW = "\033[33m"     # Warnings / nombres de tools
RED = "\033[31m"        # Errores
BLUE = "\033[34m"       # Resultados
WHITE = "\033[97m"      # Texto normal destacado


def header(text: str, color: str) -> None:
    """Imprime un header de sección con color."""
    width = 70
    print(f"\n{color}{BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{RESET}\n")


def subheader(text: str, color: str) -> None:
    """Imprime un subheader."""
    print(f"  {color}{BOLD}▸ {text}{RESET}")


def tool_call(name: str, args: dict | None = None) -> None:
    """Imprime qué tool se va a invocar."""
    args_str = json.dumps(args, ensure_ascii=False) if args else ""
    print(f"    {YELLOW}⚡ {name}({args_str}){RESET}")


def result(text: str, indent: int = 6) -> None:
    """Imprime el resultado de una tool."""
    prefix = " " * indent
    for line in str(text).splitlines():
        print(f"{prefix}{BLUE}{line}{RESET}")


def success(text: str) -> None:
    """Imprime un mensaje de éxito."""
    print(f"    {GREEN}✓ {text}{RESET}")


def error(text: str) -> None:
    """Imprime un mensaje de error."""
    print(f"    {RED}✗ {text}{RESET}")


def separator() -> None:
    """Línea divisoria ligera."""
    print(f"  {DIM}{'─' * 60}{RESET}")


# ─── Helpers MCP ─────────────────────────────────────────────────────────────

async def call_and_show(session, name: str, args: dict | None = None) -> str:
    """Invoca una tool MCP e imprime el resultado. Devuelve el texto."""
    tool_call(name, args)
    res = await session.call_tool(name, arguments=args or {})
    text = "\n".join(
        block.text for block in res.content if hasattr(block, "text")
    )
    result(text)
    return text


# ─── Práctica 1: HolaMCP ─────────────────────────────────────────────────────

async def demo_practica_1():
    """Demo del servidor HolaMCP (server.py)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    header("PRÁCTICA 1 — Servidor HolaMCP (server.py)", CYAN)

    server_params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Listar tools
            tools = await session.list_tools()
            subheader(f"Tools disponibles: {len(tools.tools)}", CYAN)
            for t in tools.tools:
                print(f"    {WHITE}• {t.name}{DIM} — {t.description.splitlines()[0]}{RESET}")
            separator()

            # Test: hora en CDMX
            subheader("Test: get_current_time con timezone válida", CYAN)
            await call_and_show(session, "get_current_time", {"timezone": "America/Mexico_City"})
            success("Hora obtenida correctamente")
            separator()

            # Test: hora con timezone inválida
            subheader("Test: get_current_time con timezone inválida", CYAN)
            text = await call_and_show(session, "get_current_time", {"timezone": "Mars/Olympus"})
            if "error" in text.lower() or "no válida" in text.lower():
                success("Error manejado correctamente (no crasheó)")
            else:
                error("Debería haber devuelto un error")
            separator()

            # Test: cálculo válido
            subheader("Test: calculate con expresión válida", CYAN)
            text = await call_and_show(session, "calculate", {"expression": "2 * (3 + 4)"})
            if "14" in text:
                success("Resultado correcto: 14")
            else:
                error(f"Esperado '14', obtenido '{text}'")
            separator()

            # Test: potencia
            subheader("Test: calculate con potencia", CYAN)
            text = await call_and_show(session, "calculate", {"expression": "2 ** 10"})
            if "1024" in text:
                success("Resultado correcto: 1024")
            else:
                error(f"Esperado '1024', obtenido '{text}'")
            separator()

            # Test: expresión maliciosa
            subheader("Test: calculate rechaza código malicioso", CYAN)
            text = await call_and_show(session, "calculate", {"expression": "__import__('os').system('ls')"})
            if "no válida" in text.lower() or "no permitido" in text.lower() or "error" in text.lower():
                success("Expresión maliciosa rechazada correctamente")
            else:
                error("PELIGRO: la expresión no fue rechazada")
            separator()

            # Test: import
            subheader("Test: calculate rechaza import", CYAN)
            text = await call_and_show(session, "calculate", {"expression": "import os"})
            if "error" in text.lower() or "no válida" in text.lower() or "sintaxis" in text.lower():
                success("Import rechazado correctamente")
            else:
                error("PELIGRO: el import no fue rechazado")

    print(f"\n  {CYAN}{BOLD}Práctica 1 completada.{RESET}\n")


# ─── Práctica 2: DocsServer ──────────────────────────────────────────────────

async def demo_practica_2():
    """Demo del servidor DocsServer (docs_server.py)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    header("PRÁCTICA 2 — Servidor DocsServer (docs_server.py)", GREEN)

    server_params = StdioServerParameters(command="python", args=["docs_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Listar tools
            tools = await session.list_tools()
            subheader(f"Tools disponibles: {len(tools.tools)}", GREEN)
            for t in tools.tools:
                print(f"    {WHITE}• {t.name}{DIM} — {t.description.splitlines()[0]}{RESET}")
            separator()

            # Test: list_docs
            subheader("Test: list_docs()", GREEN)
            text = await call_and_show(session, "list_docs")
            if "api.md" in text and "changelog.md" in text and "troubleshooting.md" in text:
                success("Los 3 archivos listados correctamente")
            else:
                error("Faltan archivos en la lista")
            separator()

            # Test: read_doc válido
            subheader("Test: read_doc('api.md')", GREEN)
            text = await call_and_show(session, "read_doc", {"filename": "api.md"})
            if "API Reference" in text:
                success("Contenido leído correctamente")
            else:
                error("Contenido inesperado")
            separator()

            # Test: read_doc con path traversal
            subheader("Test: read_doc con path traversal", GREEN)
            text = await call_and_show(session, "read_doc", {"filename": "../../../etc/passwd"})
            if "inválido" in text.lower() or "traversal" in text.lower():
                success("Path traversal rechazado correctamente")
            else:
                error("PELIGRO: path traversal no fue rechazado")
            separator()

            # Test: read_doc archivo inexistente
            subheader("Test: read_doc con archivo inexistente", GREEN)
            text = await call_and_show(session, "read_doc", {"filename": "noexiste.md"})
            if "no encontrado" in text.lower() and "disponibles" in text.lower():
                success("Error descriptivo con archivos disponibles")
            else:
                error("El mensaje de error no incluye archivos disponibles")
            separator()

            # Test: search_docs
            subheader("Test: search_docs('rate limit')", GREEN)
            text = await call_and_show(session, "search_docs", {"query": "rate limit"})
            if "troubleshooting" in text.lower() and "changelog" in text.lower():
                success("Matches encontrados en ambos archivos")
            elif "troubleshooting" in text.lower() or "changelog" in text.lower():
                success("Match encontrado en al menos un archivo")
            else:
                error("No se encontraron matches esperados")
            separator()

            # Test: search_docs case-insensitive
            subheader("Test: search_docs('ENDPOINT') — case-insensitive", GREEN)
            text = await call_and_show(session, "search_docs", {"query": "ENDPOINT"})
            if "api.md" in text.lower():
                success("Búsqueda case-insensitive funciona")
            else:
                error("La búsqueda no encontró matches en api.md")
            separator()

            # Test: get_doc_sections
            subheader("Test: get_doc_sections('api.md')", GREEN)
            text = await call_and_show(session, "get_doc_sections", {"filename": "api.md"})
            if "Authentication" in text and "Endpoints" in text:
                success("Secciones extraídas correctamente")
            else:
                error("Faltan secciones esperadas")
            separator()

            # ── Resources ──
            subheader("Resources disponibles", GREEN)
            resources = await session.list_resources()
            templates = await session.list_resource_templates()
            for r in resources.resources:
                print(f"    {WHITE}• {r.uri}{DIM} — {r.description or 'sin descripción'}{RESET}")
            for t in templates.resourceTemplates:
                print(f"    {WHITE}• {t.uriTemplate}{DIM} (template) — {t.description or 'sin descripción'}{RESET}")
            total_r = len(resources.resources) + len(templates.resourceTemplates)
            if total_r >= 2:
                success(f"{total_r} resource(s) registrados")
            else:
                error("Se esperaban al menos 2 resources")
            separator()

            # Test: leer resource estático docs://index
            subheader("Test: leer resource docs://index", GREEN)
            try:
                res = await session.read_resource("docs://index")
                index_text = "\n".join(
                    c.text for c in res.contents if hasattr(c, "text")
                )
                # Mostrar solo las primeras líneas
                for line in index_text.splitlines()[:8]:
                    result(line)
                if "api.md" in index_text and "troubleshooting" in index_text.lower():
                    success("Índice contiene los 3 documentos")
                else:
                    error("El índice no contiene todos los documentos esperados")
            except Exception as e:
                error(f"Error leyendo resource: {e}")
            separator()

            # Test: leer resource template docs://files/api.md
            subheader("Test: leer resource docs://files/api.md", GREEN)
            try:
                res = await session.read_resource("docs://files/api.md")
                file_text = "\n".join(
                    c.text for c in res.contents if hasattr(c, "text")
                )
                if "API Reference" in file_text:
                    success("Resource template devuelve contenido correcto")
                else:
                    error("Contenido inesperado del resource template")
            except Exception as e:
                error(f"Error leyendo resource template: {e}")
            separator()

            # ── Prompts ──
            subheader("Prompts disponibles", GREEN)
            prompts = await session.list_prompts()
            for p in prompts.prompts:
                args_info = ""
                if p.arguments:
                    args_list = [a.name for a in p.arguments]
                    args_info = f" ({', '.join(args_list)})"
                print(f"    {WHITE}• {p.name}{args_info}{DIM} — {p.description or ''}{RESET}")
            if len(prompts.prompts) >= 3:
                success(f"{len(prompts.prompts)} prompt(s) registrados")
            else:
                error("Se esperaban al menos 3 prompts")
            separator()

            # Test: ejecutar prompt summarize_doc
            subheader("Test: prompt summarize_doc(filename='changelog.md')", GREEN)
            try:
                prompt_result = await session.get_prompt(
                    "summarize_doc",
                    arguments={"filename": "changelog.md"},
                )
                if prompt_result.messages:
                    msg = prompt_result.messages[0]
                    content_text = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
                    # Mostrar solo un snippet
                    snippet = content_text[:200]
                    result(f"{snippet}...")
                    if "changelog" in content_text.lower():
                        success("Prompt inyecta contenido del documento")
                    else:
                        error("Prompt no parece incluir el contenido del doc")
                else:
                    error("Prompt no devolvió mensajes")
            except Exception as e:
                error(f"Error ejecutando prompt: {e}")
            separator()

            # Test: ejecutar prompt troubleshoot
            subheader("Test: prompt troubleshoot(error_description='error 429')", GREEN)
            try:
                prompt_result = await session.get_prompt(
                    "troubleshoot",
                    arguments={"error_description": "Estoy recibiendo un error 429 constantemente"},
                )
                if prompt_result.messages:
                    msg = prompt_result.messages[0]
                    content_text = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
                    if "429" in content_text and "troubleshooting" in content_text.lower():
                        success("Prompt de troubleshoot inyecta guía + error del usuario")
                    else:
                        result(content_text[:150] + "...")
                        error("Prompt no combina la guía con la descripción del error")
                else:
                    error("Prompt no devolvió mensajes")
            except Exception as e:
                error(f"Error ejecutando prompt: {e}")

    print(f"\n  {GREEN}{BOLD}Práctica 2 completada.{RESET}\n")


# ─── Práctica 3: Cliente MCP con Groq ────────────────────────────────────────

async def demo_practica_3():
    """Demo del cliente MCP con Groq (client.py)."""
    from dotenv import load_dotenv
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    header("PRÁCTICA 3 — Cliente MCP con Groq (client.py)", MAGENTA)

    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        error("GROQ_API_KEY no encontrada en .env — saltando Práctica 3")
        print(f"    {DIM}Copia .env.example a .env y agrega tu API key de Groq{RESET}")
        return

    # Verificar sintaxis del archivo primero
    subheader("Verificando sintaxis de client.py...", MAGENTA)
    import ast as _ast
    try:
        with open("client.py") as f:
            _ast.parse(f.read())
        success("Sintaxis correcta")
    except SyntaxError as e:
        error(f"Error de sintaxis: {e}")
        return
    separator()

    # Importar el módulo del cliente
    # Usamos el run_agent directamente conectándonos al docs_server
    from openai import OpenAI

    groq = OpenAI(
        base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.environ["GROQ_API_KEY"],
    )
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Importar la función de conversión del cliente
    sys.path.insert(0, ".")
    from client import mcp_tool_to_openai, run_agent

    server_params = StdioServerParameters(command="python", args=["docs_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Listar tools convertidas
            tools_result = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]
            subheader(f"Tools convertidas a formato OpenAI: {len(openai_tools)}", MAGENTA)
            for t in openai_tools:
                print(f"    {WHITE}• {t['function']['name']}{RESET}")
            separator()

            # Prueba 1: pregunta simple (una tool)
            questions = [
                ("¿Qué archivos de documentación tengo disponibles?", "lista de archivos"),
                ("¿Qué endpoints están documentados en mi API?", "endpoints de la API"),
                ("Tengo un error 429, ¿qué hago?", "manejo del error 429"),
            ]

            for i, (question, desc) in enumerate(questions, 1):
                subheader(f"Prueba {i}: {desc}", MAGENTA)
                print(f"    {DIM}Pregunta: {question}{RESET}")
                print()

                try:
                    answer = await run_agent(session, question)
                    # Mostrar respuesta truncada si es muy larga
                    lines = answer.strip().splitlines()
                    if len(lines) > 15:
                        for line in lines[:12]:
                            result(line)
                        result(f"... ({len(lines) - 12} líneas más)")
                    else:
                        for line in lines:
                            result(line)
                    success("Respuesta obtenida correctamente")
                except Exception as e:
                    error(f"Error: {e}")

                if i < len(questions):
                    separator()

    print(f"\n  {MAGENTA}{BOLD}Práctica 3 completada.{RESET}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

async def run_all(practices: list[int]):
    """Ejecuta las prácticas seleccionadas."""
    print(f"\n{BOLD}{WHITE}{'═' * 70}")
    print(f"  CLASE 13 — Demos Model Context Protocol")
    print(f"{'═' * 70}{RESET}")

    demos = {
        1: demo_practica_1,
        2: demo_practica_2,
        3: demo_practica_3,
    }

    for p in practices:
        if p in demos:
            await demos[p]()

    print(f"\n{BOLD}{WHITE}{'═' * 70}")
    print(f"  Demos finalizadas")
    print(f"{'═' * 70}{RESET}\n")


def main():
    # Parsear argumentos
    args = sys.argv[1:]
    if args:
        try:
            practices = [int(a) for a in args]
            invalid = [p for p in practices if p not in (1, 2, 3)]
            if invalid:
                print(f"{RED}Prácticas inválidas: {invalid}. Usa 1, 2 o 3.{RESET}")
                sys.exit(1)
        except ValueError:
            print(f"{RED}Uso: python run_demos.py [1] [2] [3]{RESET}")
            sys.exit(1)
    else:
        practices = [1, 2, 3]

    # Cambiar al directorio del script para que los paths relativos funcionen
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    asyncio.run(run_all(practices))


if __name__ == "__main__":
    main()
