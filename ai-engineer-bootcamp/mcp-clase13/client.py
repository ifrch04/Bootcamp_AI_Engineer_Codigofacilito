"""
Práctica 3 — Cliente MCP con Groq

Cliente MCP custom que se conecta al docs_server.py y usa Groq como LLM.
Implementa el loop agéntico completo: envía preguntas al modelo, ejecuta
las tools que el modelo solicita vía MCP, y devuelve la respuesta final.
También consume resources y prompts del servidor MCP.
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

# Cliente de Groq (API compatible con OpenAI)
groq = OpenAI(
    base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ["GROQ_API_KEY"],
)
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def mcp_tool_to_openai(tool) -> dict:
    """Convierte una tool del formato MCP al formato de function calling de OpenAI/Groq.

    MCP expone tools con name, description e inputSchema (JSON Schema).
    La API de OpenAI/Groq espera un dict con type="function" y el schema
    dentro de "parameters".

    Args:
        tool: Objeto tool de MCP con atributos name, description, inputSchema.

    Returns:
        Dict en formato OpenAI function calling.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


async def load_resources_context(session: ClientSession) -> str:
    """Lee el resource de índice de documentación para inyectarlo como contexto del sistema.

    El resource docs://index provee un resumen estructurado de todos los docs
    disponibles, lo que le da al LLM una visión general sin gastar una tool call.

    Args:
        session: Sesión MCP activa.

    Returns:
        Texto del índice de documentación, o string vacío si no está disponible.
    """
    try:
        result = await session.read_resource("docs://index")
        parts = []
        for content in result.contents:
            if hasattr(content, "text"):
                parts.append(content.text)
        return "\n".join(parts)
    except Exception as e:
        print(f"[DEBUG] No se pudo leer resource docs://index: {e}", file=sys.stderr)
        return ""


async def handle_prompt_command(session: ClientSession, command: str) -> str | None:
    """Procesa comandos especiales que invocan prompts del servidor MCP.

    Comandos disponibles:
        /prompts                         — Lista prompts disponibles
        /prompt <nombre> [arg=valor ...]  — Ejecuta un prompt y envía al LLM
        /resources                       — Lista resources disponibles
        /resource <uri>                  — Lee un resource por URI

    Args:
        session: Sesión MCP activa.
        command: Texto del usuario (ya sin strip).

    Returns:
        La respuesta generada, o None si el comando no es un /comando especial.
    """
    if not command.startswith("/"):
        return None

    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args_str = parts[1] if len(parts) > 1 else ""

    # ── /prompts — listar prompts disponibles ──
    if cmd == "/prompts":
        result = await session.list_prompts()
        if not result.prompts:
            return "No hay prompts disponibles en el servidor."

        lines = ["Prompts disponibles:\n"]
        for p in result.prompts:
            desc = f" — {p.description}" if p.description else ""
            lines.append(f"  {p.name}{desc}")
            if p.arguments:
                for arg in p.arguments:
                    req = " (requerido)" if arg.required else " (opcional)"
                    lines.append(f"    - {arg.name}{req}")
        return "\n".join(lines)

    # ── /prompt <nombre> [arg=valor ...] — ejecutar un prompt ──
    if cmd == "/prompt":
        if not args_str:
            return "Uso: /prompt <nombre> [arg=valor ...]\nEjemplo: /prompt summarize_doc filename=api.md"

        prompt_parts = args_str.split()
        prompt_name = prompt_parts[0]

        # Parsear argumentos key=value
        prompt_args = {}
        for part in prompt_parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                prompt_args[key] = value
            else:
                return f"Argumento inválido: '{part}'. Usa el formato key=value."

        print(
            f"[DEBUG] Ejecutando prompt: {prompt_name}({prompt_args})",
            file=sys.stderr,
        )

        try:
            result = await session.get_prompt(
                prompt_name,
                arguments=prompt_args if prompt_args else None,
            )
        except Exception as e:
            return f"Error al obtener prompt '{prompt_name}': {e}"

        # Convertir los mensajes del prompt al formato OpenAI
        messages = []
        for msg in result.messages:
            # Extraer texto del contenido del mensaje
            if hasattr(msg.content, "text"):
                content_text = msg.content.text
            elif isinstance(msg.content, str):
                content_text = msg.content
            else:
                content_text = str(msg.content)
            messages.append({"role": msg.role, "content": content_text})

        # Enviar los mensajes del prompt al LLM
        print(
            f"[DEBUG] Enviando {len(messages)} mensaje(s) del prompt al LLM",
            file=sys.stderr,
        )

        response = groq.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente técnico. Responde de forma concisa y precisa.",
                },
                *messages,
            ],
        )

        return response.choices[0].message.content or "(Sin respuesta)"

    # ── /resources — listar resources disponibles ──
    if cmd == "/resources":
        # Listar resources estáticos
        result = await session.list_resources()
        templates = await session.list_resource_templates()

        lines = ["Resources disponibles:\n"]

        if result.resources:
            lines.append("  Estáticos:")
            for r in result.resources:
                desc = f" — {r.description}" if r.description else ""
                lines.append(f"    {r.uri}{desc}")

        if templates.resourceTemplates:
            lines.append("  Templates:")
            for t in templates.resourceTemplates:
                desc = f" — {t.description}" if t.description else ""
                lines.append(f"    {t.uriTemplate}{desc}")

        if not result.resources and not templates.resourceTemplates:
            return "No hay resources disponibles en el servidor."

        return "\n".join(lines)

    # ── /resource <uri> — leer un resource ──
    if cmd == "/resource":
        if not args_str:
            return "Uso: /resource <uri>\nEjemplo: /resource docs://index"

        uri = args_str.strip()
        print(f"[DEBUG] Leyendo resource: {uri}", file=sys.stderr)

        try:
            result = await session.read_resource(uri)
            parts = []
            for content in result.contents:
                if hasattr(content, "text"):
                    parts.append(content.text)
            return "\n".join(parts) if parts else "(Resource vacío)"
        except Exception as e:
            return f"Error al leer resource '{uri}': {e}"

    return None  # No es un comando reconocido, tratar como pregunta normal


async def run_agent(session: ClientSession, user_question: str) -> str:
    """Ejecuta el loop agéntico hasta que el LLM devuelva una respuesta final.

    El loop funciona así:
    1. Carga el índice de docs como contexto del sistema (vía resource)
    2. Envía la pregunta + tools al LLM
    3. Si el LLM responde con tool_calls, ejecuta cada tool vía MCP
    4. Agrega los resultados al historial y vuelve a llamar al LLM
    5. Si el LLM responde con texto sin tool_calls, devuelve ese texto

    Args:
        session: Sesión MCP activa conectada al servidor.
        user_question: Pregunta del usuario.

    Returns:
        La respuesta final del LLM como string.
    """
    # Obtener tools del servidor MCP y convertirlas al formato OpenAI
    tools_result = await session.list_tools()
    openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]

    # Cargar el índice de documentación como contexto del sistema
    docs_context = await load_resources_context(session)
    system_content = (
        "Eres un asistente técnico que responde preguntas sobre documentación "
        "usando las tools disponibles. Sé conciso y preciso. Si no encuentras "
        "la información, dilo honestamente."
    )
    if docs_context:
        system_content += (
            f"\n\nAquí tienes un índice de la documentación disponible "
            f"para que sepas qué archivos y secciones existen:\n\n{docs_context}"
        )

    # Inicializar historial de mensajes
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question},
    ]

    max_iterations = 10

    for iteration in range(max_iterations):
        # Llamar al LLM con el historial y las tools disponibles
        response = groq.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools,
        )

        assistant_message = response.choices[0].message

        # Caso A: el LLM devolvió una respuesta final sin tool_calls
        if not assistant_message.tool_calls:
            return assistant_message.content or "(Sin respuesta del modelo)"

        # Caso B: el LLM quiere invocar tools
        # Primero, agregar el mensaje del assistant con los tool_calls al historial
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ],
        })

        # Ejecutar cada tool y agregar resultados al historial
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(
                f"[DEBUG] Invocando: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})",
                file=sys.stderr,
            )

            try:
                result = await session.call_tool(tool_name, arguments=tool_args)
                # Extraer texto del resultado MCP
                result_text = "\n".join(
                    block.text for block in result.content if hasattr(block, "text")
                )
            except Exception as e:
                result_text = f"Error al ejecutar {tool_name}: {e}"
                print(f"[DEBUG] Error: {result_text}", file=sys.stderr)

            # Agregar el resultado al historial con el mismo tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

        print(
            f"[DEBUG] Iteración {iteration + 1} completada, {len(assistant_message.tool_calls)} tool(s) ejecutada(s)",
            file=sys.stderr,
        )

    return f"El agente no pudo resolver la pregunta en {max_iterations} pasos."


async def main():
    """Punto de entrada principal. Arranca el servidor MCP y corre el REPL."""
    # Parámetros para arrancar docs_server.py como subproceso por stdio
    server_params = StdioServerParameters(
        command="python",
        args=["docs_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Mostrar las capabilities del servidor al inicio
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]

            resources_result = await session.list_resources()
            templates_result = await session.list_resource_templates()
            resource_count = len(resources_result.resources) + len(templates_result.resourceTemplates)

            prompts_result = await session.list_prompts()
            prompt_names = [p.name for p in prompts_result.prompts]

            print("Cliente MCP conectado a DocsServer.")
            print(f"  Tools:     {', '.join(tool_names)}")
            print(f"  Resources: {resource_count} disponible(s)")
            print(f"  Prompts:   {', '.join(prompt_names) or 'ninguno'}")
            print()
            print("Comandos especiales:")
            print("  /prompts                          — Lista prompts disponibles")
            print("  /prompt <nombre> [arg=valor ...]   — Ejecuta un prompt")
            print("  /resources                        — Lista resources disponibles")
            print("  /resource <uri>                   — Lee un resource")
            print("  salir                             — Termina el cliente")
            print()

            while True:
                try:
                    question = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nHasta luego.")
                    break

                if question.lower() in ("salir", "exit", "quit"):
                    print("Hasta luego.")
                    break
                if not question:
                    continue

                try:
                    # Primero intentar como comando /prompt o /resource
                    prompt_result = await handle_prompt_command(session, question)
                    if prompt_result is not None:
                        print(f"\n{prompt_result}\n")
                        continue

                    # Si no es un comando especial, usar el loop agéntico
                    answer = await run_agent(session, question)
                    print(f"\n{answer}\n")
                except Exception as e:
                    print(f"Error al procesar la pregunta: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
