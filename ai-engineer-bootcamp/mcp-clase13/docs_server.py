"""
Práctica 2 — Servidor MCP "DocsServer"

Servidor MCP que opera sobre una carpeta docs/ con archivos markdown.
Expone tools, resources y prompts para interactuar con la documentación.
Este es el patrón base del DocOps Agent del capstone.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocsServer")
DOCS_DIR = Path("docs")


def _safe_path(filename: str) -> Path:
    """Valida un filename contra path traversal y devuelve el Path seguro.

    Rechaza filenames que contengan '/', '\\' o '..' para prevenir acceso
    a archivos fuera de DOCS_DIR. Además verifica con resolve() que el
    path resultante esté dentro de DOCS_DIR.

    Args:
        filename: Nombre del archivo (solo el nombre, sin directorios).

    Returns:
        Path completo al archivo dentro de DOCS_DIR.

    Raises:
        ValueError: Si el filename contiene caracteres sospechosos o
                    el path resuelto está fuera de DOCS_DIR.
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"Filename inválido: '{filename}'. No se permiten '/', '\\\\' ni '..'.")
    path = DOCS_DIR / filename
    if not path.resolve().is_relative_to(DOCS_DIR.resolve()):
        raise ValueError("Path traversal detectado.")
    return path


def _list_available_docs() -> list[str]:
    """Devuelve la lista de archivos .md disponibles, para usar en mensajes de error."""
    if not DOCS_DIR.exists():
        return []
    return sorted(p.name for p in DOCS_DIR.glob("*.md"))


@mcp.tool()
def list_docs() -> list[str]:
    """Lista los archivos de documentación disponibles en la carpeta docs/.

    Úsala como primer paso cuando el usuario pregunte qué documentación existe
    o cuando necesites saber qué archivos hay disponibles antes de leer uno.
    No la uses si ya conoces el nombre exacto del archivo que necesitas.

    Returns:
        Lista de nombres de archivos .md disponibles, ordenados alfabéticamente.
        Lista vacía si no hay archivos o la carpeta no existe.

    Ejemplo:
        list_docs() → ["api.md", "changelog.md", "troubleshooting.md"]
    """
    try:
        return _list_available_docs()
    except Exception as e:
        return [f"Error al listar documentos: {e}"]


@mcp.tool()
def read_doc(filename: str) -> str:
    """Lee y devuelve el contenido completo de un archivo de documentación.

    Úsala cuando necesites ver el contenido de un documento específico cuyo
    nombre ya conoces. Si no estás seguro de qué archivo contiene la
    información que buscas, usa search_docs primero.

    El filename debe ser solo el nombre del archivo (ej: "api.md"), sin rutas
    ni directorios. Paths como "../secreto.txt" serán rechazados por seguridad.

    Args:
        filename: Nombre del archivo .md a leer, por ejemplo "api.md".

    Returns:
        El contenido completo del archivo como string.
        Si el archivo no existe o el filename es inválido, devuelve un
        mensaje de error con la lista de archivos disponibles.

    Ejemplo:
        read_doc("api.md") → "# API Reference\\n\\n## Authentication\\n..."
    """
    try:
        path = _safe_path(filename)
    except ValueError as e:
        available = _list_available_docs()
        return f"Error: {e} Archivos disponibles: {', '.join(available) or 'ninguno'}."

    if not path.exists():
        available = _list_available_docs()
        return (
            f"Archivo '{filename}' no encontrado. "
            f"Archivos disponibles: {', '.join(available) or 'ninguno'}."
        )

    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error al leer '{filename}': {e}"


@mcp.tool()
def search_docs(query: str) -> list[dict]:
    """Busca un término en todos los documentos disponibles (case-insensitive).

    Úsala cuando el usuario pregunte algo general y no sepas qué documento
    tiene la respuesta. Por ejemplo: "¿cómo manejo el error 429?" o "¿qué
    cambió en la última versión?".
    No la uses si ya sabes e l nombre del archivo — en ese caso llama a
    read_doc directamente, es más eficiente.

    Args:
        query: Término a buscar. La búsqueda es case-insensitive y busca
               substrings. Usa palabras específicas para mejores resultados.

    Returns:
        Lista de matches. Cada match es un dict con:
        - "filename": nombre del archivo donde se encontró
        - "line_number": número de línea del match (1-indexed)
        - "snippet": la línea del match más 1 línea de contexto arriba y abajo

        Lista vacía si no hay coincidencias.

    Ejemplo:
        search_docs("rate limit") → [{"filename": "troubleshooting.md", "line_number": 14, "snippet": "..."}]
    """
    if not query or not query.strip():
        return []

    results = []
    query_lower = query.lower()

    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        try:
            lines = doc_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            if query_lower in line.lower():
                # Construir snippet con 1 línea de contexto arriba y abajo
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                snippet = "\n".join(lines[start:end])

                results.append({
                    "filename": doc_path.name,
                    "line_number": i + 1,  # 1-indexed
                    "snippet": snippet,
                })

    return results


@mcp.tool()
def get_doc_sections(filename: str) -> list[str]:
    """Extrae los headings (secciones) de un documento markdown.

    Úsala para obtener la estructura de un documento antes de leerlo completo.
    Es útil cuando quieres saber qué temas cubre un archivo sin leer todo
    su contenido, o cuando necesitas dirigir al usuario a una sección específica.

    Args:
        filename: Nombre del archivo .md, por ejemplo "api.md".

    Returns:
        Lista de headings preservando el nivel markdown. Por ejemplo:
        ["# API Reference", "## Authentication", "## Endpoints", "### GET /users"].

        Si el archivo no existe o el filename es inválido, devuelve una lista
        con un solo string describiendo el error.

    Ejemplo:
        get_doc_sections("api.md") → ["# API Reference", "## Authentication", ...]
    """
    try:
        path = _safe_path(filename)
    except ValueError as e:
        return [f"Error: {e}"]

    if not path.exists():
        available = _list_available_docs()
        return [
            f"Archivo '{filename}' no encontrado. "
            f"Archivos disponibles: {', '.join(available) or 'ninguno'}."
        ]

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        # Extraer líneas que empiecen con uno o más # seguidos de espacio
        return [line for line in lines if line.lstrip().startswith("#") and " " in line]
    except Exception as e:
        return [f"Error al leer '{filename}': {e}"]


# ─── Resources ────────────────────────────────────────────────────────────────
# Los resources exponen datos estáticos o semi-estáticos por URI.
# A diferencia de las tools (que ejecutan lógica), un resource simplemente
# devuelve contenido — el cliente puede leerlos sin que el LLM los invoque.


@mcp.resource("docs://index", description="Índice de todos los documentos disponibles")
def docs_index() -> str:
    """Devuelve un índice con los nombres y secciones de todos los docs."""
    docs = _list_available_docs()
    if not docs:
        return "No hay documentos disponibles."

    lines = ["# Índice de Documentación", ""]
    for doc_name in docs:
        path = DOCS_DIR / doc_name
        try:
            content = path.read_text(encoding="utf-8")
            sections = [
                line.strip()
                for line in content.splitlines()
                if line.strip().startswith("#") and " " in line
            ]
            lines.append(f"## {doc_name}")
            for section in sections:
                # Indentar sub-secciones para legibilidad
                lines.append(f"  - {section}")
            lines.append("")
        except Exception:
            lines.append(f"## {doc_name} (error al leer)")
            lines.append("")

    return "\n".join(lines)


@mcp.resource(
    "docs://files/{filename}",
    description="Contenido de un archivo de documentación específico",
    mime_type="text/markdown",
)
def docs_file(filename: str) -> str:
    """Devuelve el contenido de un archivo markdown por nombre.

    El URI template permite acceder a cada doc individualmente,
    por ejemplo: docs://files/api.md
    """
    try:
        path = _safe_path(filename)
    except ValueError as e:
        return f"Error: {e}"

    if not path.exists():
        available = _list_available_docs()
        return (
            f"Archivo '{filename}' no encontrado. "
            f"Disponibles: {', '.join(available) or 'ninguno'}."
        )

    return path.read_text(encoding="utf-8")


# ─── Prompts ──────────────────────────────────────────────────────────────────
# Los prompts son plantillas de mensajes reutilizables. El cliente puede
# listarlos, pedir uno con argumentos, e inyectar los mensajes resultantes
# en la conversación con el LLM. Son útiles para estandarizar flujos.


@mcp.prompt(
    name="summarize_doc",
    description="Genera un prompt para que el LLM resuma un documento específico",
)
def summarize_doc(filename: str) -> list[dict]:
    """Construye un prompt de resumen inyectando el contenido del doc.

    Args:
        filename: Nombre del archivo a resumir (ej: "api.md").
    """
    try:
        path = _safe_path(filename)
    except ValueError as e:
        return [{"role": "user", "content": f"Error: {e}"}]

    if not path.exists():
        available = _list_available_docs()
        return [
            {
                "role": "user",
                "content": (
                    f"No se encontró '{filename}'. "
                    f"Archivos disponibles: {', '.join(available)}."
                ),
            }
        ]

    content = path.read_text(encoding="utf-8")
    return [
        {
            "role": "user",
            "content": (
                f"Resume el siguiente documento de forma concisa. "
                f"Incluye los puntos más importantes y cualquier breaking change.\n\n"
                f"Documento: {filename}\n\n{content}"
            ),
        },
    ]


@mcp.prompt(
    name="troubleshoot",
    description="Genera un prompt para diagnosticar un error específico usando la documentación",
)
def troubleshoot(error_description: str) -> list[dict]:
    """Construye un prompt de troubleshooting con contexto de la documentación.

    Lee el archivo troubleshooting.md (si existe) y lo inyecta como contexto
    para que el LLM pueda diagnosticar el error del usuario.

    Args:
        error_description: Descripción del error que está experimentando el usuario.
    """
    # Intentar leer troubleshooting.md como contexto
    troubleshooting_path = DOCS_DIR / "troubleshooting.md"
    context = ""
    if troubleshooting_path.exists():
        context = troubleshooting_path.read_text(encoding="utf-8")

    messages = []

    if context:
        messages.append({
            "role": "user",
            "content": (
                f"Tengo el siguiente problema:\n\n{error_description}\n\n"
                f"Usa esta guía de troubleshooting como referencia para diagnosticar "
                f"y sugerir una solución:\n\n{context}"
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": (
                f"Tengo el siguiente problema:\n\n{error_description}\n\n"
                f"No hay guía de troubleshooting disponible. "
                f"Sugiere pasos generales de diagnóstico."
            ),
        })

    return messages


@mcp.prompt(
    name="compare_versions",
    description="Genera un prompt para comparar cambios entre versiones del changelog",
)
def compare_versions(version: str) -> list[dict]:
    """Construye un prompt para analizar los cambios de una versión específica.

    Lee el changelog y pide al LLM que analice los cambios, destacando
    breaking changes y su impacto.

    Args:
        version: Versión a analizar (ej: "v2.3.0").
    """
    changelog_path = DOCS_DIR / "changelog.md"

    if not changelog_path.exists():
        return [
            {
                "role": "user",
                "content": "No se encontró changelog.md en la documentación.",
            }
        ]

    content = changelog_path.read_text(encoding="utf-8")
    return [
        {
            "role": "user",
            "content": (
                f"Analiza los cambios de la versión {version} en este changelog. "
                f"Destaca:\n"
                f"1. Breaking changes y su impacto\n"
                f"2. Nuevas funcionalidades\n"
                f"3. Bug fixes importantes\n"
                f"4. Acciones que un developer debe tomar al actualizar\n\n"
                f"{content}"
            ),
        },
    ]


if __name__ == "__main__":
    mcp.run()
