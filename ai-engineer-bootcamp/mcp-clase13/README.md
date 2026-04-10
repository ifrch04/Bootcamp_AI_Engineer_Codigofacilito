# Clase 13 MCP — Soluciones de Referencia

> **Este repo contiene las soluciones de referencia del instructor.** Los estudiantes tienen sus propias instrucciones en `PRACTICAS.md`. No compartas este código directamente.

## Contexto

Clase 13 del Bootcamp AI Engineer (Código Facilito). Cubre Model Context Protocol (MCP) en tres prácticas acumulativas:

1. **server.py** — Servidor MCP simple con tools de hora y calculadora
2. **docs_server.py** — Servidor MCP DocOps que opera sobre documentación markdown
3. **client.py** — Cliente MCP custom que usa Groq como LLM e implementa el loop agéntico

## Setup

```bash
# Desde la raíz del repo del bootcamp (ai-engineer-bootcamp/)
# Activar el venv del proyecto
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Las dependencias ya están en el venv raíz. Si necesitas reinstalar:
pip install "mcp[cli]" openai python-dotenv

# Configurar variables de entorno
cd mcp-clase13
cp .env.example .env
# Edita .env con tu API key de Groq
```

## Cómo correr cada práctica

### Práctica 1 — server.py con Inspector

```bash
cd mcp-clase13
npx @modelcontextprotocol/inspector python server.py
```

Abre el Inspector en `http://localhost:5173`. Deberías ver 2 tools: `get_current_time` y `calculate`.

Pruebas manuales:
- `get_current_time("America/Mexico_City")` → hora actual en formato ISO
- `get_current_time("Mars/Olympus")` → mensaje de error descriptivo
- `calculate("2 * (3 + 4)")` → `14`
- `calculate("import os")` → error, rechaza expresión insegura

### Práctica 2 — docs_server.py con Inspector

```bash
npx @modelcontextprotocol/inspector python docs_server.py
```

Deberías ver 4 tools. Pruebas clave:
- `list_docs()` → `["api.md", "changelog.md", "troubleshooting.md"]`
- `read_doc("api.md")` → contenido completo
- `read_doc("../../../etc/passwd")` → error de path traversal
- `read_doc("noexiste.md")` → error con lista de archivos disponibles
- `search_docs("rate limit")` → matches en troubleshooting.md Y changelog.md
- `search_docs("ENDPOINT")` → case-insensitive, encuentra en api.md
- `get_doc_sections("api.md")` → lista de headings con niveles

### Práctica 3 — client.py (loop agéntico)

```bash
python client.py
```

Preguntas de prueba:
1. `¿Qué archivos de documentación tengo disponibles?` → llama `list_docs()`
2. `¿Qué endpoints están documentados en mi API?` → llama `list_docs()` o `search_docs()`, luego `read_doc("api.md")`
3. `Tengo un error 429, ¿qué hago?` → llama `search_docs("429")` o `search_docs("rate limit")`

Los logs `[DEBUG]` en stderr muestran cada tool invocada.

## Puntos de atención para enseñar

- **stdout vs stderr en servidores MCP stdio:** `print()` sin `file=sys.stderr` rompe el protocolo MCP porque stdout es el canal de comunicación. Es el error más silencioso y confuso que van a encontrar.

- **Docstrings como documentación de API para el LLM:** Un docstring vago = el LLM ignora la tool o la llama mal. Mostrar la diferencia entre un docstring bueno y uno malo es un momento pedagógico fuerte.

- **Seguridad en `calculate`:** Muchos van a querer usar `eval()`. Mostrar por qué es peligroso (`eval("__import__('os').system('rm -rf /')")`) y cómo el walker de AST resuelve el problema.

- **`tool_call_id` debe matchear exactamente:** Si el id que devuelves en el `role=tool` message no coincide con el que envió el modelo, el siguiente request al LLM falla. Es un bug sutil.

- **Orden del historial importa:** El message del assistant con `tool_calls` debe ir ANTES de los messages `role=tool`. Si el orden está invertido, el modelo no entiende qué tools ya ejecutó.

- **Path traversal en `read_doc`:** Un LLM puede alucinar paths como `"../../etc/passwd"` sin mala intención. Validar siempre.

## Script de demos automatizado

`run_demos.py` ejecuta las demos de las 3 prácticas de forma automatizada con output colorizado:

```bash
cd mcp-clase13
python run_demos.py         # Ejecuta las 3 prácticas
python run_demos.py 1       # Solo práctica 1
python run_demos.py 2       # Solo práctica 2
python run_demos.py 3       # Solo práctica 3 (requiere GROQ_API_KEY en .env)
python run_demos.py 1 2     # Prácticas 1 y 2
```

El script se conecta a cada servidor MCP, ejecuta las tools programáticamente, y valida los resultados con indicadores visuales (checkmarks verdes / errores rojos).
