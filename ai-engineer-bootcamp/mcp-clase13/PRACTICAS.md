# Clase 13 — Prácticas MCP

**Duración total:** ~110 minutos · **Stack:** Python 3.11+, pip, Groq free tier, MCP Inspector

Al final de esta clase vas a tener servidores MCP funcionando y los vas a consumir desde un cliente real: ya sea Claude Desktop o un cliente CLI que tú escribes. Todo lo que construyes hoy conecta con el **DocOps Agent del proyecto final**.

Las tres prácticas son **acumulativas**: en las Prácticas 1 y 2 construyes los servidores MCP, y en la Práctica 3 los conectas a un cliente que los consume. Tienes dos caminos para la Práctica 3 — elige el que prefieras.

---

## Setup común (10 minutos)

Haz esto una sola vez al inicio. Todas las prácticas comparten el mismo entorno.

### Prerrequisitos

Necesitas tener instalado:

1. **Python 3.11 o superior** — `python --version` para verificar
2. **pip** — viene incluido con Python
3. **Node.js 18+** — lo necesitas para correr el MCP Inspector (debugging visual). `node --version` para verificar. Si no lo tienes, instálalo desde nodejs.org
4. **Una API key de Groq** — ya la tienes del bootcamp. Si no, créala gratis en console.groq.com/keys. Solo necesaria si eliges la Opción B de la Práctica 3.
5. **Claude Desktop** (opcional) — solo si eliges la Opción A de la Práctica 3. Descárgalo desde claude.ai/download.

### Crear el proyecto

Abre una terminal y ejecuta:

```bash
mkdir mcp-clase13 && cd mcp-clase13
```

Crea un virtual environment e instala las dependencias:

```bash
# Linux / Mac
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv venv
venv\Scripts\activate.bat
```

Instala los paquetes necesarios:

```bash
pip install "mcp[cli]" openai python-dotenv
```

`mcp[cli]` incluye el SDK de servidor/cliente más el comando `mcp` para dev tooling. `openai` es el SDK para hablar con Groq (Groq es compatible con la API de OpenAI, así que el mismo SDK funciona sin cambios).

### Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
GROQ_API_KEY=tu_api_key_aqui
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile
```

Usamos `llama-3.3-70b-versatile` porque soporta tool calling de forma robusta en el free tier de Groq.

Crea un `.gitignore` para no subir archivos sensibles:

```
venv/
.env
__pycache__/
*.pyc
```

### Verificar que el MCP Inspector funciona

El Inspector es una interfaz web que se conecta a tu servidor y te deja invocar tools manualmente sin gastar tokens del LLM. Lo vamos a usar en las Prácticas 1 y 2 como herramienta principal de debugging.

```bash
npx @modelcontextprotocol/inspector --help
```

La primera vez npm descarga el paquete (~30 segundos). Si ves la ayuda del comando, estás listo.

---

## Práctica 1 — Hola MCP (20 minutos)

### Contexto

El objetivo de esta práctica no es construir algo útil — es matar la fricción de setup y ver con tus propios ojos el ciclo completo: defines una función en Python, la decoras como tool, arrancas el servidor, y la invocas desde el Inspector. Cuando veas tu tool ejecutándose, vas a entender que MCP no es magia — es un protocolo muy delgado encima de funciones Python.

### Lo que vas a construir

Un archivo `server.py` con dos tools:

**`get_current_time(timezone: str) -> str`** — Devuelve la hora actual en formato ISO 8601 para la zona horaria dada. Usa el módulo `zoneinfo` de la stdlib (no uses `pytz`). Si la timezone es inválida (ej. `"Mars/Olympus"`), devuelve un mensaje de error claro, no lances excepción.

- `get_current_time("America/Mexico_City")` → `"2026-04-08T14:23:45-06:00"`
- `get_current_time("UTC")` → `"2026-04-08T20:23:45+00:00"`
- `get_current_time("Marte/Olympus")` → mensaje de error descriptivo

**`calculate(expression: str) -> str`** — Evalúa una expresión matemática de forma segura y devuelve el resultado como string.

**Requisito crítico de seguridad:** NO uses `eval()` crudo. Un `eval()` sin restricciones ejecuta cualquier código Python — `eval("__import__('os').system('rm -rf /')")` borraría tu disco. Implementa un evaluador seguro usando `ast.parse` + un walker recursivo que **solo acepte**: números (`int`, `float`), operadores aritméticos (`+`, `-`, `*`, `/`, `%`, `**`), operador unario negativo, y paréntesis. Cualquier otro nodo del AST (Name, Call, Attribute, Import, etc.) debe rechazarse.

- `calculate("2 * (3 + 4)")` → `"14"`
- `calculate("(100 - 25) / 5")` → `"15.0"`
- `calculate("2 ** 10")` → `"1024"`
- `calculate("import os")` → error, rechaza
- `calculate("__import__('os').system('ls')")` → error, rechaza

### Estructura inicial

Crea `server.py` con este esqueleto. Tú completas las tools:

```python
import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

# Crea el servidor
mcp = FastMCP("HolaMCP")

# TODO: define aquí los operadores permitidos para el evaluador seguro
# Pista: un dict que mapee nodos AST (ast.Add, ast.Sub, etc.) a funciones
# del módulo operator (operator.add, operator.sub, etc.)


# TODO: función helper _safe_eval(node) que recorra el AST recursivamente
# y solo acepte Constant (números), BinOp y UnaryOp con operadores permitidos


@mcp.tool()
def get_current_time(timezone: str) -> str:
    """
    TODO: escribe un docstring útil.

    Recuerda: el LLM lee este docstring para decidir cuándo usar la tool.
    Describe qué hace, qué recibe, qué devuelve, y cuándo usarla.
    """
    # TODO: implementa
    pass


@mcp.tool()
def calculate(expression: str) -> str:
    """
    TODO: docstring.
    """
    # TODO: implementa de forma segura con ast.parse + _safe_eval
    pass


if __name__ == "__main__":
    mcp.run()
```

### Cómo probarlo

Desde la raíz del proyecto:

```bash
npx @modelcontextprotocol/inspector python server.py
```

Esto lanza tu servidor con `python server.py` y abre el Inspector en el navegador (típicamente `http://localhost:5173`). El Inspector se conecta a tu servidor por stdio automáticamente.

En la UI del Inspector deberías ver:
- Un panel "Tools" con tus dos tools listadas
- Al hacer clic en cada tool, un formulario con los parámetros
- Un botón "Run Tool" que invoca la tool y muestra el resultado

**Pruebas manuales a ejecutar:**

| Input | Resultado esperado |
|---|---|
| `get_current_time("America/Mexico_City")` | Hora actual en formato ISO 8601 |
| `get_current_time("UTC")` | Hora en UTC |
| `get_current_time("Mars/Olympus")` | Mensaje de error descriptivo |
| `calculate("2 * (3 + 4)")` | `14` |
| `calculate("2 ** 10")` | `1024` |
| `calculate("(100 - 25) / 5")` | `15.0` |
| `calculate("import os")` | Error: rechaza la expresión |
| `calculate("__import__('os').system('ls')")` | Error: rechaza la expresión |

### Criterio de éxito

1. Las dos tools aparecen listadas en el Inspector con sus docstrings visibles
2. `get_current_time("America/Mexico_City")` devuelve la hora actual correcta
3. `get_current_time("Mars/Olympus")` devuelve un mensaje de error sin crashear el servidor
4. `calculate("2 * (3 + 4)")` devuelve `14`
5. `calculate("import os; os.system('ls')")` **NO** ejecuta nada — rechaza la entrada con un error claro
6. Ninguna tool lanza excepciones no capturadas (el servidor nunca crashea)

### Reflexión

Antes de pasar a la Práctica 2, responde estas preguntas:

1. **Sobre el decorador `@mcp.tool()`:** ¿Qué crees que hace internamente? ¿Cómo sabe el servidor qué parámetros acepta cada tool? Pista: piensa en type hints y la función `inspect` de Python.

2. **Sobre los docstrings:** Si escribes un docstring genérico como "Hace un cálculo", ¿cómo afecta esto al LLM que va a decidir cuándo usar la tool? ¿Qué información crítica le estarías quitando?

3. **Sobre la seguridad de `calculate`:** ¿Por qué no basta con hacer `eval(expression)` dentro de un try/except? ¿Qué podría ejecutar un usuario malicioso (o un LLM que alucina) si usas eval sin restricciones?

### Bonus

- Agrega una tercera tool `convert_temperature(value: float, from_unit: str, to_unit: str)` que convierta entre Celsius, Fahrenheit y Kelvin
- Haz que `get_current_time` acepte un formato opcional (`format: str = "iso"`) — si es `"human"`, devuelve algo como `"Miércoles 8 de abril, 2:23 PM"`
- Agrega validación en `calculate` para división entre cero con un mensaje claro

---

## Práctica 2 — Servidor DocOps (45 minutos)

### Contexto

Ahora vas a construir algo útil de verdad: un servidor MCP que expone una carpeta de documentación markdown como tools, resources y prompts consumibles por un LLM. Este es exactamente el patrón que vas a extender en el DocOps Agent del capstone.

El foco de esta práctica **no es la cantidad de tools sino su diseño**. Vas a pensar en cada tool como una API que otro developer (el LLM) va a consumir a ciegas leyendo solo los docstrings. Si el docstring es ambiguo, el LLM ignora la tool o la llama mal.

### Lo que vas a construir

Un servidor `docs_server.py` que opera sobre una carpeta `docs/` con archivos markdown. Expone **4 tools**, **2 resources** y **3 prompts**.

#### Tools

1. **`list_docs() -> list[str]`** — Devuelve los nombres de los archivos `.md` disponibles en `docs/`, ordenados alfabéticamente.

2. **`read_doc(filename: str) -> str`** — Devuelve el contenido completo de un doc por nombre. Debe rechazar rutas con `..` o `/` (path traversal) y devolver un error claro si el archivo no existe. El error debe incluir la lista de archivos disponibles para que el LLM pueda autocorregirse.

3. **`search_docs(query: str) -> list[dict]`** — Busca el query como substring (case-insensitive) en todos los docs. Devuelve una lista de resultados donde cada resultado es un dict con `{"filename": str, "snippet": str, "line_number": int}`. El snippet incluye la línea que hizo match más 1 línea de contexto arriba y abajo.

4. **`get_doc_sections(filename: str) -> list[str]`** — Lee un doc y devuelve la lista de headings (líneas que empiezan con `#`, `##`, etc.), preservando el nivel.

#### Resources

Los resources exponen datos por URI. A diferencia de las tools (que ejecutan lógica), un resource simplemente devuelve contenido — el cliente puede leerlos directamente sin que el LLM los invoque.

5. **`docs://index`** (resource estático) — Devuelve un índice estructurado con los nombres y secciones de todos los documentos. Útil para que el cliente inyecte contexto en el system prompt sin gastar una tool call.

6. **`docs://files/{filename}`** (resource template) — Expone cada archivo markdown por URI. Por ejemplo, `docs://files/api.md` devuelve el contenido de `api.md`. El `{filename}` en la URI es un parámetro que se pasa a la función.

#### Prompts

Los prompts son plantillas de mensajes reutilizables. El cliente puede listarlos, pedir uno con argumentos, y recibir mensajes pre-construidos para inyectar en la conversación con el LLM.

7. **`summarize_doc(filename: str)`** — Genera un mensaje que incluye el contenido de un doc y pide al LLM que lo resuma.

8. **`troubleshoot(error_description: str)`** — Lee `troubleshooting.md` como contexto y genera un mensaje para que el LLM diagnostique un error.

9. **`compare_versions(version: str)`** — Lee el changelog y genera un mensaje para que el LLM analice los cambios de una versión.

### Preparar la data de prueba

Crea una carpeta `docs/` en tu proyecto con tres archivos markdown. El contenido debe ser realista — no lorem ipsum. Aquí tienes una guía de qué poner en cada uno (personaliza el contenido como quieras):

**`docs/api.md`** (~40-60 líneas) — Documentación de una API REST ficticia:
- Sección de autenticación (Bearer token)
- Al menos 3-4 endpoints documentados (`GET /users`, `POST /users`, etc.)
- Cada endpoint con descripción, parámetros y códigos de error
- Una sección de rate limiting que mencione "100 requests por minuto"

**`docs/troubleshooting.md`** (~30-50 líneas) — Guía de troubleshooting:
- Al menos 4 secciones de error (401, 403, 429, 500, database timeout, etc.)
- La sección de 429 debe mencionar explícitamente "rate limit"
- Cada sección con causa y solución

**`docs/changelog.md`** (~25-40 líneas) — Changelog:
- Al menos 3 versiones
- Al menos un breaking change marcado explícitamente
- Al menos un bugfix relacionado con rate limiting (para que las búsquedas encuentren matches en dos archivos)

### Estructura inicial

Crea `docs_server.py` con este esqueleto:

```python
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocsServer")
DOCS_DIR = Path("docs")


# TODO: función helper _safe_path(filename) para validar filenames
# (ver sección "Consideraciones de diseño" abajo)


# ─── Tools ────────────────────────────────────────────────────────

@mcp.tool()
def list_docs() -> list[str]:
    """
    TODO: docstring claro.
    Piensa: ¿cuándo querría el LLM llamar esta tool?
    """
    # TODO: implementa
    pass


@mcp.tool()
def read_doc(filename: str) -> str:
    """
    TODO: docstring.
    Documenta explícitamente qué pasa si el archivo no existe
    o si el filename contiene caracteres sospechosos.
    """
    # TODO: valida el filename con _safe_path, lee el archivo, maneja errores
    pass


@mcp.tool()
def search_docs(query: str) -> list[dict]:
    """
    TODO: docstring.
    Incluye cuándo usar esta tool y cuándo NO usarla.
    """
    # TODO: itera archivos, busca substring case-insensitive,
    # devuelve snippets con contexto
    pass


@mcp.tool()
def get_doc_sections(filename: str) -> list[str]:
    """
    TODO: docstring.
    """
    # TODO: lee el archivo, extrae headings (líneas que empiezan con #)
    pass


# ─── Resources ────────────────────────────────────────────────────

@mcp.resource("docs://index", description="Índice de todos los documentos disponibles")
def docs_index() -> str:
    """Devuelve un índice con los nombres y secciones de todos los docs."""
    # TODO: itera los archivos .md, extrae sus headings,
    # y devuelve un string markdown con la estructura:
    #   ## api.md
    #     - # API Reference
    #     - ## Authentication
    #     - ...
    pass


@mcp.resource(
    "docs://files/{filename}",
    description="Contenido de un archivo de documentación específico",
    mime_type="text/markdown",
)
def docs_file(filename: str) -> str:
    """Devuelve el contenido de un archivo markdown por nombre.

    El parámetro {filename} de la URI se pasa como argumento a la función.
    Por ejemplo: docs://files/api.md → filename="api.md"
    """
    # TODO: valida con _safe_path y devuelve el contenido
    pass


# ─── Prompts ──────────────────────────────────────────────────────

@mcp.prompt(
    name="summarize_doc",
    description="Genera un prompt para que el LLM resuma un documento específico",
)
def summarize_doc(filename: str) -> list[dict]:
    """Construye un prompt de resumen inyectando el contenido del doc."""
    # TODO: lee el archivo y devuelve una lista con un dict:
    # [{"role": "user", "content": "Resume el siguiente documento...\n\n{contenido}"}]
    pass


@mcp.prompt(
    name="troubleshoot",
    description="Genera un prompt para diagnosticar un error usando la documentación",
)
def troubleshoot(error_description: str) -> list[dict]:
    """Construye un prompt de troubleshooting con contexto de la documentación."""
    # TODO: lee troubleshooting.md como contexto y genera un mensaje
    # que combine la descripción del error con la guía
    pass


@mcp.prompt(
    name="compare_versions",
    description="Genera un prompt para analizar cambios de una versión del changelog",
)
def compare_versions(version: str) -> list[dict]:
    """Construye un prompt para analizar los cambios de una versión."""
    # TODO: lee changelog.md y genera un mensaje pidiendo al LLM
    # que analice breaking changes, nuevas features y bug fixes
    pass


if __name__ == "__main__":
    mcp.run()
```

### Consideraciones de diseño críticas

#### 1. Docstrings como API para el LLM

El LLM lee tus docstrings para decidir qué tool usar. Regla práctica: tu docstring debe dejar claras tres cosas — **qué hace**, **qué recibe**, y **cuándo usarla** (y cuándo NO). Un buen ejemplo:

```python
@mcp.tool()
def search_docs(query: str) -> list[dict]:
    """Busca un término en todos los documentos disponibles.

    Úsalo cuando el usuario pregunte algo general y no sepas qué documento
    tiene la respuesta. Por ejemplo: "¿cómo manejo el error 429?".

    No lo uses si ya sabes el nombre del archivo — en ese caso llama a
    read_doc directamente.

    Args:
        query: término a buscar, case-insensitive.

    Returns:
        Lista de matches con filename, snippet y line_number.
    """
```

#### 2. Validación de path traversal

`read_doc`, `get_doc_sections` y el resource `docs://files/{filename}` reciben un filename. Un LLM podría pasar `"../../../etc/passwd"` (con toda la buena intención del mundo, porque alucinó). Crea un helper `_safe_path` que valide esto:

```python
def _safe_path(filename: str) -> Path:
    """Valida un filename contra path traversal."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"Filename inválido: {filename}")
    path = DOCS_DIR / filename
    if not path.resolve().is_relative_to(DOCS_DIR.resolve()):
        raise ValueError("Path traversal detectado")
    return path
```

La doble validación (caracteres sospechosos + `resolve().is_relative_to()`) es importante: la primera atrapa los casos obvios, la segunda atrapa edge cases que no anticipaste.

#### 3. Mensajes de error que permiten autocorrección

Si el archivo no existe, no devuelvas un error críptico. Devuelve un mensaje que el LLM pueda usar para corregir su estrategia:

```
"Archivo 'xxx.md' no encontrado. Archivos disponibles: api.md, troubleshooting.md, changelog.md"
```

Esto le permite al LLM autocorregirse en el siguiente paso del loop agéntico.

#### 4. Diferencias entre tools, resources y prompts

Piensa en las tres primitivas así:

- **Tools** = funciones que el LLM invoca activamente durante el razonamiento. El LLM decide cuándo y con qué argumentos llamarlas.
- **Resources** = datos que el cliente puede leer en cualquier momento sin intervención del LLM. Son como endpoints GET de una API REST. Útiles para inyectar contexto al inicio de una conversación.
- **Prompts** = plantillas de mensajes pre-construidos. El cliente los pide, recibe una lista de mensajes con roles (`user`, `assistant`), y los inyecta directamente en la conversación con el LLM. Útiles para estandarizar flujos comunes.

### Cómo probarlo

```bash
npx @modelcontextprotocol/inspector python docs_server.py
```

En el Inspector vas a ver tres paneles: **Tools**, **Resources** y **Prompts**. Prueba cada uno.

**Pruebas de tools:**

| Input | Resultado esperado |
|---|---|
| `list_docs()` | `["api.md", "changelog.md", "troubleshooting.md"]` |
| `read_doc("api.md")` | Contenido completo del archivo |
| `read_doc("../../../etc/passwd")` | Error: rechaza path traversal |
| `read_doc("noexiste.md")` | Error con lista de archivos disponibles |
| `search_docs("rate limit")` | Matches en troubleshooting.md Y changelog.md |
| `search_docs("ENDPOINT")` | Matches en api.md (case-insensitive) |
| `get_doc_sections("api.md")` | `["# API Reference", "## Authentication", "## Endpoints", ...]` |

**Pruebas de resources:**

| URI | Resultado esperado |
|---|---|
| `docs://index` | Índice markdown con nombres de archivos y sus secciones |
| `docs://files/api.md` | Contenido completo de api.md |
| `docs://files/noexiste.md` | Mensaje de error con archivos disponibles |

**Pruebas de prompts:**

| Prompt | Argumentos | Resultado esperado |
|---|---|---|
| `summarize_doc` | `filename=changelog.md` | Mensaje pidiendo resumir el changelog (con su contenido inyectado) |
| `troubleshoot` | `error_description=Tengo un error 429` | Mensaje combinando la descripción del error con la guía de troubleshooting |
| `compare_versions` | `version=v2.3.0` | Mensaje pidiendo analizar los cambios de v2.3.0 |

### Criterio de éxito

1. Las 4 tools aparecen en el Inspector con docstrings completos
2. Los 2 resources aparecen en el panel Resources (1 estático + 1 template)
3. Los 3 prompts aparecen en el panel Prompts con sus argumentos listados
4. `list_docs()` devuelve exactamente los 3 archivos markdown, ordenados
5. `search_docs("rate limit")` encuentra matches en al menos 2 archivos distintos
6. `read_doc` rechaza correctamente paths sospechosos sin crashear
7. El resource `docs://index` devuelve un índice que incluye los 3 archivos
8. El prompt `summarize_doc` con `filename=api.md` devuelve un mensaje que incluye el contenido del archivo
9. Ninguna tool usa `print()` a stdout (stdout está reservado para el protocolo MCP en stdio)

### Reflexión

1. **Sobre path traversal:** ¿Por qué necesitamos validar los filenames que recibimos? El LLM es "nuestro" — ¿por qué desconfiar de sus inputs? Piensa en el principio de defensa en profundidad.

2. **Sobre stdout vs stderr:** ¿Por qué un `print("debug info")` en un servidor MCP con stdio rompe todo? ¿Qué está pasando en el canal de stdout que no debes interrumpir?

3. **Tool vs resource vs prompt:** Si quieres exponer el contenido de un archivo, ¿cuándo usarías una tool (`read_doc`), cuándo un resource (`docs://files/{filename}`), y cuándo un prompt (`summarize_doc`)? Piensa en quién decide cuándo se accede a cada uno.

### Bonus

- Agrega una tool `count_docs_stats()` que devuelva estadísticas: `{"total_docs": N, "total_lines": M, "total_words": K}`
- Haz que `search_docs` soporte un parámetro opcional `filename` para buscar solo en un archivo específico
- Agrega un resource `docs://stats` que devuelva las estadísticas de la carpeta docs en formato JSON

---

## Práctica 3 — Conecta tu servidor a un cliente (45 minutos)

Hasta ahora probaste tus servidores con el Inspector — una herramienta de debugging. Ahora vas a conectarlos a un **cliente real** que use un LLM para invocar las tools automáticamente.

Tienes dos opciones. **Elige la que prefieras** (o haz las dos si te sobra tiempo):

- **Opción A: Claude Desktop** — Conectas tu servidor a la app de escritorio de Claude. Cero código de cliente: solo configuración. Ideal si tienes Claude Desktop instalado y quieres ver la integración más rápido.
- **Opción B: Cliente CLI custom** — Escribes tu propio cliente en Python que conecta con tu servidor vía MCP y usa Groq como LLM. Más trabajo, pero entiendes exactamente qué hace Claude Desktop por dentro.

---

### Opción A — Conectar a Claude Desktop

#### Contexto

Claude Desktop actúa como cliente MCP. Cuando le configuras un servidor, Claude puede ver las tools disponibles y decidir cuándo invocarlas durante la conversación. Tú no necesitas escribir ningún código de cliente — Claude Desktop maneja todo el loop agéntico internamente.

#### Paso 1: Localiza el archivo de configuración

Claude Desktop busca sus servidores MCP en un archivo JSON. La ubicación depende de tu sistema operativo:

| OS | Ruta del archivo |
|---|---|
| **macOS** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json` |
| **Linux** | `~/.config/Claude/claude_desktop_config.json` |

Si el archivo no existe, créalo. Si ya existe (porque tienes otros servidores configurados), solo agrega la entrada dentro del objeto `mcpServers`.

#### Paso 2: Configura tu servidor

Abre el archivo de configuración y agrega tu servidor. Necesitas la **ruta absoluta** a tu Python del venv y a tu archivo `server.py`:

```json
{
  "mcpServers": {
    "hola-mcp": {
      "command": "/ruta/absoluta/a/tu/venv/bin/python",
      "args": ["/ruta/absoluta/a/tu/mcp-clase13/server.py"]
    }
  }
}
```

**Ejemplo concreto en macOS/Linux:**

```json
{
  "mcpServers": {
    "hola-mcp": {
      "command": "/home/tu_usuario/mcp-clase13/venv/bin/python",
      "args": ["/home/tu_usuario/mcp-clase13/server.py"]
    }
  }
}
```

**Ejemplo concreto en Windows:**

```json
{
  "mcpServers": {
    "hola-mcp": {
      "command": "C:\\Users\\tu_usuario\\mcp-clase13\\venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\tu_usuario\\mcp-clase13\\server.py"]
    }
  }
}
```

**Puntos críticos:**

- Usa **rutas absolutas** siempre. Rutas relativas no funcionan porque Claude Desktop no sabe cuál es tu directorio de trabajo.
- Apunta al Python **dentro del venv**, no al Python del sistema. Así Claude Desktop usa el entorno donde tienes `mcp` instalado.
- El `command` es el ejecutable de Python y `args` es una lista con la ruta al script.

#### Paso 3: Reinicia Claude Desktop

Cierra Claude Desktop completamente y vuélvelo a abrir. Al arrancar, Claude lee el archivo de configuración e intenta conectarse a cada servidor listado.

Para verificar que la conexión fue exitosa, abre una nueva conversación en Claude Desktop. Deberías ver un icono de herramientas (o un indicador de MCP) en la interfaz que muestra tus tools disponibles. Si no lo ves, revisa la sección de debugging al final.

#### Paso 4: Prueba las tools

Escribe estos mensajes en Claude Desktop y observa cómo invoca las tools:

**Prueba 1 — Tool de hora:**
```
¿Qué hora es en Tokio y en Ciudad de México?
```
Claude debería invocar `get_current_time` dos veces con diferentes timezones y presentarte los resultados comparados.

**Prueba 2 — Tool de calculadora:**
```
Si compro 15 artículos a $47.50 cada uno, ¿cuánto pago en total? ¿Y con 16% de IVA?
```
Claude debería invocar `calculate` con las expresiones necesarias.

**Prueba 3 — Ambas tools:**
```
¿Cuántas horas de diferencia hay entre la hora actual en Nueva York y la de Tokio?
```
Claude debería usar `get_current_time` para ambas ciudades y `calculate` para la diferencia.

#### Paso 5 (bonus): Agrega el DocsServer

Si quieres conectar también tu `docs_server.py`, agrega otra entrada al JSON:

```json
{
  "mcpServers": {
    "hola-mcp": {
      "command": "/ruta/absoluta/venv/bin/python",
      "args": ["/ruta/absoluta/mcp-clase13/server.py"]
    },
    "docs-server": {
      "command": "/ruta/absoluta/venv/bin/python",
      "args": ["/ruta/absoluta/mcp-clase13/docs_server.py"]
    }
  }
}
```

Reinicia Claude Desktop. Ahora Claude tiene acceso a las tools de ambos servidores. Prueba:

```
Busca en mi documentación qué hacer cuando recibo un error 429
```

Claude debería invocar `search_docs("429")` del DocsServer y darte la respuesta con el contexto de tu documentación.

#### Criterio de éxito

1. Claude Desktop muestra las tools de tu servidor en la interfaz
2. Claude invoca `get_current_time` correctamente cuando le preguntas la hora
3. Claude invoca `calculate` cuando le pides un cálculo
4. Claude combina ambas tools en una sola conversación cuando es necesario
5. (Bonus) Claude usa las tools del DocsServer para responder preguntas sobre la documentación

#### Debugging de Claude Desktop

**"Claude Desktop no muestra mis tools"**
- Verifica que las rutas en el JSON son absolutas y correctas
- Verifica que apuntas al Python del venv (no al del sistema)
- Abre una terminal y ejecuta manualmente el comando: `/ruta/a/python /ruta/a/server.py` — si crashea, el problema está en tu servidor
- Revisa los logs de Claude Desktop:
  - macOS: `~/Library/Logs/Claude/`
  - Windows: `%APPDATA%\Claude\logs\`

**"Claude ve las tools pero nunca las usa"**
- Tus docstrings probablemente no son claros. Revisa que describan **cuándo** usar la tool
- Prueba con un mensaje más directo: "Usa la tool get_current_time para decirme la hora en UTC"

**"Claude Desktop crashea al arrancar"**
- Tu JSON probablemente tiene un error de sintaxis. Valídalo en jsonlint.com
- Errores comunes: coma después del último elemento, comillas faltantes, backslashes sin escapar en Windows (usa `\\` o `/`)

---

### Opción B — Cliente CLI custom con Groq

#### Contexto

Ahora vas a construir lo que Claude Desktop hace internamente — pero con tu propio código y usando Groq como LLM en vez de Claude. Al terminar vas a entender exactamente qué hacen Claude Desktop, Cursor, Continue y cualquier otro cliente MCP por dentro. Spoiler: no es magia, son ~150 líneas de Python.

Al usar Groq, vas a ver que MCP es verdaderamente **agnóstico al modelo**. Si funciona con Groq, funciona con cualquier API compatible con OpenAI.

#### Lo que vas a construir

Un archivo `client.py` que:

1. Arranca tu `server.py` (o `docs_server.py`) como subproceso por stdio
2. Inicializa la sesión MCP y lista las tools disponibles
3. Convierte las tools del formato MCP al formato de function calling de OpenAI
4. Lee una pregunta del usuario por la terminal
5. Envía pregunta + tools a Groq (modelo `llama-3.3-70b-versatile`)
6. Cuando Groq devuelve `tool_calls`, invoca las tools correspondientes vía la sesión MCP y devuelve los resultados al modelo
7. Repite el loop hasta que el modelo devuelva una respuesta final sin más `tool_calls`
8. Imprime la respuesta

#### El loop agéntico en 5 pasos

Este es el patrón central de todo agente LLM con tools. Asegúrate de que lo entiendes antes de escribir código:

```
1. El usuario envía una pregunta
2. El cliente envía (pregunta + lista de tools) al LLM
3. El LLM responde con UNA de dos cosas:
   (a) una respuesta de texto final → termina el loop
   (b) una o más tool_calls → el cliente ejecuta las tools,
       agrega los resultados al historial como mensajes role="tool",
       y vuelve al paso 2
4. Repite hasta respuesta final o límite de iteraciones
5. Imprime la respuesta final al usuario
```

El secreto: cada iteración del loop es una llamada más al LLM, con un historial de mensajes que crece. El mensaje `system` + `user` + `assistant` (con tool_calls) + `tool` (resultados) + `assistant` (final). El LLM lo ve todo como contexto.

#### Estructura inicial

Crea `client.py` con este esqueleto:

```python
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
    """
    Convierte una tool de MCP al formato que espera la API de OpenAI/Groq.

    TODO: implementa la conversión (ver sección "Los tres detalles" abajo)
    """
    pass


async def run_agent(session: ClientSession, user_question: str) -> str:
    """
    Ejecuta el loop agéntico hasta que el LLM devuelva una respuesta final.
    """
    # 1. Obtener tools del servidor MCP y convertirlas
    tools_result = await session.list_tools()
    openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]

    # 2. Inicializar historial de mensajes
    messages = [
        {"role": "system", "content": "Eres un asistente técnico que responde preguntas usando las tools disponibles. Sé conciso y preciso."},
        {"role": "user", "content": user_question},
    ]

    # 3. Loop agéntico
    max_iterations = 10
    for iteration in range(max_iterations):
        # TODO: llamar a Groq con messages + tools
        # TODO: obtener assistant_message de la respuesta
        # TODO: si NO hay tool_calls → devolver el contenido de texto
        # TODO: si HAY tool_calls:
        #       - agregar el assistant message al historial
        #       - por cada tool_call:
        #           - parsear arguments con json.loads
        #           - llamar session.call_tool
        #           - agregar resultado como message role="tool"
        #       - continuar el loop
        pass

    return f"El agente no pudo resolver la pregunta en {max_iterations} pasos."


async def main():
    # Parámetros para arrancar el servidor como subproceso
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],  # cambia a "docs_server.py" si quieres probar con ese
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Cliente MCP conectado. Escribe 'salir' para terminar.\n")
            while True:
                try:
                    question = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if question.lower() in ("salir", "exit", "quit"):
                    break
                if not question:
                    continue

                try:
                    answer = await run_agent(session, question)
                    print(f"\n{answer}\n")
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
```

#### Los tres detalles donde la gente se traba

**1. La conversión de tools MCP a formato OpenAI.**

El objeto `tool` de MCP tiene `name`, `description` e `inputSchema` (que ya es JSON Schema). OpenAI espera este dict:

```python
{
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema,  # directamente, ya es JSON Schema
    }
}
```

**2. Extraer el resultado de `session.call_tool`.**

Devuelve un objeto con una lista de `content` blocks. Para tools de texto, el contenido está en `block.text`:

```python
result = await session.call_tool(tool_name, arguments=tool_args)
result_text = "\n".join(
    block.text for block in result.content if hasattr(block, "text")
)
```

**3. El `tool_call_id` en el historial de mensajes.**

Cuando el LLM devuelve tool_calls, cada uno tiene un `id`. Cuando devuelves el resultado, **tienes que referenciar ese mismo id**:

```python
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,  # MISMO id que envió el modelo
    "content": result_text,
})
```

Si te equivocas en este id, el modelo se confunde y el loop falla. Además, el message del assistant con los `tool_calls` debe ir en el historial **ANTES** de los messages `role=tool`. Si el orden está mal, el modelo no sabe qué tools ya ejecutó.

#### Pruebas del criterio de éxito

Con `python client.py` corriendo (apuntando a `server.py`), prueba:

**Prueba 1 — Tool de hora:**
```
> ¿Qué hora es en Ciudad de México?
```
Esperado: el cliente llama `get_current_time("America/Mexico_City")` y el modelo responde con la hora.

**Prueba 2 — Tool de calculadora:**
```
> ¿Cuánto es 2 elevado a la 16?
```
Esperado: el cliente llama `calculate("2 ** 16")` y el modelo responde con 65536.

**Prueba 3 — Ambas tools:**
```
> ¿Qué hora es en UTC y cuánto es 365 * 24?
```
Esperado: el modelo invoca ambas tools y combina los resultados.

Ahora cambia `args=["server.py"]` por `args=["docs_server.py"]` y reinicia. Prueba:

**Prueba 4 — DocsServer:**
```
> ¿Qué archivos de documentación tengo?
```
Esperado: el cliente llama `list_docs()` y responde con la lista.

**Prueba 5 — Búsqueda en docs:**
```
> Tengo un error 429, ¿qué hago?
```
Esperado: el cliente llama `search_docs("429")` o `search_docs("rate limit")` y el modelo responde con la solución.

#### Criterio de éxito

1. El cliente se conecta al servidor sin errores
2. El LLM invoca las tools correctas según la pregunta
3. Los resultados de las tools se inyectan en el historial y el LLM los usa para responder
4. El loop termina cuando el LLM da una respuesta final (no se queda en un ciclo infinito)
5. El cliente funciona tanto con `server.py` como con `docs_server.py`

#### Bonus

- **Consume resources en el system prompt.** Al inicio de `run_agent`, lee el resource `docs://index` con `session.read_resource("docs://index")` e inyecta su contenido en el system message. Así el LLM sabe qué docs existen sin gastar una tool call.

- **Agrega comandos para prompts.** Implementa comandos como `/prompt summarize_doc filename=api.md` que pidan un prompt al servidor con `session.get_prompt(name, arguments)`, inyecten los mensajes en la conversación, y envíen todo al LLM.

- **Multi-servidor.** Conecta tu cliente simultáneamente al `docs_server.py` Y al `server.py`. El modelo debería poder combinar tools de ambos. Pista: mantén dos sesiones MCP y un dict que mapee cada tool name a la sesión correcta.

- **Logging verboso.** Imprime cada paso del loop con colores: qué tool invoca el modelo, con qué argumentos, qué devolvió. Usa `\033[33m` para amarillo (tools) y `\033[34m` para azul (resultados).

---

### Reflexión (ambas opciones)

Responde estas preguntas sin importar qué opción elegiste:

1. **Tool vs resource:** En la Práctica 2 implementaste `read_doc` (tool) y `docs://files/{filename}` (resource), que hacen esencialmente lo mismo. ¿Cuál es la diferencia conceptual? ¿Quién decide cuándo se usa cada uno?

2. **Prompts como plantillas:** ¿En qué se diferencia un prompt MCP de simplemente hardcodear un system message en el cliente? ¿Qué ganas al mover la plantilla al servidor?

3. **Sobre agnosticismo de modelo:** Si elegiste la Opción A, ¿qué tendrías que cambiar para usar Groq en vez de Claude? Si elegiste la Opción B, ¿qué tendrías que cambiar para usar Claude en vez de Groq? ¿Qué parte del código (o configuración) es agnóstica al modelo y qué parte no?

4. **Sobre el protocolo:** Claude Desktop y tu cliente CLI (si lo escribiste) hacen esencialmente lo mismo. ¿Qué pasos del proceso son idénticos en ambos? ¿Dónde está el valor de que MCP sea un estándar?

---

## Entregables

Para la próxima clase, sube tu trabajo a un repo de GitHub con esta estructura:

```
mcp-clase13/
├── .env.example          # SIN tu API key, solo los nombres de las variables
├── .gitignore            # debe incluir .env y venv/
├── requirements.txt      # pip freeze > requirements.txt
├── server.py             # Práctica 1
├── docs_server.py        # Práctica 2
├── client.py             # Práctica 3 Opción B (si la elegiste)
├── docs/
│   ├── api.md
│   ├── troubleshooting.md
│   └── changelog.md
└── README.md             # instrucciones para correr tu proyecto
```

Si elegiste la Opción A, incluye en tu README:
- Tu `claude_desktop_config.json` (con rutas anonimizadas, no tu path real)
- Un screenshot o descripción de Claude Desktop invocando tus tools

Tu README debe incluir:
- Cómo instalar dependencias (`pip install -r requirements.txt`)
- Cómo correr cada práctica (comandos exactos)
- Los comandos para abrir el Inspector
- Una sección **"¿Qué aprendí?"** — 3-5 frases con el insight más importante que te llevaste de cada práctica

---

## Tips de debugging

**"El Inspector no encuentra mi servidor"**
Casi siempre es un error en la ruta del comando. Prueba con rutas absolutas. En Windows, la ruta a Python típicamente es `C:\Users\<tu_usuario>\...\python.exe`. Verifica que estás en el directorio correcto.

**"El Inspector abre pero no lista mis tools"**
Revisa que tu servidor no haga `print()` a stdout sin `file=sys.stderr`. En un servidor MCP con stdio, stdout está reservado para el protocolo. Un print que se va a stdout rompe el protocolo silenciosamente. Usa `print(..., file=sys.stderr)` para logs de debug.

**"Claude Desktop no muestra mis tools"**
Verifica rutas absolutas en el JSON, que apuntes al Python del venv, y que el JSON sea válido (sin comas extra, comillas correctas). Revisa los logs de Claude Desktop para errores de conexión.

**"El LLM nunca llama mis tools"**
Tu docstring no está claro. Reescríbelo pensando: ¿qué palabras exactas usaría un usuario que necesita esta tool? Incluye esas palabras en el docstring. Además verifica que el modelo de Groq soporta tool calling — `llama-3.3-70b-versatile` sí lo soporta, `llama-3.1-8b-instant` NO.

**"Groq me devuelve un error 429"**
Pasaste el rate limit del free tier. Espera un minuto y reintenta. Para desarrollo, el límite suele ser suficiente si no haces stress testing.

**"Mi tool crashea y el cliente se cuelga"**
Envuelve la lógica de cada tool en un try/except y devuelve errores como strings, no como excepciones. El LLM puede recuperarse de un string de error; no puede recuperarse de una excepción que mata el proceso.

**"El loop agéntico nunca termina"**
Verifica que estás agregando el message del assistant con los `tool_calls` al historial ANTES de los messages de `role=tool`. Si el orden está mal, el modelo no sabe qué tools ya ejecutó y vuelve a pedirlas. Agrega un `max_iterations` como safety net.

**"Los resources o prompts no aparecen en el Inspector"**
Verifica que estás usando los decoradores correctos: `@mcp.resource("uri://...")` para resources y `@mcp.prompt()` para prompts. El Inspector tiene paneles separados para Tools, Resources y Prompts — revisa que estés mirando el panel correcto.

---

## Recursos de referencia

- **MCP Python SDK:** github.com/modelcontextprotocol/python-sdk
- **Especificación del protocolo:** modelcontextprotocol.io
- **Configuración de Claude Desktop:** modelcontextprotocol.io/quickstart/user — guía oficial para conectar servidores
- **Inspiración:** github.com/daveebbelaar/ai-cookbook/tree/main/mcp/crash-course
- **Servidores oficiales de referencia:** github.com/modelcontextprotocol/servers — filesystem, SQLite, GitHub y más, todos open source
- **Groq API docs:** console.groq.com/docs — lista de modelos con soporte de tool calling
- **MCP Inspector:** github.com/modelcontextprotocol/inspector
