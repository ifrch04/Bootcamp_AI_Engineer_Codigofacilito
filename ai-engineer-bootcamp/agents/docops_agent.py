"""Agente DocOps — pipeline secuencial para consultas sobre documentos.

Implementa un agente simple basado en pipeline lineal:
    Retrieve → Process → Generate

Cada paso es una función decorada con ``@pipeline_step`` del módulo
de orquestación. El agente conecta las herramientas del ``ToolRegistry``
con un LLM (Groq vía OpenAI API) para responder preguntas usando
contexto recuperado de documentos reales indexados en ChromaDB.

Este es un agente "clásico" sin ciclos de razonamiento — ejecuta los
pasos en orden fijo y produce una respuesta.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from orchestration.pipelines import Pipeline, PipelineResult, pipeline_step
from orchestration.tools import (
    ToolDefinition,
    ToolRegistry,
    get_current_datetime,
)
from rag.vectorstore import create_vectorstore, search, SearchResult

load_dotenv()

logger = logging.getLogger("agents.docops")

# ---------------------------------------------------------------------------
# Colores ANSI
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"
BLUE = "\033[94m"
WHITE = "\033[97m"

# ---------------------------------------------------------------------------
# Configuración LLM (Groq vía OpenAI-compatible API)
# ---------------------------------------------------------------------------

_GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_SYSTEM_PROMPT = (
    "Eres un asistente de documentación técnica. "
    "Responde basándote ÚNICAMENTE en el contexto proporcionado. "
    "Si no encuentras la respuesta en el contexto, di 'No tengo "
    "información suficiente para responder esa pregunta.' "
    "Responde en español de forma clara y concisa."
)


def _print_header(title: str, color: str = CYAN) -> None:
    print(f"\n{color}{BOLD}{'=' * 70}{RESET}")
    print(f"{color}{BOLD}  {title}{RESET}")
    print(f"{color}{BOLD}{'=' * 70}{RESET}")


def _print_step(step_name: str, color: str, msg: str) -> None:
    print(f"{color}{BOLD}[{step_name}]{RESET} {msg}")


def _print_metric(label: str, value: str, color: str = DIM) -> None:
    print(f"  {color}{label}: {RESET}{value}")


# ---------------------------------------------------------------------------
# Resultado del agente
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Resultado de la ejecución del agente DocOps."""

    query: str
    context: str = ""
    answer: str = ""
    pipeline_result: PipelineResult | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieved_chunks: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.pipeline_result is not None and self.pipeline_result.success

    def summary(self) -> str:
        status = f"{GREEN}SUCCESS{RESET}" if self.success else f"{RED}FAILED{RESET}"
        total_tokens = self.prompt_tokens + self.completion_tokens
        duration = self.pipeline_result.total_duration if self.pipeline_result else 0.0

        lines = [
            f"\n{WHITE}{BOLD}{'─' * 70}{RESET}",
            f"{WHITE}{BOLD}  RESUMEN DEL AGENTE{RESET}",
            f"{WHITE}{BOLD}{'─' * 70}{RESET}",
            f"  Status:        {status}",
            f"  Query:         {self.query}",
            f"  Chunks usados: {len(self.retrieved_chunks)}",
            f"  Contexto:      {len(self.context)} chars",
            f"  Tokens:        {YELLOW}{total_tokens}{RESET} (prompt={self.prompt_tokens}, completion={self.completion_tokens})",
            f"  Duración:      {YELLOW}{duration:.2f}s{RESET}",
        ]

        if self.pipeline_result:
            lines.append(f"\n  {DIM}Desglose por paso:{RESET}")
            step_names = ["RETRIEVE", "PROCESS", "GENERATE"]
            step_colors = [CYAN, MAGENTA, GREEN]
            for i, step in enumerate(self.pipeline_result.steps):
                name = step_names[i] if i < len(step_names) else f"STEP {i+1}"
                color = step_colors[i] if i < len(step_colors) else DIM
                mark = f"{GREEN}OK{RESET}" if step.success else f"{RED}FAIL{RESET}"
                lines.append(
                    f"    {color}{BOLD}[{name}]{RESET} {mark} — {step.duration_seconds:.3f}s"
                )
                if step.error:
                    lines.append(f"      {RED}Error: {step.error}{RESET}")

        if self.answer:
            lines.append(f"\n  {GREEN}{BOLD}Respuesta:{RESET}")
            lines.append(f"  {self.answer}")

        lines.append(f"{WHITE}{BOLD}{'─' * 70}{RESET}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agente DocOps
# ---------------------------------------------------------------------------


class DocOpsAgent:
    """Agente secuencial para consultas sobre documentos.

    Ejecuta un pipeline de 3 pasos:
        1. **Retrieve**: busca documentos reales en ChromaDB via ToolRegistry.
        2. **Process**: formatea el contexto recuperado con metadata.
        3. **Generate**: envía el contexto + pregunta al LLM.

    Args:
        registry: Registro de herramientas. Si no se provee, construye uno
            conectado a la colección ChromaDB ``novatech_docs``.
        model: Modelo de Groq a utilizar.
        temperature: Temperatura para la generación.
        system_prompt: Prompt del sistema para el LLM.
        collection_name: Nombre de la colección en ChromaDB.
        chroma_dir: Directorio de persistencia de ChromaDB.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        system_prompt: str | None = None,
        collection_name: str = "novatech_docs",
        chroma_dir: str = "./chroma_db",
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key, base_url=_GROQ_BASE_URL)
        self.model = model or _GROQ_MODEL
        self.temperature = temperature
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self.collection_name = collection_name
        self.chroma_dir = chroma_dir
        self.registry = registry or self._default_registry()

    def run(self, query: str) -> AgentResult:
        """Ejecuta el pipeline secuencial para responder la consulta.

        Args:
            query: Pregunta del usuario.

        Returns:
            ``AgentResult`` con contexto, respuesta y métricas.
        """
        agent_result = AgentResult(query=query)

        _print_header(f"QUERY: {query}")

        retrieve = self._make_retrieve_step(agent_result)
        process = self._make_process_step()
        generate = self._make_generate_step(query, agent_result)

        pipe = Pipeline(
            name="docops",
            steps=[retrieve, process, generate],
        )

        pipeline_result = pipe.run(query)
        agent_result.pipeline_result = pipeline_result

        if pipeline_result.success:
            agent_result.answer = pipeline_result.final_output
        else:
            agent_result.answer = "Error: el pipeline no completó todos los pasos."

        # Imprimir respuesta final
        print(f"\n{GREEN}{BOLD}  RESPUESTA:{RESET}")
        print(f"{GREEN}  {agent_result.answer}{RESET}")

        return agent_result

    def _make_retrieve_step(self, agent_result: AgentResult):
        """Crea el paso de recuperación de documentos reales."""
        registry = self.registry

        @pipeline_step(name="retrieve", max_retries=2, timeout_seconds=10)
        def retrieve(query: str) -> list[dict]:
            _print_step("RETRIEVE", CYAN, f"Buscando documentos para: '{query}'")
            t0 = time.time()

            raw = registry.execute_tool(
                "search_documents", {"query": query, "top_k": 3}
            )

            elapsed = time.time() - t0
            _print_metric("Tiempo de búsqueda", f"{elapsed:.3f}s", CYAN)

            # Parsear los resultados (vienen como string serializado del tool)
            # La herramienta real retorna una lista de dicts
            if isinstance(raw, list):
                chunks = raw
            else:
                # Si viene como string (del execute_tool), intentar evaluar
                try:
                    import ast
                    chunks = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    chunks = [{"content": raw, "source": "raw", "score": 0.0}]

            agent_result.retrieved_chunks = chunks
            _print_metric("Chunks recuperados", str(len(chunks)), CYAN)

            for i, chunk in enumerate(chunks):
                source = os.path.basename(chunk.get("source", "?"))
                score = chunk.get("score", 0.0)
                preview = chunk.get("content", "")[:100].replace("\n", " ")
                print(
                    f"    {CYAN}{i+1}. [{score:.3f}] {BOLD}{source}{RESET}"
                    f" {DIM}{preview}...{RESET}"
                )

            return chunks

        return retrieve

    def _make_process_step(self):
        """Crea el paso de procesamiento del contexto."""

        @pipeline_step(name="process", max_retries=1, timeout_seconds=5)
        def process(chunks: list[dict]) -> str:
            _print_step("PROCESS", MAGENTA, f"Formateando {len(chunks)} chunks como contexto")

            context_parts = []
            total_chars = 0

            for i, chunk in enumerate(chunks):
                source = os.path.basename(chunk.get("source", "desconocido"))
                content = chunk.get("content", "")
                block = f"[Fuente: {source}]\n{content}"
                context_parts.append(block)
                total_chars += len(content)

            context = "\n\n---\n\n".join(context_parts)

            # Agregar timestamp
            try:
                timestamp = self.registry.execute_tool("get_current_datetime", {})
                context += f"\n\n[Consulta realizada: {timestamp}]"
            except Exception:
                pass

            _print_metric("Caracteres de contexto", str(total_chars), MAGENTA)
            _print_metric("Fuentes utilizadas",
                          ", ".join(os.path.basename(c.get("source", "?")) for c in chunks),
                          MAGENTA)

            return context

        return process

    def _make_generate_step(self, query: str, agent_result: AgentResult):
        """Crea el paso de generación con el LLM."""
        client = self.client
        model = self.model
        temperature = self.temperature
        system_prompt = self.system_prompt

        @pipeline_step(name="generate", max_retries=2, timeout_seconds=30)
        def generate(context: str) -> str:
            agent_result.context = context

            _print_step("GENERATE", GREEN, f"Enviando al LLM ({model})")
            _print_metric("Temperatura", str(temperature), GREEN)
            _print_metric("Contexto enviado", f"{len(context)} chars", GREEN)

            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"CONTEXTO:\n{context}\n\n"
                            f"PREGUNTA: {query}"
                        ),
                    },
                ],
                temperature=temperature,
                max_tokens=1024,
            )
            llm_elapsed = time.time() - t0

            usage = response.usage
            prompt_tokens = 0
            completion_tokens = 0
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                agent_result.prompt_tokens = prompt_tokens
                agent_result.completion_tokens = completion_tokens

            answer = response.choices[0].message.content or ""

            _print_metric("Latencia LLM", f"{llm_elapsed:.2f}s", GREEN)
            _print_metric("Prompt tokens", str(prompt_tokens), GREEN)
            _print_metric("Completion tokens", str(completion_tokens), GREEN)
            _print_metric("Total tokens", str(prompt_tokens + completion_tokens), GREEN)

            return answer

        return generate

    def _default_registry(self) -> ToolRegistry:
        """Crea un registro con herramientas conectadas a ChromaDB real."""
        registry = ToolRegistry()
        collection = create_vectorstore(self.collection_name, self.chroma_dir)

        def real_search(query: str, top_k: int = 3) -> list[dict]:
            """Busca en ChromaDB y retorna chunks reales."""
            results: list[SearchResult] = search(collection, query, n_results=top_k)
            return [
                {
                    "content": r.content,
                    "source": r.metadata.get("source", "desconocido"),
                    "score": r.score,
                    "chunk_id": r.chunk_id,
                }
                for r in results
            ]

        registry.register(ToolDefinition(
            name="search_documents",
            description="Busca documentos relevantes en la base de conocimiento.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de búsqueda",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de resultados (default: 3)",
                    },
                },
                "required": ["query"],
            },
            function=real_search,
        ))

        registry.register(ToolDefinition(
            name="get_current_datetime",
            description="Obtiene la fecha y hora actual.",
            parameters={"type": "object", "properties": {}},
            function=get_current_datetime,
        ))

        return registry


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    agent = DocOpsAgent()

    queries = [
        "¿Cuál es el horario de trabajo?",
        "¿Cuántos días de vacaciones corresponden el primer año?",
        "¿Qué equipo de cómputo reciben los desarrolladores?",
    ]

    for q in queries:
        result = agent.run(q)
        print(result.summary())
