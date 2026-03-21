"""
DocOps Agent — Sistema multiagente con LangGraph
Clase 11: Orquestación supervisor-workers con bucle de calidad

Usa GPT-OSS-120B vía Groq (OpenAI-compatible endpoint)
"""

import logging
import os
import operator
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, START, END

# RAG pipeline (clases 5-7)
from rag.vectorstore import create_vectorstore, search as vector_search, SearchResult
from rag.ingestion import load_directory, chunk_by_paragraphs, Chunk
from rag.retrieval import HybridRetriever, rerank

logger = logging.getLogger(__name__)

load_dotenv()

# ─── Colores ANSI ───────────────────────────────────────────
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_WHITE = "\033[97m"

# ─── Metadata de agentes (color, icono, descripción) ────────
AGENT_META = {
    "planner": {
        "color": _BLUE,
        "icon": "📋",
        "label": "PLANNER",
        "desc": "Analiza la consulta y genera un plan de acción estructurado",
    },
    "retriever": {
        "color": _CYAN,
        "icon": "🔍",
        "label": "RETRIEVER",
        "desc": "Busca documentos relevantes en el vector store según el plan",
    },
    "executor": {
        "color": _YELLOW,
        "icon": "⚙️",
        "label": "EXECUTOR",
        "desc": "Genera la respuesta basándose en el plan y los documentos",
    },
    "verifier": {
        "color": _MAGENTA,
        "icon": "✅",
        "label": "VERIFIER",
        "desc": "Evalúa la calidad de la respuesta y decide si aceptar o revisar",
    },
}

# ─── LLM via Groq (OpenAI-compatible) ───────────────────────
llm = ChatOpenAI(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    temperature=0,
    max_tokens=2048,
)

# LLM de respaldo (modelo más pequeño/económico en Groq)
fallback_llm = ChatOpenAI(
    model=os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    temperature=0,
    max_tokens=1024,
)


# ─── ESTADO COMPARTIDO ──────────────────────────────────────
class DocOpsState(TypedDict):
    """Estado compartido entre todos los agentes del sistema."""
    messages: Annotated[list[AnyMessage], operator.add]
    plan: str  #Resultado del Planner
    search_results: str  # Resultado del Retriever
    draft: str  # Resultado del Executor
    feedback: str  # Resultado del Verifier
    quality_score: float # Resultado del Verifier
    iteration: int


# ─── CONTRATOS (Structured Output) ──────────────────────────
class QualityCheck(BaseModel):
    """Contrato de salida del Verifier."""
    score: float = Field(
        description="Score de calidad de 0.0 a 1.0. "
                    "1.0 = respuesta perfecta, 0.0 = inaceptable."
    )
    feedback: str = Field(
        description="Retroalimentación específica si el score es menor a 0.8. "
                    "Indica qué mejorar concretamente."
    )
    decision: Literal["accept", "revise"] = Field(
        description="'accept' si score >= 0.8, 'revise' si necesita mejora."
    )


# ─── AGENTE 1: PLANNER ──────────────────────────────────────
def planner_agent(state: DocOpsState) -> dict:
    """
    Analiza la consulta del usuario y genera un plan estructurado.
    No busca información — solo planifica los pasos a seguir.
    """
    user_query = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=(
            "Eres un planificador experto para un sistema de consulta de documentos. "
            "Tu trabajo es analizar la consulta del usuario y generar un plan claro "
            "con los pasos necesarios para responderla.\n\n"
            "Reglas:\n"
            "- Identifica qué información necesitas buscar\n"
            "- Define los criterios de una buena respuesta\n"
            "- Sé específico sobre qué buscar en los documentos\n"
            "- Responde SOLO con el plan, sin ejecutar ningún paso"
        )),
        HumanMessage(content=f"Genera un plan para responder: {user_query}")
    ])

    return {"plan": response.content, "iteration": 0}


# ─── RAG: carga de colección e índice ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
_DATA_DIR = str(PROJECT_ROOT / "data")
_COLLECTION_NAME = "docops_multiagent"

_collection = None
_chunks: list[Chunk] = []


def _get_rag_resources():
    """Inicializa (lazy) la colección ChromaDB y los chunks para el retriever."""
    global _collection, _chunks

    if _collection is not None:
        return _collection, _chunks

    # 1. Abrir o crear la colección
    _collection = create_vectorstore(_COLLECTION_NAME, _CHROMA_DIR)

    # 2. Si la colección está vacía, indexar los documentos de /data
    if _collection.count() == 0:
        docs = load_directory(_DATA_DIR)
        if docs:
            from rag.vectorstore import index_chunks
            all_chunks: list[Chunk] = []
            for doc in docs:
                all_chunks.extend(chunk_by_paragraphs(doc, max_chunk_size=500))
            index_chunks(_collection, all_chunks)
            _chunks = all_chunks
            logger.info(
                "Indexados %d chunks de %d documentos en '%s'",
                len(all_chunks), len(docs), _COLLECTION_NAME,
            )
        else:
            logger.warning("No se encontraron documentos en %s", _DATA_DIR)
    else:
        # Colección ya tiene datos — reconstruir chunks para BM25
        all_data = _collection.get(include=["documents", "metadatas"])
        _chunks = [
            Chunk(
                content=doc,
                metadata=meta,
                chunk_id=cid,
            )
            for cid, doc, meta in zip(
                all_data["ids"], all_data["documents"], all_data["metadatas"]
            )
        ]
        logger.info(
            "Colección '%s' cargada con %d chunks existentes",
            _COLLECTION_NAME, len(_chunks),
        )

    return _collection, _chunks


# ─── AGENTE 2: RETRIEVER ────────────────────────────────────
def retriever_agent(state: DocOpsState) -> dict:
    """
    Busca información relevante usando el pipeline RAG (clases 5-7).

    Pipeline:
      1. Búsqueda híbrida (BM25 + vector) vía HybridRetriever
      2. Reranking con cross-encoder
      3. Formato de contexto con fuentes

    Fallback: si el RAG falla, usa el LLM para simular resultados.
    """
    query = state["messages"][-1].content
    plan = state["plan"]
    search_query = f"{query} {plan[:200]}"

    try:
        collection, chunks = _get_rag_resources()

        if not chunks:
            raise ValueError("No hay chunks indexados en la colección")

        # Paso 1: Búsqueda híbrida (BM25 + vector, clase 6)
        hybrid = HybridRetriever(collection, chunks, alpha=0.5)
        results: list[SearchResult] = hybrid.search(search_query, top_k=10)

        if not results:
            raise ValueError("Búsqueda híbrida no retornó resultados")

        # Paso 2: Reranking con cross-encoder (clase 6)
        reranked = rerank(query, results, top_k=5)

        # Paso 3: Formatear contexto con fuentes
        context_parts = []
        for i, r in enumerate(reranked, 1):
            source = r.metadata.get("source", "desconocida")
            score = f"{r.score:.3f}"
            context_parts.append(
                f"[{i}] (fuente: {source} | score: {score})\n{r.content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            "Retriever: %d resultados tras reranking (query: %s)",
            len(reranked), query[:60],
        )

        return {"search_results": context}

    except Exception as e:
        # Fallback: búsqueda simulada con LLM
        logger.warning("RAG pipeline falló (%s), usando fallback LLM", e)

        response = llm.invoke([
            SystemMessage(content=(
                "Eres un agente de búsqueda. Dado el siguiente plan, "
                "genera información relevante que podría encontrarse en "
                "documentos empresariales internos. Simula resultados de búsqueda "
                "realistas y útiles.\n\n"
                "Formato: Presenta 3-5 fragmentos de documentos relevantes, "
                "cada uno con su fuente ficticia."
            )),
            HumanMessage(content=f"Plan de búsqueda:\n{plan}")
        ])

        return {
            "search_results": (
                f"[Fallback — resultados simulados por LLM]\n\n"
                f"{response.content}"
            )
        }


# ─── AGENTE 3: EXECUTOR (con fallback) ──────────────────────
def executor_agent(state: DocOpsState) -> dict:
    """
    Genera la respuesta usando el plan y los documentos recuperados.
    Incluye fallback a modelo más pequeño si el principal falla.
    """
    # Si hay feedback de una iteración anterior, incluirlo
    feedback_section = ""
    if state.get("feedback") and state["iteration"] > 0:
        feedback_section = (
            f"\n\nFEEDBACK DE REVISIÓN ANTERIOR (iteración {state['iteration']}):\n"
            f"{state['feedback']}\n"
            "Corrige los problemas señalados en el feedback."
        )

    prompt_content = (
        f"Plan:\n{state['plan']}\n\n"
        f"Documentos encontrados:\n{state['search_results']}\n\n"
        f"Consulta original: {state['messages'][-1].content}"
        f"{feedback_section}"
    )

    try:
        # Intento principal con modelo fuerte (GPT-OSS-120B)
        response = llm.invoke([
            SystemMessage(content=(
                "Eres un asistente experto que genera respuestas precisas "
                "basándose en documentos internos. Tu respuesta debe:\n"
                "- Ser fiel a la información de los documentos (no inventar)\n"
                "- Citar las fuentes cuando sea posible\n"
                "- Ser clara, estructurada y completa\n"
                "- Responder directamente a la consulta del usuario"
            )),
            HumanMessage(content=prompt_content)
        ])
        return {"draft": response.content}

    except Exception as e:
        # Fallback a modelo más económico (llama-3.3-70b)
        try:
            response = fallback_llm.invoke([
                SystemMessage(content=(
                    "Genera una respuesta concisa basada en el contexto."
                )),
                HumanMessage(content=prompt_content)
            ])
            return {
                "draft": (
                    f"[Generado con modelo de respaldo]\n\n"
                    f"{response.content}"
                )
            }
        except Exception as e2:
            # Último recurso: respuesta degradada
            return {
                "draft": (
                    f"No pude generar una respuesta completa. "
                    f"Error: {str(e2)[:200]}\n\n"
                    f"Documentos encontrados:\n"
                    f"{state['search_results'][:500]}"
                ),
                "quality_score": 0.3,
            }


# ─── AGENTE 4: VERIFIER ─────────────────────────────────────
def verifier_agent(state: DocOpsState) -> dict:
    """
    Evalúa la calidad del borrador y decide si aceptar o pedir revisión.
    Usa structured output para garantizar formato consistente.
    """
    quality_checker = llm.with_structured_output(QualityCheck)

    try:
        check = quality_checker.invoke([
            SystemMessage(content=(
                "Eres un verificador de calidad. Evalúa la respuesta generada "
                "contra los documentos fuente y la consulta original.\n\n"
                "Criterios de evaluación:\n"
                "- Fidelidad: ¿La respuesta es fiel a los documentos?\n"
                "- Completitud: ¿Responde toda la consulta?\n"
                "- Claridad: ¿Es clara y bien estructurada?\n"
                "- Relevancia: ¿Incluye solo información pertinente?\n\n"
                "Score:\n"
                "- 0.9-1.0: Excelente, aceptar\n"
                "- 0.8-0.89: Buena, aceptar\n"
                "- 0.6-0.79: Necesita mejora, revisar con feedback\n"
                "- <0.6: Mala, revisar con feedback detallado"
            )),
            HumanMessage(content=(
                f"CONSULTA: {state['messages'][-1].content}\n\n"
                f"DOCUMENTOS:\n{state['search_results']}\n\n"
                f"RESPUESTA A EVALUAR:\n{state['draft']}"
            ))
        ])

        return {
            "quality_score": check.score,
            "feedback": check.feedback,
            "iteration": state["iteration"] + 1,
        }

    except Exception as e:
        # Si el structured output falla, aceptar con score medio
        return {
            "quality_score": 0.7,
            "feedback": f"Verificación falló: {str(e)[:200]}",
            "iteration": state["iteration"] + 1,
        }


# ─── ARISTA CONDICIONAL: BUCLE DE CALIDAD ───────────────────
def should_revise(state: DocOpsState) -> Literal["accept", "revise"]:
    """
    Decide si la respuesta es aceptable o necesita revisión.

    Acepta si:
    - quality_score >= 0.8 (calidad suficiente)
    - iteration >= 3 (máximo de intentos alcanzado)

    Rechaza si:
    - quality_score < 0.8 AND iteration < 3
    """
    if state["quality_score"] >= 0.8:
        return "accept"
    if state["iteration"] >= 3:
        return "accept"
    return "revise"


# ─── CONSTRUCCIÓN DEL GRAFO ─────────────────────────────────
def build_docops_agent():
    """
    Construye el grafo multiagente del DocOps Agent.

    Flujo: START → planner → retriever → executor → verifier → (accept|revise)
    Bucle: verifier --revise--> executor (con feedback)
    Salida: verifier --accept--> END
    """
    workflow = StateGraph(DocOpsState)

    # Registrar nodos (agentes)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("verifier", verifier_agent)

    # Flujo principal (aristas directas)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "executor")
    workflow.add_edge("executor", "verifier")

    # Bucle de calidad (arista condicional)
    workflow.add_conditional_edges(
        "verifier",
        should_revise,
        {
            "accept": END,
            "revise": "executor",
        },
    )

    return workflow.compile()


# Instancia global del grafo multiagente compilado
docops_agent = build_docops_agent()


# ─── UTILIDADES ──────────────────────────────────────────────
def invoke_docops(query: str, verbose: bool = False) -> dict:
    """
    Invoca el sistema multiagente con una consulta.

    Args:
        query: Consulta del usuario en lenguaje natural
        verbose: Si True, imprime el estado de cada paso

    Returns:
        dict con keys: answer, quality_score, iterations, plan
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": "",
        "search_results": "",
        "draft": "",
        "feedback": "",
        "quality_score": 0.0,
        "iteration": 0,
    }

    if verbose:
        print(f"\n{_BOLD}{_WHITE}{'═'*60}{_RESET}")
        print(f"{_BOLD}{_WHITE}  🚀 DOCOPS MULTIAGENTE — INICIO DE EJECUCIÓN{_RESET}")
        print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}")
        print(f"{_DIM}  Consulta: {query}{_RESET}")

        step = 0
        for event in docops_agent.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                step += 1
                meta = AGENT_META.get(node_name, {})
                color = meta.get("color", _WHITE)
                icon = meta.get("icon", "▸")
                label = meta.get("label", node_name.upper())
                desc = meta.get("desc", "")

                print(f"\n{color}{_BOLD}{'─'*60}{_RESET}")
                print(f"{color}{_BOLD}  {icon}  [{step}] {label}{_RESET}")
                print(f"{color}{_DIM}  {desc}{_RESET}")
                print(f"{color}{'─'*60}{_RESET}")

                for key, value in node_output.items():
                    if key == "messages":
                        continue
                    preview = str(value)[:300]
                    # Colores especiales para campos clave
                    if key == "quality_score":
                        score = float(value)
                        score_color = _GREEN if score >= 0.8 else _RED
                        print(f"  {_DIM}↳ {key}:{_RESET} {score_color}{_BOLD}{preview}{_RESET}")
                    elif key == "feedback":
                        print(f"  {_DIM}↳ {key}:{_RESET} {_MAGENTA}{preview}{_RESET}")
                    elif key == "iteration":
                        print(f"  {_DIM}↳ {key}:{_RESET} {_YELLOW}{preview}{_RESET}")
                    else:
                        print(f"  {_DIM}↳ {key}:{_RESET} {color}{preview}{_RESET}")

        print(f"\n{_GREEN}{_BOLD}{'═'*60}{_RESET}")
        print(f"{_GREEN}{_BOLD}  ✔ EJECUCIÓN COMPLETADA{_RESET}")
        print(f"{_GREEN}{_BOLD}{'═'*60}{_RESET}")

    result = docops_agent.invoke(initial_state)

    return {
        "answer": result["draft"],
        "quality_score": result["quality_score"],
        "iterations": result["iteration"],
        "plan": result["plan"],
    }


def visualize_graph():
    """Genera y muestra el diagrama del grafo (requiere IPython)."""
    try:
        from IPython.display import Image, display
        img = docops_agent.get_graph(xray=True).draw_mermaid_png()
        display(Image(img))
    except ImportError:
        print(docops_agent.get_graph(xray=True).draw_mermaid())


# ─── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    fallback = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")

    print(f"\n{_BOLD}{_WHITE}{'═'*60}{_RESET}")
    print(f"{_BOLD}{_WHITE}  DocOps Agent — Sistema Multiagente con LangGraph{_RESET}")
    print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}")
    print(f"  {_DIM}Modelo principal:{_RESET} {_CYAN}{model} via Groq{_RESET}")
    print(f"  {_DIM}Modelo fallback: {_RESET} {_YELLOW}{fallback}{_RESET}")
    print()
    print(f"  {_DIM}Flujo:{_RESET} {_BLUE}Planner{_RESET} → {_CYAN}Retriever{_RESET} → {_YELLOW}Executor{_RESET} → {_MAGENTA}Verifier{_RESET} → {_GREEN}END{_RESET}")
    print(f"  {_DIM}Bucle:{_RESET} {_MAGENTA}Verifier{_RESET} {_RED}--revise-->{_RESET} {_YELLOW}Executor{_RESET}")
    print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}")

    query = "¿Cuál es la política de reembolso para clientes premium?"

    result = invoke_docops(query, verbose=True)

    # Respuesta final
    score = result["quality_score"]
    score_color = _GREEN if score >= 0.8 else _RED

    print(f"\n{_GREEN}{_BOLD}{'─'*60}{_RESET}")
    print(f"{_GREEN}{_BOLD}  📄 RESPUESTA FINAL{_RESET}")
    print(f"{_GREEN}{_BOLD}{'─'*60}{_RESET}")
    print(f"{_WHITE}{result['answer']}{_RESET}")
    print(f"\n  {_DIM}Score de calidad:{_RESET} {score_color}{_BOLD}{score}{_RESET}")
    print(f"  {_DIM}Iteraciones:{_RESET}      {_YELLOW}{_BOLD}{result['iterations']}{_RESET}")
    print()
