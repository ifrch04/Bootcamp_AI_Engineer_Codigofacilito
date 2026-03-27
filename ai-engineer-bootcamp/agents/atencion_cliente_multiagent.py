"""
Atención al Cliente Multiagente — Caso práctico Clase 11

Caso: El usuario manda UN solo mensaje con preguntas de DISTINTOS temas:
  "Hola cómo estás? Cuál es la dirección de la sede de Medellín,
   cuál es la misión de la empresa y cuánto vale la torta americana?"

Flujo:
  1. SUPERVISOR analiza el mensaje y clasifica cada pregunta por tema
  2. Despacha a los WORKERS especializados (solo los que apliquen)
  3. Cada WORKER responde usando su base de conocimiento
  4. SUPERVISOR recopila todo y arma UNA respuesta unificada

Arquitectura (grafo):

  START → supervisor → greeting_worker ─┐
                     → company_worker  ──┤→ aggregator → END
                     → products_worker ──┘

Usa GPT-OSS-120B vía Groq (OpenAI-compatible endpoint)
"""

import os
import operator
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

AGENT_META = {
    "supervisor": {
        "color": _BLUE,
        "icon": "🧠",
        "label": "SUPERVISOR",
        "desc": "Analiza el mensaje, clasifica cada pregunta por tema y asigna workers",
    },
    "greeting_worker": {
        "color": _GREEN,
        "icon": "👋",
        "label": "WORKER SALUDOS",
        "desc": "Maneja saludos y conversación casual con el cliente",
    },
    "company_worker": {
        "color": _CYAN,
        "icon": "🏢",
        "label": "WORKER EMPRESA",
        "desc": "Responde sobre sedes, misión, horarios y datos de la empresa",
    },
    "products_worker": {
        "color": _YELLOW,
        "icon": "🛒",
        "label": "WORKER PRODUCTOS",
        "desc": "Consulta catálogo de productos, precios y disponibilidad",
    },
    "aggregator": {
        "color": _MAGENTA,
        "icon": "📝",
        "label": "AGREGADOR",
        "desc": "Combina las respuestas de todos los workers en una respuesta unificada",
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


# ─── BASES DE CONOCIMIENTO (simuladas) ──────────────────────
# En producción esto vendría de ChromaDB / RAG pipeline (clases 5-7)

KB_EMPRESA = """
INFORMACIÓN DE LA EMPRESA — Pastelería La Delicia

MISIÓN: Endulzar la vida de nuestros clientes con productos artesanales
de la más alta calidad, elaborados con ingredientes frescos y recetas
tradicionales, brindando una experiencia cálida y memorable en cada visita.

VISIÓN: Ser la cadena de pastelerías artesanales más querida de Colombia
para 2027, reconocida por su calidad, innovación y servicio excepcional.

SEDES:
- Medellín: Calle 10 #43A-25, El Poblado. Tel: (604) 444-5678. Horario: Lun-Sáb 7am-8pm, Dom 8am-6pm.
- Bogotá: Carrera 11 #82-71, Zona G. Tel: (601) 321-9876. Horario: Lun-Sáb 7am-9pm, Dom 8am-7pm.
- Cali: Avenida 6N #17-50, Granada. Tel: (602) 555-1234. Horario: Lun-Sáb 8am-8pm, Dom 9am-5pm.

FUNDACIÓN: 2015, Medellín, Colombia.
CONTACTO: info@ladelicia.com.co | WhatsApp: +57 300 123 4567
"""

KB_PRODUCTOS = """
CATÁLOGO DE PRODUCTOS — Pastelería La Delicia (precios en COP)

TORTAS:
- Torta Americana (chocolate con frosting de vainilla): $45.000 porción / $180.000 entera (20 porciones)
- Torta de Zanahoria: $42.000 porción / $165.000 entera
- Torta Red Velvet: $48.000 porción / $190.000 entera
- Torta de Tres Leches: $38.000 porción / $150.000 entera
- Torta Selva Negra: $50.000 porción / $200.000 entera

PANES:
- Pan artesanal (hogaza): $12.000
- Croissant de mantequilla: $8.500
- Pan de queso (6 unidades): $15.000

BEBIDAS:
- Café americano: $6.000
- Cappuccino: $9.500
- Chocolate caliente: $8.000
- Jugo natural: $10.000

COMBOS:
- Combo Desayuno (croissant + café): $14.000
- Combo Tarde (porción de torta + bebida caliente): $18.000 (ahorro de $5.500)

Nota: Precios incluyen IVA. Domicilios disponibles vía app con costo adicional.
"""


# ─── CONTRATO: Clasificación del Supervisor ──────────────────
class QuestionClassification(BaseModel):
    """El supervisor clasifica cada pregunta del usuario por tema."""
    greeting: str = Field(
        default="",
        description="Texto del saludo o conversación casual. Vacío si no hay saludo.",
    )
    company: str = Field(
        default="",
        description="Preguntas sobre la empresa: sedes, misión, visión, horarios, contacto. Vacío si no hay.",
    )
    products: str = Field(
        default="",
        description="Preguntas sobre productos, precios, menú, disponibilidad. Vacío si no hay.",
    )

    #Fallback

# ─── ESTADO COMPARTIDO ──────────────────────────────────────
class CustomerState(TypedDict):
    """Estado compartido entre el supervisor y los workers."""
    messages: Annotated[list[AnyMessage], operator.add]
    # Clasificación del supervisor
    greeting_question: str
    company_question: str
    products_question: str
    # Respuestas de cada worker
    greeting_response: str
    company_response: str
    products_response: str
    # Respuesta final
    final_response: str


# ─── NODO 1: SUPERVISOR ─────────────────────────────────────
def supervisor(state: CustomerState) -> dict:
    """
    Analiza el mensaje del cliente y clasifica cada parte por tema.
    Es el "cerebro" que decide qué workers necesitan trabajar.
    """
    user_msg = state["messages"][-1].content

    classifier = llm.with_structured_output(QuestionClassification)

    try:
        result = classifier.invoke([
            SystemMessage(content=(
                "Eres el supervisor de atención al cliente de una pastelería.\n"
                "El cliente puede mandar UN solo mensaje con varias preguntas mezcladas.\n\n"
                "Tu trabajo: separar y clasificar cada parte del mensaje en categorías:\n"
                "- greeting: saludos, despedidas, conversación casual\n"
                "- company: preguntas sobre la empresa (sedes, dirección, misión, visión, horarios)\n"
                "- products: preguntas sobre productos, precios, menú, disponibilidad\n\n"
                "Copia el texto EXACTO de cada pregunta en su campo correspondiente.\n"
                "Si una categoría no aplica, déjala vacía.\n"
                "Una pregunta puede pertenecer a UNA sola categoría."
            )),
            HumanMessage(content=f"Mensaje del cliente:\n{user_msg}"),
        ])

        return {
            "greeting_question": result.greeting,
            "company_question": result.company,
            "products_question": result.products,
        }

    except Exception as e:
        # Fallback: todo va al worker de empresa
        return {
            "greeting_question": "",
            "company_question": user_msg,
            "products_question": "",
        }


# ─── NODO 2: WORKER DE SALUDOS ──────────────────────────────
def greeting_worker(state: CustomerState) -> dict:
    """
    Maneja la parte social: saludos, despedidas, conversación casual.
    No necesita base de conocimiento — solo ser amable.
    """
    question = state.get("greeting_question", "")
    if not question.strip():
        return {"greeting_response": ""}

    response = llm.invoke([
        SystemMessage(content=(
            "Eres un agente amable de atención al cliente de 'Pastelería La Delicia'.\n"
            "Responde SOLO al saludo o conversación casual del cliente.\n"
            "Sé cálido, breve y profesional. Máximo 1-2 oraciones.\n"
            "NO respondas preguntas sobre productos o la empresa aquí."
        )),
        HumanMessage(content=question),
    ])

    return {"greeting_response": response.content}


# ─── NODO 3: WORKER DE EMPRESA ──────────────────────────────
def company_worker(state: CustomerState) -> dict:
    """
    Responde preguntas sobre la empresa usando la base de conocimiento.
    Sedes, misión, visión, horarios, contacto.
    """
    question = state.get("company_question", "")
    if not question.strip():
        return {"company_response": ""}

    response = llm.invoke([
        SystemMessage(content=(
            "Eres un agente de atención al cliente de 'Pastelería La Delicia'.\n"
            "Responde SOLO con información de la base de conocimiento proporcionada.\n"
            "Si la información no está en la base, di que no tienes esa información.\n"
            "Sé preciso y conciso.\n\n"
            f"BASE DE CONOCIMIENTO:\n{KB_EMPRESA}"
        )),
        HumanMessage(content=question),
    ])

    return {"company_response": response.content}


# ─── NODO 4: WORKER DE PRODUCTOS ────────────────────────────
def products_worker(state: CustomerState) -> dict:
    """
    Responde preguntas sobre productos, precios y disponibilidad.
    Consulta el catálogo de la pastelería.
    """
    question = state.get("products_question", "")
    if not question.strip():
        return {"products_response": ""}

    response = llm.invoke([
        SystemMessage(content=(
            "Eres un agente de atención al cliente de 'Pastelería La Delicia'.\n"
            "Responde SOLO con información del catálogo de productos.\n"
            "Incluye precios exactos cuando los pregunten.\n"
            "Si el producto no está en el catálogo, dilo amablemente.\n"
            "Sé preciso y conciso.\n\n"
            f"CATÁLOGO:\n{KB_PRODUCTOS}"
        )),
        HumanMessage(content=question),
    ])

    return {"products_response": response.content}


# ─── NODO 5: AGREGADOR ──────────────────────────────────────
def aggregator(state: CustomerState) -> dict:
    """
    Combina las respuestas de todos los workers en UNA sola
    respuesta coherente y natural para el cliente.
    """
    parts = []
    if state.get("greeting_response"):
        parts.append(f"[Saludo]\n{state['greeting_response']}")
    if state.get("company_response"):
        parts.append(f"[Info empresa]\n{state['company_response']}")
    if state.get("products_response"):
        parts.append(f"[Productos]\n{state['products_response']}")

    if not parts:
        return {"final_response": "Lo siento, no pude procesar tu consulta."}

    combined = "\n\n".join(parts)

    response = llm.invoke([
        SystemMessage(content=(
            "Eres el agente final de atención al cliente de 'Pastelería La Delicia'.\n\n"
            "Recibiste las respuestas individuales de varios especialistas.\n"
            "Tu trabajo: combinarlas en UNA SOLA respuesta natural y fluida.\n\n"
            "Reglas:\n"
            "- Empieza con el saludo (si lo hay)\n"
            "- Luego responde cada tema en orden natural\n"
            "- No uses etiquetas como [Saludo] o [Productos]\n"
            "- Que se lea como si fuera UNA sola persona respondiendo\n"
            "- Cierra con un ofrecimiento de ayuda adicional\n"
            "- Tono: cálido, profesional, conciso"
        )),
        HumanMessage(content=(
            f"Mensaje original del cliente: {state['messages'][-1].content}\n\n"
            f"Respuestas de los especialistas:\n{combined}"
        )),
    ])

    return {"final_response": response.content}


# ─── CONSTRUCCIÓN DEL GRAFO ─────────────────────────────────
def build_customer_agent():
    """
    Construye el grafo multiagente de atención al cliente.

    Flujo:
      START → supervisor → greeting_worker → company_worker
            → products_worker → aggregator → END

    Los workers corren en secuencia pero cada uno solo procesa
    su parte (si no tiene preguntas asignadas, retorna vacío).
    """
    workflow = StateGraph(CustomerState)

    # Registrar nodos
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("greeting_worker", greeting_worker)
    workflow.add_node("company_worker", company_worker)
    workflow.add_node("products_worker", products_worker)
    workflow.add_node("aggregator", aggregator)

    # Flujo: supervisor clasifica → workers procesan → agregador combina
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "greeting_worker")
    workflow.add_edge("greeting_worker", "company_worker")
    workflow.add_edge("company_worker", "products_worker")
    workflow.add_edge("products_worker", "aggregator")
    workflow.add_edge("aggregator", END)

    return workflow.compile()


# Instancia global
customer_agent = build_customer_agent()


# ─── UTILIDADES ──────────────────────────────────────────────
def invoke_customer_agent(query: str, verbose: bool = False) -> dict:
    """
    Invoca el sistema multiagente de atención al cliente.

    Args:
        query: Mensaje del cliente (puede tener varias preguntas mezcladas)
        verbose: Si True, imprime el estado de cada paso con colores

    Returns:
        dict con keys: answer, classification, worker_responses
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "greeting_question": "",
        "company_question": "",
        "products_question": "",
        "greeting_response": "",
        "company_response": "",
        "products_response": "",
        "final_response": "",
    }

    if verbose:
        print(f"\n{_BOLD}{_WHITE}{'═'*60}{_RESET}")
        print(f"{_BOLD}{_WHITE}  🏪 PASTELERÍA LA DELICIA — Atención al Cliente{_RESET}")
        print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}")
        print(f"  {_DIM}Cliente dice:{_RESET} {_WHITE}{query}{_RESET}")
        print()
        print(f"  {_DIM}Flujo:{_RESET} {_BLUE}Supervisor{_RESET} → {_GREEN}Saludos{_RESET} → {_CYAN}Empresa{_RESET} → {_YELLOW}Productos{_RESET} → {_MAGENTA}Agregador{_RESET}")
        print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}")

        step = 0
        for event in customer_agent.stream(initial_state, stream_mode="updates"):
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
                    val_str = str(value).strip()
                    if not val_str:
                        print(f"  {_DIM}↳ {key}: (sin contenido — no aplica){_RESET}")
                    else:
                        preview = val_str[:300]
                        print(f"  {_DIM}↳ {key}:{_RESET} {color}{preview}{_RESET}")

        print(f"\n{_GREEN}{_BOLD}{'═'*60}{_RESET}")
        print(f"{_GREEN}{_BOLD}  ✔ ATENCIÓN COMPLETADA{_RESET}")
        print(f"{_GREEN}{_BOLD}{'═'*60}{_RESET}")

    result = customer_agent.invoke(initial_state)

    return {
        "answer": result["final_response"],
        "classification": {
            "saludo": result["greeting_question"],
            "empresa": result["company_question"],
            "productos": result["products_question"],
        },
        "worker_responses": {
            "saludo": result["greeting_response"],
            "empresa": result["company_response"],
            "productos": result["products_response"],
        },
    }


# ─── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{_BOLD}{_WHITE}{'═'*60}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Caso práctico: Mensaje multi-tema de un cliente{_RESET}")
    print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}")
    print(f"  {_DIM}Modelo:{_RESET} {_CYAN}{os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')} via Groq{_RESET}")
    print()

    # ── El caso exacto de la clase ──
    query = (
        "Hola cómo estás?"
        "cuánto vale la torta americana?"
    )

    result = invoke_customer_agent(query, verbose=True)

    # ── Respuesta final unificada ──
    print(f"\n{_GREEN}{_BOLD}{'─'*60}{_RESET}")
    print(f"{_GREEN}{_BOLD}  💬 RESPUESTA AL CLIENTE{_RESET}")
    print(f"{_GREEN}{_BOLD}{'─'*60}{_RESET}")
    print(f"{_WHITE}{result['answer']}{_RESET}")

    # ── Resumen de clasificación (para que los alumnos vean el desglose) ──
    print(f"\n{_BLUE}{_BOLD}{'─'*60}{_RESET}")
    print(f"{_BLUE}{_BOLD}  🔎 DESGLOSE (cómo el Supervisor clasificó el mensaje){_RESET}")
    print(f"{_BLUE}{_BOLD}{'─'*60}{_RESET}")
    for tema, pregunta in result["classification"].items():
        color = {"saludo": _GREEN, "empresa": _CYAN, "productos": _YELLOW}[tema]
        if pregunta:
            print(f"  {color}● {tema.upper()}: {pregunta}{_RESET}")
        else:
            print(f"  {_DIM}○ {tema.upper()}: (no detectado){_RESET}")
    print()
