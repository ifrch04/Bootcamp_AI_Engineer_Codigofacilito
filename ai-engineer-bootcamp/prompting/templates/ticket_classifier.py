"""Ticket classifier prompt templates (v1-v4) for Clase 3."""

from __future__ import annotations

from prompting.promptkit import PromptChain, PromptRegistry, PromptTemplate

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------
registry = PromptRegistry()

# ---------------------------------------------------------------------------
# v1 — Base (intentionally weak / ambiguous)
# ---------------------------------------------------------------------------
v1 = PromptTemplate(
    name="ticket_classifier_v1",
    template=(
        "Clasifica el siguiente ticket de soporte.\n"
        "Ticket: {ticket}\n"
        "Responde con la categoría y la prioridad."
    ),
    metadata={"version": "1.0", "description": "Base - prompt débil y ambiguo"},
)

# ---------------------------------------------------------------------------
# v2 — Few-shot with explicit format
# ---------------------------------------------------------------------------
v2 = PromptTemplate(
    name="ticket_classifier_v2",
    template=(
        "Clasifica el siguiente ticket de soporte técnico.\n"
        "\n"
        "Categorías válidas: acceso, rendimiento, facturación, infraestructura, "
        "feature_request, bug, datos\n"
        "Prioridades válidas: crítica, alta, media, baja\n"
        "\n"
        'Responde en formato JSON: {{"categoria": "...", "prioridad": "..."}}\n'
        "\n"
        "Ejemplos:\n"
        "\n"
        'Input: "Mi cuenta está bloqueada después de varios intentos de login"\n'
        'Output: {{"categoria": "acceso", "prioridad": "alta"}}\n'
        "\n"
        'Input: "Las consultas SQL tardan más de 10 segundos"\n'
        'Output: {{"categoria": "rendimiento", "prioridad": "media"}}\n'
        "\n"
        'Input: "Quisiera que agregaran soporte para exportar a Excel"\n'
        'Output: {{"categoria": "feature_request", "prioridad": "baja"}}\n'
        "\n"
        "Ticket: {ticket}"
    ),
    metadata={"version": "2.0", "description": "Few-shot con formato JSON explícito"},
)

# ---------------------------------------------------------------------------
# v3 — Restrictions on top of v2
# ---------------------------------------------------------------------------
v3 = PromptTemplate(
    name="ticket_classifier_v3",
    template=(
        "Clasifica el siguiente ticket de soporte técnico.\n"
        "\n"
        "Categorías válidas: acceso, rendimiento, facturación, infraestructura, "
        "feature_request, bug, datos\n"
        "Prioridades válidas: crítica, alta, media, baja\n"
        "\n"
        "Reglas:\n"
        "- Responde SOLO con JSON válido, sin texto adicional ni markdown\n"
        "- Usa EXACTAMENTE una de las categorías listadas\n"
        "- Si el ticket menciona que afecta a múltiples usuarios, sube la prioridad un nivel\n"
        '- Si no puedes determinar la categoría, usa "otros"\n'
        "\n"
        'Formato de respuesta: {{"categoria": "...", "prioridad": "..."}}\n'
        "\n"
        "Ejemplos:\n"
        "\n"
        'Input: "Mi cuenta está bloqueada después de varios intentos de login"\n'
        'Output: {{"categoria": "acceso", "prioridad": "alta"}}\n'
        "\n"
        'Input: "Las consultas SQL tardan más de 10 segundos"\n'
        'Output: {{"categoria": "rendimiento", "prioridad": "media"}}\n'
        "\n"
        'Input: "Quisiera que agregaran soporte para exportar a Excel"\n'
        'Output: {{"categoria": "feature_request", "prioridad": "baja"}}\n'
        "\n"
        "Ticket: {ticket}"
    ),
    metadata={"version": "3.0", "description": "Restricciones estrictas sobre v2"},
)

# ---------------------------------------------------------------------------
# v4 — Two-step PromptChain
# ---------------------------------------------------------------------------
_v4_step1 = PromptTemplate(
    name="ticket_classifier_v4_extractor",
    template=(
        "Del siguiente ticket de soporte, extrae la siguiente información:\n"
        "- problema_principal: descripción breve del problema\n"
        '- usuarios_afectados: número estimado o "desconocido"\n'
        "- urgencia_implícita: alta, media, o baja\n"
        "- palabras_clave: lista de palabras clave relevantes\n"
        "\n"
        "Responde en JSON.\n"
        "\n"
        "Ticket: {ticket}"
    ),
    metadata={"version": "4.0", "description": "Chain paso 1 - extractor"},
)

_v4_step2 = PromptTemplate(
    name="ticket_classifier_v4_clasificador",
    template=(
        "Usando la siguiente extracción de un ticket de soporte:\n"
        "{extraction_result}\n"
        "\n"
        "Clasifica el ticket con categoría y prioridad.\n"
        "\n"
        "Categorías válidas: acceso, rendimiento, facturación, infraestructura, "
        "feature_request, bug, datos\n"
        "Prioridades válidas: crítica, alta, media, baja\n"
        "\n"
        "Reglas:\n"
        "- Responde SOLO con JSON válido, sin texto adicional ni markdown\n"
        '- Formato: {{"categoria": "...", "prioridad": "..."}}\n'
    ),
    metadata={"version": "4.0", "description": "Chain paso 2 - clasificador"},
)

v4_chain = PromptChain(templates=[_v4_step1, _v4_step2])

# ---------------------------------------------------------------------------
# Register all templates
# ---------------------------------------------------------------------------
registry.register(v1)
registry.register(v2)
registry.register(v3)
registry.register(_v4_step1)
registry.register(_v4_step2)
