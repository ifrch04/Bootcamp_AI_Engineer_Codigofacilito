"""
agents/hitl.py — Human-in-the-Loop para DocOps Agent
Clase 12: Aprobaciones humanas condicionales

Implementa:
- Evaluación de riesgo por contenido y quality_score
- Gate humano con interrupt() condicional
- Logging de decisiones humanas para auditoría
"""
import json
import logging
from datetime import datetime, timezone
from typing import Literal

from langgraph.types import interrupt

logger = logging.getLogger("docops.audit")


# ─── CLASIFICACIÓN DE RIESGO ───
RISKY_KEYWORDS = {
    "high": [
        "eliminar", "borrar", "enviar", "delete", "send",
        "remove", "drop", "truncate", "modificar", "update",
        "transferir", "transfer", "publicar", "publish",
    ],
    "critical": [
        "drop table", "delete all", "rm -rf", "format",
        "borrar todo", "eliminar cuenta", "delete account",
    ],
}


def assess_risk(state: dict) -> Literal["low", "medium", "high", "critical"]:
    """
    Evalúa el nivel de riesgo de la acción actual del agente.

    Criterios:
    - quality_score < 0.6 → high (baja confianza del verifier)
    - Palabras clave críticas en el draft → critical
    - Palabras clave de riesgo en el draft → high
    - quality_score < 0.75 → medium (confianza moderada)
    - Todo lo demás → low

    Args:
        state: Estado actual del grafo DocOps

    Returns:
        Nivel de riesgo: "low", "medium", "high", o "critical"
    """
    draft = state.get("draft", "").lower()
    score = state.get("quality_score", 0.0)

    # Regla 1: Score muy bajo = siempre alto riesgo
    if score < 0.6:
        return "high"

    # Regla 2: Palabras clave críticas
    for keyword in RISKY_KEYWORDS["critical"]:
        if keyword in draft:
            return "critical"

    # Regla 3: Palabras clave de riesgo
    for keyword in RISKY_KEYWORDS["high"]:
        if keyword in draft:
            return "high"

    # Regla 4: Score moderado = riesgo medio
    if score < 0.75:
        return "medium"

    # Default: riesgo bajo
    return "low"


# ─── LOGGING DE DECISIONES HUMANAS ───
def log_human_decision(
    thread_id: str,
    decision: dict,
    user_id: str = "unknown",
    risk_level: str = "unknown",
):
    """
    Registra cada intervención humana para auditoría.

    Args:
        thread_id: ID del thread donde ocurrió
        decision: Diccionario con la decisión del humano
        user_id: Identificador del humano que decidió
        risk_level: Nivel de riesgo que disparó la interrupción
    """
    entry = {
        "event": "human_decision",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "user_id": user_id,
        "risk_level": risk_level,
        "approved": decision.get("approved", False),
        "edited": "edited_draft" in decision,
        "reason": decision.get("reason", ""),
    }
    logger.info(json.dumps(entry))
    return entry


def log_auto_approved(thread_id: str, risk_level: str):
    """Registra cuando una acción se auto-aprueba por bajo riesgo."""
    entry = {
        "event": "auto_approved",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "risk_level": risk_level,
    }
    logger.info(json.dumps(entry))
    return entry


# ─── GATE HUMANO (NODO DEL GRAFO) ───
def human_gate(state: dict) -> dict:
    """
    Nodo del grafo que evalúa riesgo y decide si pausar para aprobación.

    Se inserta entre verifier y END en el grafo multiagente.

    Comportamiento:
    - Riesgo "low" o "medium": continúa sin interrupción (auto-aprobado)
    - Riesgo "high" o "critical": pausa con interrupt() y espera decisión

    La decisión del humano puede ser:
    - {"approved": True} → continúa con el draft actual
    - {"approved": True, "edited_draft": "..."} → continúa con draft editado
    - {"approved": False, "reason": "..."} → rechazado

    Args:
        state: Estado completo del grafo DocOps

    Returns:
        Estado actualizado según la decisión (o sin cambios si auto-aprobado)
    """
    risk = assess_risk(state)
    force = state.get("force_review", False)

    # ─── RIESGO ALTO/CRÍTICO o force_review=True: PEDIR APROBACIÓN ───
    if risk in ("high", "critical") or force:
        effective_risk = risk if risk in ("high", "critical") else "review"
        # Preparar la información para el humano
        interrupt_payload = {
            "risk_level": effective_risk,
            "message": _build_human_message(effective_risk, state),
            "draft_preview": state.get("draft", "")[:1000],
            "quality_score": state.get("quality_score", 0.0),
            "iteration": state.get("iteration", 0),
            "action": "approve_edit_reject",
        }

        # ── PAUSA: espera decisión humana ──
        decision = interrupt(interrupt_payload)

        # ── REANUDADO: procesar decisión ──
        if not decision.get("approved", False):
            # Rechazado
            rejection_reason = decision.get("reason", "Rechazado por humano")
            return {
                "draft": (
                    f"[RECHAZADO por supervisor]\n"
                    f"Razón: {rejection_reason}\n\n"
                    f"Draft original:\n{state.get('draft', '')[:500]}"
                ),
                "quality_score": 0.0,
            }

        # Aprobado (posiblemente con ediciones)
        if "edited_draft" in decision:
            return {"draft": decision["edited_draft"]}

        # Aprobado sin cambios
        return {}

    # ─── RIESGO BAJO O MEDIO: AUTO-APROBADO ───
    # No se interrumpe, el agente continúa normalmente
    return {}


def _build_human_message(risk: str, state: dict) -> str:
    """Construye un mensaje legible para el humano."""
    risk_emoji = {"high": "⚠️", "critical": "🚨", "review": "👁️"}.get(risk, "ℹ️")
    score = state.get("quality_score", 0.0)
    iterations = state.get("iteration", 0)

    msg = (
        f"{risk_emoji} Acción de riesgo {risk.upper()} detectada.\n\n"
        f"Quality score: {score:.2f} | Iteraciones: {iterations}\n\n"
        f"Opciones:\n"
        f'  - Aprobar: {{"approved": true}}\n'
        f'  - Editar:  {{"approved": true, "edited_draft": "tu texto"}}\n'
        f'  - Rechazar: {{"approved": false, "reason": "motivo"}}\n'
    )
    return msg


# ─── VARIANTE: GATE CON DOBLE APROBACIÓN (RIESGO CRÍTICO) ───
def human_gate_strict(state: dict) -> dict:
    """
    Variante del gate que requiere doble aprobación para riesgo crítico.
    Usa dos llamadas a interrupt() en secuencia.

    Úsalo en lugar de human_gate si necesitas mayor seguridad.
    """
    risk = assess_risk(state)

    if risk == "critical":
        # Primera aprobación
        first_decision = interrupt({
            "step": "first_approval",
            "risk_level": risk,
            "message": "🚨 RIESGO CRÍTICO — Se requiere primera aprobación.",
            "draft_preview": state.get("draft", "")[:1000],
            "action": "approve_reject",
        })

        if not first_decision.get("approved", False):
            return {
                "draft": "[RECHAZADO en primera aprobación]",
                "quality_score": 0.0,
            }

        # Segunda aprobación (diferente aprobador idealmente)
        second_decision = interrupt({
            "step": "second_approval",
            "risk_level": risk,
            "message": "🚨 RIESGO CRÍTICO — Se requiere segunda aprobación.",
            "draft_preview": state.get("draft", "")[:1000],
            "first_approved_by": first_decision.get("user_id", "unknown"),
            "action": "approve_reject",
        })

        if not second_decision.get("approved", False):
            return {
                "draft": "[RECHAZADO en segunda aprobación]",
                "quality_score": 0.0,
            }

        return {}  # Doble aprobación exitosa

    elif risk == "high":
        # Misma lógica que human_gate para riesgo alto
        decision = interrupt({
            "risk_level": risk,
            "message": "⚠️ Riesgo ALTO — Aprobación requerida.",
            "draft_preview": state.get("draft", "")[:1000],
            "action": "approve_edit_reject",
        })

        if not decision.get("approved", False):
            return {
                "draft": f"[RECHAZADO: {decision.get('reason', '')}]",
                "quality_score": 0.0,
            }
        if "edited_draft" in decision:
            return {"draft": decision["edited_draft"]}
        return {}

    # Riesgo bajo/medio: auto-aprobado
    return {}


# ─── MAIN (para testing) ───
if __name__ == "__main__":
    # Test de evaluación de riesgo
    test_cases = [
        {"draft": "La política indica que el reembolso es de 30 días.", "quality_score": 0.9},
        {"draft": "Procedo a eliminar los registros de la base de datos.", "quality_score": 0.85},
        {"draft": "Resultado parcial.", "quality_score": 0.5},
        {"draft": "Ejecutando DROP TABLE users", "quality_score": 0.95},
    ]

    for i, tc in enumerate(test_cases):
        risk = assess_risk(tc)
        print(f"Caso {i+1}: score={tc['quality_score']}, risk={risk}")
        print(f"  Draft: {tc['draft'][:60]}...")
        print()
