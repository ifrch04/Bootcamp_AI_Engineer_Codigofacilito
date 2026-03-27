"""
memory/store.py — Configuración de checkpointing para DocOps Agent
Clase 12: Memoria persistente

Provee el checkpointer que persiste el estado del grafo.
Cambiar de desarrollo a producción solo requiere cambiar la instancia.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── SELECCIÓN DE CHECKPOINTER POR ENTORNO ───
ENVIRONMENT = os.getenv("DOCOPS_ENV", "development")


def get_checkpointer():
    """
    Factory que retorna el checkpointer apropiado según el entorno.

    Entornos:
    - "development": MemorySaver (RAM, se pierde al morir el proceso)
    - "staging": SqliteSaver (archivo .db, sobrevive reinicios)
    - "production": PostgresSaver (PostgreSQL, multi-proceso, durable)
    """
    if ENVIRONMENT == "production":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            DATABASE_URL = os.getenv("DATABASE_URL")
            if not DATABASE_URL:
                raise ValueError(
                    "DATABASE_URL requerido para entorno de producción"
                )
            return PostgresSaver.from_conn_string(DATABASE_URL)
        except ImportError:
            print(
                "WARN: langgraph-checkpoint-postgres no instalado. "
                "Usando SqliteSaver como fallback."
            )
            # Fallback a SQLite
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = os.getenv("SQLITE_DB_PATH", "docops_checkpoints.db")
            return SqliteSaver.from_conn_string(db_path)

    elif ENVIRONMENT == "staging":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = os.getenv("SQLITE_DB_PATH", "docops_checkpoints.db")
        return SqliteSaver.from_conn_string(db_path)

    else:
        # development (default)
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


# ─── INSTANCIA GLOBAL ───
checkpointer = get_checkpointer()


# ─── UTILIDADES DE ESTADO ───
def inspect_thread(graph, thread_id: str) -> dict:
    """
    Inspecciona el estado actual de un thread.

    Args:
        graph: El grafo compilado (docops_agent)
        thread_id: ID del thread a inspeccionar

    Returns:
        dict con next_node, values y metadata
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    return {
        "thread_id": thread_id,
        "next_node": snapshot.next if snapshot.next else ["END"],
        "values": {
            k: (str(v)[:200] + "..." if len(str(v)) > 200 else v)
            for k, v in snapshot.values.items()
            if k != "messages"  # Excluir mensajes para legibilidad
        },
        "message_count": len(snapshot.values.get("messages", [])),
        "has_checkpoint": snapshot.config is not None,
    }


def get_thread_history(graph, thread_id: str, max_steps: int = 20) -> list:
    """
    Retorna el historial de pasos de un thread para auditoría.

    Args:
        graph: El grafo compilado
        thread_id: ID del thread
        max_steps: Máximo de pasos a retornar

    Returns:
        Lista de dicts con info de cada paso
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = []

    for i, state in enumerate(graph.get_state_history(config)):
        if i >= max_steps:
            break
        history.append({
            "step": i,
            "next_node": state.next[0] if state.next else "END",
            "quality_score": state.values.get("quality_score", None),
            "iteration": state.values.get("iteration", 0),
            "message_count": len(state.values.get("messages", [])),
            "config": state.config,
        })

    return history


def rollback_thread(graph, thread_id: str, steps_back: int = 1):
    """
    Rebobina un thread N pasos hacia atrás.

    Args:
        graph: El grafo compilado
        thread_id: ID del thread
        steps_back: Cuántos pasos retroceder

    Returns:
        El config del checkpoint restaurado, o None si no hay historial
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))

    if steps_back >= len(history):
        print(f"Solo hay {len(history)} pasos. No se puede retroceder {steps_back}.")
        return None

    target = history[steps_back]
    graph.update_state(target.config, {})
    print(f"Thread '{thread_id}' rebobinado {steps_back} paso(s).")
    return target.config


# ─── MAIN (para testing) ───
if __name__ == "__main__":
    print(f"Entorno: {ENVIRONMENT}")
    print(f"Checkpointer: {type(checkpointer).__name__}")
    print("memory/store.py cargado correctamente.")
