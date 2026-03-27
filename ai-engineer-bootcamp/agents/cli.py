"""
agents/cli.py — CLI interactivo para DocOps Agent con HITL y persistencia SQLite
Clase 12: Human-in-the-Loop real desde terminal

Uso:
    python agents/cli.py                         # nuevo thread automático
    python agents/cli.py --thread mi-sesion      # retomar thread existente
    python agents/cli.py --db /ruta/custom.db    # base de datos personalizada

Comandos en sesión:
    /new            Crear nuevo thread
    /thread <id>    Cambiar a thread existente
    /threads        Listar todos los threads guardados
    /history        Ver historial de pasos del thread actual
    /exit           Salir
"""
import argparse
import sqlite3
import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agents.multi_agent_graph import build_docops_agent

# ─── Configuración ───
DEFAULT_DB = "docops_checkpoints.db"

# ─── Colores ANSI ───
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
MAGENTA = "\033[35m"
WHITE  = "\033[97m"
BLUE   = "\033[34m"

NODE_ICONS = {
    "planner":    ("📋", BLUE),
    "retriever":  ("🔍", CYAN),
    "executor":   ("⚙️ ", YELLOW),
    "verifier":   ("✅", MAGENTA),
    "human_gate": ("🔐", RED),
}


# ─── UI helpers ─────────────────────────────────────────────

def banner(db_path: str):
    print(f"\n{BOLD}{WHITE}{'═'*62}{RESET}")
    print(f"{BOLD}{WHITE}   DocOps Agent CLI  —  HITL Interactivo{RESET}")
    print(f"{BOLD}{WHITE}   Persistencia: {DIM}SQLite → {db_path}{RESET}")
    print(f"{BOLD}{WHITE}{'═'*62}{RESET}")
    print(f"  {DIM}Comandos:{RESET}  "
          f"{CYAN}/new{RESET}  "
          f"{CYAN}/thread <id>{RESET}  "
          f"{CYAN}/threads{RESET}  "
          f"{CYAN}/history{RESET}  "
          f"{CYAN}/exit{RESET}")
    print(f"{BOLD}{WHITE}{'═'*62}{RESET}\n")


def print_node_event(node_name: str, node_output: dict):
    icon, color = NODE_ICONS.get(node_name, ("▸", WHITE))
    score_str = ""
    if isinstance(node_output, dict) and "quality_score" in node_output:
        s = node_output["quality_score"]
        sc = GREEN if s >= 0.8 else YELLOW if s >= 0.6 else RED
        score_str = f"  {sc}score={s:.2f}{RESET}"
    print(f"  {color}{icon} {node_name}{RESET}{score_str}")


def print_interrupt_panel(payload: dict):
    risk = payload.get("risk_level", "?")
    score = payload.get("quality_score", 0.0)
    draft = payload.get("draft_preview", "")
    rc = RED if risk == "critical" else YELLOW

    print(f"\n{rc}{BOLD}{'─'*62}{RESET}")
    print(f"{rc}{BOLD}  ⏸  PAUSA — Aprobación humana requerida{RESET}")
    print(f"{rc}{BOLD}{'─'*62}{RESET}")
    print(f"  {DIM}Nivel de riesgo:{RESET} {rc}{BOLD}{risk.upper()}{RESET}   "
          f"{DIM}Quality score:{RESET} {score:.2f}")
    print(f"\n{DIM}  Draft generado:{RESET}")
    for line in draft.split("\n"):
        print(f"    {line}")
    print(f"\n  {GREEN}[a]{RESET} Aprobar tal cual")
    print(f"  {YELLOW}[e]{RESET} Editar draft antes de aprobar")
    print(f"  {RED}[r]{RESET} Rechazar")
    print(f"{rc}{'─'*62}{RESET}")


def ask_decision(payload: dict) -> dict:
    """Solicita la decisión del humano y devuelve el dict para Command(resume=...)."""
    print_interrupt_panel(payload)

    while True:
        try:
            choice = input(f"\n{BOLD}Decisión [a/e/r]: {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Sin input — aprobando automáticamente.{RESET}")
            return {"approved": True}

        if choice in ("a", "aprobar", "approve", "y", "s", "si", ""):
            print(f"{GREEN}✓ Aprobado{RESET}")
            return {"approved": True}

        elif choice in ("e", "edit", "editar"):
            print(f"{YELLOW}Escribe el draft corregido.")
            print(f"{DIM}(Línea con solo '###' para terminar){RESET}")
            lines = []
            try:
                while True:
                    line = input()
                    if line.strip() == "###":
                        break
                    lines.append(line)
            except EOFError:
                pass
            edited = "\n".join(lines).strip()
            if not edited:
                print(f"{YELLOW}Draft vacío — aprobando sin cambios.{RESET}")
                return {"approved": True}
            print(f"{GREEN}✓ Draft editado ({len(edited)} chars){RESET}")
            return {"approved": True, "edited_draft": edited}

        elif choice in ("r", "reject", "rechazar", "n", "no"):
            try:
                reason = input(f"{RED}Razón del rechazo: {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                reason = "Rechazado por el supervisor"
            print(f"{RED}✗ Rechazado{RESET}")
            return {"approved": False, "reason": reason or "Rechazado por el supervisor"}

        else:
            print(f"{DIM}Opción no reconocida. Escribe a, e o r.{RESET}")


# ─── Lógica principal ────────────────────────────────────────

def get_interrupt_payload(agent, config: dict) -> dict | None:
    """Extrae el payload del interrupt pendiente, si existe."""
    snapshot = agent.get_state(config)
    if not snapshot.next:
        return None
    if hasattr(snapshot, "tasks"):
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    return None


def run_query(agent, thread_id: str, query: str):
    """
    Ejecuta una consulta y gestiona el bucle HITL completo.
    Retorna el dict de valores finales del estado.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": "",
        "search_results": "",
        "draft": "",
        "feedback": "",
        "quality_score": 0.0,
        "iteration": 0,
    }

    print(f"\n{DIM}  thread:{RESET} {CYAN}{thread_id}{RESET}")
    print(f"{CYAN}{BOLD}  Procesando...{RESET}\n")

    # Primera ejecución con stream para mostrar progreso
    for event in agent.stream(initial_state, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            print_node_event(node_name, node_output)

    # Bucle HITL: puede haber múltiples interrupts (p.ej. human_gate_strict)
    while True:
        payload = get_interrupt_payload(agent, config)
        if payload is None:
            break

        decision = ask_decision(payload)

        print(f"\n{CYAN}{BOLD}  Reanudando...{RESET}\n")
        for event in agent.stream(Command(resume=decision), config, stream_mode="updates"):
            for node_name, node_output in event.items():
                print_node_event(node_name, node_output)

    return agent.get_state(config).values


def list_threads(conn: sqlite3.Connection):
    try:
        rows = conn.execute(
            "SELECT thread_id, COUNT(*) as steps "
            "FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
        ).fetchall()
        if not rows:
            print(f"  {DIM}No hay threads guardados.{RESET}\n")
        else:
            print(f"\n{BOLD}Threads en la base de datos:{RESET}")
            for thread_id, steps in rows:
                print(f"  {CYAN}•{RESET} {thread_id}  {DIM}({steps} checkpoints){RESET}")
            print()
    except Exception as e:
        print(f"  {RED}Error al listar threads: {e}{RESET}\n")


def show_history(agent, thread_id: str):
    from memory.store import get_thread_history
    history = get_thread_history(agent, thread_id, max_steps=15)
    if not history:
        print(f"  {DIM}Sin historial para '{thread_id}'.{RESET}\n")
        return
    print(f"\n{BOLD}Historial de '{CYAN}{thread_id}{RESET}{BOLD}':{RESET}")
    for step in history:
        score = step.get("quality_score")
        score_str = f"{score:.2f}" if score is not None else "  — "
        sc = GREEN if (score or 0) >= 0.8 else YELLOW if (score or 0) >= 0.6 else RED
        print(f"  {DIM}paso {step['step']:2d}{RESET}  →  "
              f"{WHITE}{step['next_node']:<15}{RESET}  "
              f"score={sc}{score_str}{RESET}  "
              f"{DIM}msgs={step['message_count']}{RESET}")
    print()


# ─── Entrypoint ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DocOps Agent CLI con HITL")
    parser.add_argument("--thread", default=None, help="ID del thread a retomar")
    parser.add_argument("--db", default=DEFAULT_DB, help="Ruta al archivo SQLite")
    args = parser.parse_args()

    db_path = args.db
    banner(db_path)

    # Abrir conexión SQLite persistente (sobrevive entre ejecuciones)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cp = SqliteSaver(conn)
    cp.setup()

    # Construir agente con el checkpointer SQLite
    agent = build_docops_agent(cp=cp)

    # Thread inicial
    thread_id = args.thread or f"docops-{uuid.uuid4().hex[:8]}"
    print(f"Thread activo: {CYAN}{BOLD}{thread_id}{RESET}\n")

    try:
        while True:
            try:
                raw = input(f"{BOLD}[{thread_id[:20]}]> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}Saliendo...{RESET}")
                break

            if not raw:
                continue

            # ─── Comandos ───
            if raw == "/exit":
                print(f"{DIM}¡Hasta luego!{RESET}")
                break

            elif raw == "/new":
                thread_id = f"docops-{uuid.uuid4().hex[:8]}"
                print(f"{GREEN}Nuevo thread: {CYAN}{thread_id}{RESET}\n")

            elif raw.startswith("/thread "):
                thread_id = raw.split(" ", 1)[1].strip()
                print(f"{GREEN}Thread activo: {CYAN}{thread_id}{RESET}\n")

            elif raw == "/threads":
                list_threads(conn)

            elif raw == "/history":
                show_history(agent, thread_id)

            elif raw.startswith("/"):
                print(f"{DIM}Comando desconocido. Disponibles: "
                      f"/new /thread <id> /threads /history /exit{RESET}\n")

            # ─── Consulta al agente ───
            else:
                try:
                    result = run_query(agent, thread_id, raw)

                    draft = result.get("draft", "")
                    score = result.get("quality_score", 0.0)
                    iterations = result.get("iteration", 0)
                    sc = GREEN if score >= 0.8 else YELLOW if score >= 0.6 else RED

                    print(f"\n{GREEN}{BOLD}{'─'*62}{RESET}")
                    print(f"{GREEN}{BOLD}  Respuesta final{RESET}  "
                          f"{DIM}score={sc}{score:.2f}{RESET}{DIM}  iter={iterations}{RESET}")
                    print(f"{GREEN}{BOLD}{'─'*62}{RESET}")
                    print(f"{WHITE}{draft}{RESET}")
                    print(f"{GREEN}{'─'*62}{RESET}\n")

                except KeyboardInterrupt:
                    print(f"\n{YELLOW}Consulta interrumpida.{RESET}\n")
                except Exception as e:
                    print(f"\n{RED}Error: {e}{RESET}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
