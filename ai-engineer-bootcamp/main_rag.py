import os
import shutil

from dotenv import load_dotenv
from openai import OpenAI

from rag.ingestion import load_directory, chunk_by_paragraphs
from rag.vectorstore import create_vectorstore, index_chunks, search

load_dotenv()

# --- Colores ANSI ---
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"

# --- Configuración del LLM (Groq) ---
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "openai/gpt-oss-120b"


def rag_query(collection, question: str) -> tuple[str, list]:
    """Ejecuta una query RAG: retrieve + generate. Retorna (respuesta, resultados)."""
    # 1. Retrieve
    results = search(collection, question, n_results=3)

    # 2. Construir contexto
    context = "\n\n---\n\n".join(
        [f"[Fuente: {r.metadata['source']}]\n{r.content}" for r in results]
    )

    # 3. Generate con Groq
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responde basándote ÚNICAMENTE en el contexto proporcionado. "
                    "Si no encuentras la respuesta, di 'No tengo información suficiente' "
                    "Responde en español."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}",
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content, results


def print_query_result(question: str, answer: str, results: list) -> None:
    """Imprime los resultados de una query RAG de forma legible."""
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}PREGUNTA: {question}{RESET}")
    print(f"{CYAN}{BOLD}{'='*80}{RESET}")

    print(f"\n{YELLOW}{BOLD}Chunks recuperados:{RESET}")
    for i, r in enumerate(results, 1):
        source = os.path.basename(r.metadata["source"])
        print(f"  {YELLOW}{i}. [{r.score:.3f}] [{source}]{RESET}")
        print(f"     {DIM}{r.content[:150]}...{RESET}\n")

    print(f"{GREEN}{BOLD}RESPUESTA DEL LLM:{RESET}\n{GREEN}{answer}{RESET}")
    print(f"{CYAN}{BOLD}{'='*80}{RESET}\n")


def pause(msg: str = "Presiona Enter para continuar...") -> None:
    """Pausa la ejecución hasta que el usuario presione Enter."""
    input(f"\n{DIM}>>> {msg}{RESET}")


def main() -> None:
    # Limpiar base de datos anterior para empezar limpio
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # Paso 1: Cargar documentos
    print(f"\n{CYAN}{BOLD}{'=' * 80}")
    print(f"PASO 1: Carga de documentos")
    print(f"{'=' * 80}{RESET}")
    documents = load_directory("./data/")
    print(f"\nDocumentos cargados: {BOLD}{len(documents)}{RESET}")
    for doc in documents:
        print(f"  - {doc.metadata['source']} ({len(doc.content)} caracteres)")

    pause("Enter para hacer chunking...")

    # Paso 2: Chunking
    print(f"\n{MAGENTA}{BOLD}{'=' * 80}")
    print(f"PASO 2: Chunking por párrafos (max_chunk_size=800)")
    print(f"{'=' * 80}{RESET}")
    all_chunks = []
    for doc in documents:
        chunks = chunk_by_paragraphs(doc, max_chunk_size=800)
        all_chunks.extend(chunks)
        print(f"  {MAGENTA}{doc.metadata['source']}: {BOLD}{len(chunks)} chunks{RESET}")

    print(f"\n  Total de documentos: {len(documents)}")
    print(f"  Total de chunks: {MAGENTA}{BOLD}{len(all_chunks)}{RESET}")

    pause("Enter para indexar en ChromaDB...")

    # Paso 3: Indexar
    print(f"\n{CYAN}{BOLD}{'=' * 80}")
    print(f"PASO 3: Indexación en ChromaDB")
    print(f"{'=' * 80}{RESET}")
    collection = create_vectorstore("novatech_docs")
    indexed = index_chunks(collection, all_chunks)
    print(f"\n  Indexación completa: {CYAN}{BOLD}{indexed} chunks indexados{RESET}")

    pause("Enter para ejecutar queries RAG...")

    # Paso 4: Queries RAG
    queries = [
        "¿Cuál es el horario de trabajo en NovaTech?",
        "¿Cuántos días de vacaciones me corresponden en mi primer año?",
        "¿Qué equipo de cómputo reciben los desarrolladores?",
        "¿Cuál es el proceso para crear un ticket de soporte técnico?",
        "¿Qué beneficios de seguro médico ofrece la empresa?",
    ]

    for i, question in enumerate(queries, 1):
        print(f"\n{CYAN}{BOLD}{'=' * 80}")
        print(f"QUERY {i}/{len(queries)}: {question}")
        print(f"{'=' * 80}{RESET}")

        answer, results = rag_query(collection, question)

        print(f"\n{YELLOW}{BOLD}Chunks recuperados:{RESET}")
        for j, r in enumerate(results, 1):
            source = os.path.basename(r.metadata["source"])
            print(f"  {YELLOW}{j}. [{r.score:.3f}] [{source}]{RESET}")
            print(f"     {DIM}{r.content[:150]}...{RESET}\n")

        print(f"{GREEN}{BOLD}RESPUESTA DEL LLM:{RESET}\n{GREEN}{answer}{RESET}")

        if i < len(queries):
            pause("Enter para la siguiente query...")

    pause("Enter para la prueba anti-alucinación...")

    # Paso 5: Query anti-alucinación
    question_no_answer = "¿Cuál es la política de stock options para empleados de NovaTech?"

    print(f"\n{RED}{BOLD}{'=' * 80}")
    print(f"QUERY ANTI-ALUCINACION: {question_no_answer}")
    print(f"{'=' * 80}{RESET}")

    answer, results = rag_query(collection, question_no_answer)

    print(f"\n{YELLOW}{BOLD}Chunks recuperados:{RESET}")
    for j, r in enumerate(results, 1):
        source = os.path.basename(r.metadata["source"])
        print(f"  {YELLOW}{j}. [{r.score:.3f}] [{source}]{RESET}")
        print(f"     {DIM}{r.content[:150]}...{RESET}\n")

    print(f"{GREEN}{BOLD}RESPUESTA DEL LLM:{RESET}\n{GREEN}{answer}{RESET}")
    print(f"\n{CYAN}{BOLD}{'=' * 80}")
    print(f"Pipeline RAG completado.")
    print(f"{'=' * 80}{RESET}")


if __name__ == "__main__":
    main()
