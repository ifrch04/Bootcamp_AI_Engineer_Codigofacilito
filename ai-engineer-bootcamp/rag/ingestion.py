import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    content: str
    metadata: dict
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Chunk:
    content: str
    metadata: dict
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def load_txt(path: str) -> Document:
    """Carga un archivo .txt como Document."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        content=content,
        metadata={"source": path, "type": "txt"},
    )


def load_pdf(path: str) -> Document:
    """Carga un archivo .pdf como Document usando pypdf."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    content = "\n\n".join(pages)
    return Document(
        content=content,
        metadata={"source": path, "type": "pdf", "pages": len(reader.pages)},
    )


def load_markdown(path: str) -> Document:
    """Carga un archivo .md como Document, removiendo frontmatter YAML si existe."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Remover frontmatter YAML (entre --- al inicio del archivo)
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)
    return Document(
        content=content,
        metadata={"source": path, "type": "md"},
    )


LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".md": load_markdown,
}


def load_document(path: str) -> Document:
    """Carga un documento según su extensión."""
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Extensión no soportada: {ext} (archivo: {path})")
    return loader(path)


def load_directory(dir_path: str) -> list[Document]:
    """Carga todos los documentos soportados de un directorio."""
    documents: list[Document] = []
    for filename in sorted(os.listdir(dir_path)):
        ext = Path(filename).suffix.lower()
        if ext in LOADERS:
            full_path = os.path.join(dir_path, filename)
            documents.append(load_document(full_path))
    return documents


def chunk_by_paragraphs(
    doc: Document, max_chunk_size: int = 800, separator: str = "\n\n"
) -> list[Chunk]:
    """Divide un documento en chunks por párrafos sin cortar a mitad de párrafo."""
    paragraphs = doc.content.split(separator)
    chunks: list[Chunk] = []
    current_chunk = ""
    chunk_index = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Si agregar este párrafo excede el límite y ya tenemos contenido, guardar chunk actual
        if current_chunk and len(current_chunk) + len(separator) + len(paragraph) > max_chunk_size:
            chunks.append(
                Chunk(
                    content=current_chunk.strip(),
                    metadata={**doc.metadata, "chunk_index": chunk_index},
                )
            )
            chunk_index += 1
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += separator + paragraph
            else:
                current_chunk = paragraph

    # Agregar el último chunk si queda contenido
    if current_chunk.strip():
        chunks.append(
            Chunk(
                content=current_chunk.strip(),
                metadata={**doc.metadata, "chunk_index": chunk_index},
            )
        )

    return chunks
