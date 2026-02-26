from sentence_transformers import SentenceTransformer
import numpy as np

# Cargar modelo multilingüe (primera vez descarga ~500MB)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def get_embedding(text: str) -> list[float]:
    """Genera el embedding de un texto usando el modelo multilingüe."""
    return model.encode(text).tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos en batch."""
    return model.encode(texts).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
