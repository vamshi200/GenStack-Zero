import faiss
import numpy as np


class VectorStore:
    """Simple FAISS vector store for text chunks."""

    def __init__(self, dimension: int) -> None:
        # IndexFlatL2 is beginner-friendly and uses exact nearest-neighbor search.
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: list[str] = []

    def add(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Add text chunks and their embeddings to the FAISS index."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        vectors = np.array(embeddings, dtype="float32")

        # FAISS stores the vectors, while this list keeps the original text.
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        """Return the most similar text chunks for a query embedding."""
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding], dtype="float32")
        _, indices = self.index.search(query_vector, top_k)

        # FAISS returns -1 when it cannot fill all requested results.
        return [self.texts[index] for index in indices[0] if index != -1]
