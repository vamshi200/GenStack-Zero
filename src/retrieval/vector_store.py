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
        results = self.search_with_scores(query_embedding, top_k=top_k)

        return [result["chunk"] for result in results]

    def search_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict[str, float | str]]:
        """Return matching chunks together with their FAISS distance scores."""
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)

        results: list[dict[str, float | str]] = []

        # Smaller L2 distance means the chunk is closer to the question.
        for distance, index in zip(distances[0], indices[0]):
            if index == -1:
                continue

            results.append(
                {
                    "chunk": self.texts[index],
                    "score": float(distance),
                }
            )

        return results
