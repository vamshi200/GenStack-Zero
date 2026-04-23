from sentence_transformers import SentenceTransformer


class Embedder:
    """Small wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # The model is downloaded automatically the first time it is used.
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a list of text strings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Convert numpy arrays to regular Python lists for easier handling.
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Create one embedding for a search query."""
        return self.embed_texts([query])[0]
