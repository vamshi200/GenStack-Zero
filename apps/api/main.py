from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.embeddings.embedder import Embedder
from src.ingestion.load_documents import load_text_file
from src.retrieval.vector_store import VectorStore


# Create the FastAPI application instance.
app = FastAPI(title="GenStack-Zero API")

# These globals keep the demo simple: the app builds one in-memory index.
embedder: Embedder | None = None
vector_store: VectorStore | None = None
tokenizer = None
generation_model = None


class QueryRequest(BaseModel):
    """Expected JSON body for the /query endpoint."""

    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    """JSON response returned by the /query endpoint."""

    question: str
    answer: str
    chunks: list[str]
    count: int


def build_rag_index() -> None:
    """Load documents, chunk them, embed them, and store them in FAISS."""
    global embedder, vector_store, tokenizer, generation_model

    project_root = Path(__file__).resolve().parents[2]
    sample_file = project_root / "data" / "raw" / "sample_docs.txt"

    # Load the sample text file as one document.
    document = load_text_file(str(sample_file))

    # Split the sample file by paragraph so each topic stays easy to retrieve.
    chunks = [chunk.strip() for chunk in document.split("\n\n") if chunk.strip()]

    # Create embeddings for every chunk.
    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)

    # FAISS needs to know the embedding size before creating an index.
    embedding_dimension = len(embeddings[0])
    vector_store = VectorStore(dimension=embedding_dimension)
    vector_store.add(chunks, embeddings)

    # flan-t5-small is a lightweight open-source model that follows prompts well.
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def create_prompt(question: str, chunks: list[str]) -> str:
    """Build the prompt that sends retrieved context to the generator."""
    context = "\n\n".join(chunks)

    return (
        "Answer the question using the context below:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(question: str, chunks: list[str]) -> str:
    """Generate a short answer from the retrieved chunks."""
    if tokenizer is None or generation_model is None:
        # This should not happen after startup, but it keeps the error clear.
        raise HTTPException(status_code=503, detail="Text generator is not ready")

    prompt = create_prompt(question, chunks)

    # Convert the prompt into token IDs that the model can understand.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Keep generation short so the demo responds quickly.
    outputs = generation_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
    )

    # Decode token IDs back into readable text.
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return answer or "I could not generate an answer from the retrieved context."


@app.on_event("startup")
def startup() -> None:
    """Build the simple RAG index when the API starts."""
    build_rag_index()


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health check endpoint."""
    # This endpoint is useful for confirming the API is running.
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    """Retrieve chunks and generate a short answer for a user question."""
    question = request.question.strip()

    if not question:
        # A blank question cannot be embedded or searched in a useful way.
        raise HTTPException(status_code=400, detail="question cannot be empty")

    if embedder is None or vector_store is None:
        # This should not happen after startup, but it keeps the error clear.
        raise HTTPException(status_code=503, detail="RAG index is not ready")

    # Embed the question and search for the closest chunks in FAISS.
    question_embedding = embedder.embed_query(question)
    chunks = vector_store.search(question_embedding, top_k=request.top_k)
    answer = generate_answer(question, chunks)

    return QueryResponse(
        question=question,
        answer=answer,
        chunks=chunks,
        count=len(chunks),
    )


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> QueryResponse:
    """Alias endpoint for simple RAG retrieval."""
    # Reuse /query so both endpoints behave the same way.
    return query(request)
