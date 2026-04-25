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
    used_context: str
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
    context = build_context(chunks)

    return (
        "Answer the question using only the context below in 2 to 4 sentences. "
        "If the context is not enough, say: I could not find enough context to answer this question.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_context(chunks: list[str]) -> str:
    """Join retrieved chunks into the exact context sent to the model."""
    return "\n\n".join(chunks)


def is_relevant(question: str, chunk: str, score: float) -> bool:
    """Apply a simple relevance check before using a chunk."""
    question_words = {word.strip(".,?!:;").lower() for word in question.split()}
    chunk_words = {word.strip(".,?!:;").lower() for word in chunk.split()}

    # Ignore tiny words so the overlap check stays simple and useful.
    meaningful_question_words = {
        word for word in question_words if len(word) > 2
    }
    overlap = meaningful_question_words.intersection(chunk_words)

    # Lower FAISS distance is better. We also want at least one shared keyword.
    return bool(overlap) and score < 1.5


def limit_answer_sentences(answer: str, max_sentences: int = 4) -> str:
    """Keep only the first few sentences from the generated answer."""
    sentence_endings = ".!?"
    sentences: list[str] = []
    current_sentence = ""

    for character in answer:
        current_sentence += character

        if character in sentence_endings:
            cleaned_sentence = current_sentence.strip()
            if cleaned_sentence:
                sentences.append(cleaned_sentence)
            current_sentence = ""

        if len(sentences) >= max_sentences:
            break

    if not sentences and answer.strip():
        return answer.strip()

    return " ".join(sentences).strip()


def extract_fallback_answer(chunks: list[str]) -> str:
    """Use the first relevant chunk as a safe fallback answer."""
    if not chunks:
        return "I could not find enough context to answer this question."

    # The first chunk is already the closest one from retrieval.
    return limit_answer_sentences(chunks[0], max_sentences=2)


def is_answer_grounded(answer: str, context: str) -> bool:
    """Check whether the generated answer overlaps enough with the context."""
    answer_words = {word.strip(".,?!:;").lower() for word in answer.split()}
    context_words = {word.strip(".,?!:;").lower() for word in context.split()}

    meaningful_answer_words = {word for word in answer_words if len(word) > 3}
    overlap = meaningful_answer_words.intersection(context_words)

    # Require some shared vocabulary so obviously off-context answers are filtered.
    return len(overlap) >= 2


def generate_answer(question: str, chunks: list[str]) -> str:
    """Generate a short answer from the retrieved chunks."""
    if tokenizer is None or generation_model is None:
        # This should not happen after startup, but it keeps the error clear.
        raise HTTPException(status_code=503, detail="Text generator is not ready")

    prompt = create_prompt(question, chunks)
    context = build_context(chunks)

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
    answer = limit_answer_sentences(answer)

    if not answer:
        return extract_fallback_answer(chunks)

    if answer == "I could not find enough context to answer this question.":
        return answer

    if not is_answer_grounded(answer, context):
        return extract_fallback_answer(chunks)

    return answer


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
    results = vector_store.search_with_scores(question_embedding, top_k=request.top_k)

    # Keep only chunks that pass a basic relevance check.
    relevant_chunks = [
        result["chunk"]
        for result in results
        if is_relevant(question, str(result["chunk"]), float(result["score"]))
    ]
    used_context = build_context(relevant_chunks)

    if not relevant_chunks:
        answer = "I could not find enough context to answer this question."
    else:
        answer = generate_answer(question, relevant_chunks)

    return QueryResponse(
        question=question,
        answer=answer,
        chunks=relevant_chunks,
        used_context=used_context,
        count=len(relevant_chunks),
    )


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> QueryResponse:
    """Alias endpoint for simple RAG retrieval."""
    # Reuse /query so both endpoints behave the same way.
    return query(request)
