from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.embeddings.embedder import Embedder
from src.ingestion.load_documents import load_text_file
from src.retrieval.vector_store import VectorStore


# Create the FastAPI application instance.
app = FastAPI(title="GenStack-Zero API")

# These globals keep the demo simple: the app builds one in-memory index.
embedder: Embedder | None = None
vector_store: VectorStore | None = None


class QueryRequest(BaseModel):
    """Expected JSON body for the /query endpoint."""

    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    """JSON response returned by the /query endpoint."""

    question: str
    answer: str
    answer_source: str
    chunks: list[str]
    used_context: str
    count: int


def build_chunks_from_text(text: str, file_name: str) -> list[str]:
    """Turn the master guide into metadata-rich chunks."""
    chunks: list[str] = []
    text_chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    for text_chunk in text_chunks:
        # Keep one simple source header so retrieval stays easy to inspect.
        chunks.append(f"File: {file_name} | Type: .txt\n{text_chunk}")

    return chunks


def build_rag_index() -> dict[str, int]:
    """Load the fixed GenAI guide, embed it, and store it in FAISS."""
    global embedder, vector_store

    project_root = Path(__file__).resolve().parents[2]
    guide_file = project_root / "data" / "raw" / "genai_master_guide.txt"

    # This system now uses one curated Generative AI knowledge file.
    guide_documents = load_text_file(str(guide_file))
    guide_text = guide_documents[0]["text"]
    chunks = build_chunks_from_text(guide_text, guide_file.name)

    if not chunks:
        raise ValueError("The GenAI master guide did not contain any readable text")

    # Create embeddings for every chunk.
    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)

    # FAISS needs to know the embedding size before creating an index.
    embedding_dimension = len(embeddings[0])
    vector_store = VectorStore(dimension=embedding_dimension)
    vector_store.add(chunks, embeddings)

    return {
        "document_count": 1,
        "chunk_count": len(chunks),
    }


def build_context(chunks: list[str]) -> str:
    """Join retrieved chunks into the exact context sent to the model."""
    return "\n\n".join(chunks)


def build_not_enough_context_answer() -> str:
    """Return a safe message when the knowledge base cannot support an answer."""
    return (
        "The knowledge base does not contain enough information to answer this "
        "question in a reliable way."
    )


def build_out_of_scope_answer() -> str:
    """Return the fixed message for non-GenAI questions."""
    return "This system is designed to answer only Generative AI related questions."


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


def extract_question_keywords(question: str) -> tuple[list[str], list[str]]:
    """Split a question into broad keywords and more topic-specific keywords."""
    stop_words = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "does",
        "do",
        "is",
        "are",
        "the",
        "a",
        "an",
        "in",
        "on",
        "of",
        "to",
        "for",
        "with",
        "and",
        "or",
        "about",
        "detail",
        "detailed",
        "simple",
        "example",
        "explain",
    }
    generic_domain_words = {
        "ai",
        "genai",
        "model",
        "models",
        "language",
        "system",
        "systems",
        "application",
        "applications",
    }

    all_keywords = [
        word.strip(".,?!:;").lower()
        for word in question.split()
        if len(word.strip(".,?!:;")) > 2
    ]
    meaningful_keywords = [word for word in all_keywords if word not in stop_words]
    specific_keywords = [
        word for word in meaningful_keywords if word not in generic_domain_words
    ]

    return meaningful_keywords, specific_keywords


def is_genai_question(question: str) -> bool:
    """Check whether a question is about Generative AI topics."""
    genai_keywords = {
        "generative",
        "genai",
        "llm",
        "llms",
        "token",
        "tokens",
        "tokenization",
        "embedding",
        "embeddings",
        "vector",
        "vectors",
        "database",
        "databases",
        "rag",
        "chunking",
        "retrieval",
        "reranking",
        "prompt",
        "prompting",
        "hallucination",
        "guardrails",
        "fine-tuning",
        "finetuning",
        "peft",
        "lora",
        "qlora",
        "dora",
        "quantization",
        "gpu",
        "gpus",
        "vram",
        "cuda",
        "inference",
        "batching",
        "fastapi",
        "streamlit",
        "faiss",
        "langchain",
        "llamaindex",
        "agent",
        "agents",
        "multimodal",
        "transformer",
        "transformers",
        "copilot",
    }

    lower_question = question.lower()
    meaningful_keywords, specific_keywords = extract_question_keywords(question)

    if any(keyword in lower_question for keyword in genai_keywords):
        return True

    return any(keyword in genai_keywords for keyword in meaningful_keywords + specific_keywords)


def extract_fallback_answer(chunks: list[str]) -> str:
    """Use the first relevant chunk as a safe fallback answer."""
    if not chunks:
        return build_not_enough_context_answer()

    # The first chunk is already the closest one from retrieval.
    return chunks[0].strip()


def is_answer_grounded(answer: str, context: str) -> bool:
    """Check whether the generated answer overlaps enough with the context."""
    answer_words = {word.strip(".,?!:;").lower() for word in answer.split()}
    context_words = {word.strip(".,?!:;").lower() for word in context.split()}

    meaningful_answer_words = {word for word in answer_words if len(word) > 3}
    overlap = meaningful_answer_words.intersection(context_words)

    # Require some shared vocabulary so obviously off-context answers are filtered.
    return len(overlap) >= 2


def split_into_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like pieces."""
    normalized_text = text.replace("\n", " ")
    sentences: list[str] = []
    current_sentence = ""

    for character in normalized_text:
        current_sentence += character
        if character in ".!?":
            cleaned_sentence = current_sentence.strip()
            if cleaned_sentence:
                sentences.append(cleaned_sentence)
            current_sentence = ""

    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    return sentences


def clean_chunk_text(chunk: str) -> str:
    """Remove the metadata header line from a retrieved chunk."""
    lines = chunk.splitlines()
    if len(lines) <= 1:
        return chunk.strip()

    return "\n".join(lines[1:]).strip()


def collect_context_sentences(chunks: list[str]) -> list[str]:
    """Collect readable sentences from retrieved chunks."""
    sentences: list[str] = []

    for chunk in chunks:
        chunk_text = clean_chunk_text(chunk)
        sentences.extend(split_into_sentences(chunk_text))

    return sentences


def select_sentences(
    sentences: list[str],
    keywords: list[str],
    max_sentences: int = 3,
) -> list[str]:
    """Pick sentences that best match a small set of topic keywords."""
    selected: list[str] = []

    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(keyword in lower_sentence for keyword in keywords):
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    # Fall back to the earliest sentences if no keyword match is found.
    if not selected:
        selected = sentences[:max_sentences]

    return selected


def format_section(title: str, paragraphs: list[str]) -> str:
    """Format one answer section with a markdown heading."""
    body = " ".join(paragraphs).strip()

    if not body:
        body = (
            "The knowledge base needs more information to explain this part in detail."
        )

    return f"**{title}**\n{body}"


def build_step_by_step_section(sentences: list[str]) -> str:
    """Create a simple step-by-step explanation from retrieved context."""
    steps = [
        "The system starts with a user question and turns it into an embedding so it can search semantically.",
        "It searches the indexed knowledge base to find the chunks that are most relevant to the question.",
        "Those chunks are combined into context and used as the main evidence for the final answer.",
        "The answer generator then explains the topic using that grounded context instead of guessing from memory alone.",
    ]

    matching_sentences = select_sentences(
        sentences,
        keywords=["retrieve", "retrieval", "context", "embedding", "search", "chunk"],
        max_sentences=2,
    )
    if matching_sentences:
        steps.extend(matching_sentences)

    numbered_steps = [f"{index}. {step}" for index, step in enumerate(steps, start=1)]

    return "**How It Works Step by Step**\n" + "\n".join(numbered_steps)


def build_detailed_answer(question: str, chunks: list[str]) -> str:
    """Build a long, structured answer directly from retrieved context."""
    context = build_context(chunks)
    sentences = collect_context_sentences(chunks)

    if len(sentences) < 4:
        return ""

    topic_keywords, specific_keywords = extract_question_keywords(question)

    definition_sentences = select_sentences(
        sentences,
        keywords=topic_keywords + ["is", "are", "called"],
        max_sentences=2,
    )
    importance_sentences = select_sentences(
        sentences,
        keywords=["important", "useful", "matters", "reduce", "improve", "performance"],
        max_sentences=3,
    )
    detail_sentences = select_sentences(
        sentences,
        keywords=topic_keywords + ["works", "context", "model", "retrieval", "generation"],
        max_sentences=4,
    )
    example_sentences = select_sentences(
        sentences,
        keywords=["example", "for example", "customer support", "assistant", "workflow"],
        max_sentences=3,
    )
    application_sentences = select_sentences(
        sentences,
        keywords=["businesses", "teams", "enterprise", "applications", "used", "product"],
        max_sentences=3,
    )
    genai_sentences = select_sentences(
        sentences,
        keywords=["generative ai", "llm", "hallucination", "grounded", "runtime", "knowledge base"],
        max_sentences=3,
    )

    answer_sections = [
        format_section("Simple Definition", definition_sentences),
        format_section("Detailed Explanation", detail_sentences + importance_sentences[:2]),
        build_step_by_step_section(sentences),
        format_section("Simple Example", example_sentences),
        format_section("Real-World Applications", application_sentences),
        format_section("Why It Matters in Generative AI", genai_sentences + importance_sentences[:1]),
    ]

    answer = "\n\n".join(answer_sections)

    if not is_answer_grounded(answer, context):
        return ""

    return answer


def has_strong_topic_coverage(question: str, chunks: list[str]) -> bool:
    """Check whether the retrieved context really covers the main topic."""
    context_text = build_context(chunks).lower()
    meaningful_keywords, specific_keywords = extract_question_keywords(question)

    if specific_keywords:
        matched_specific_keywords = [
            keyword for keyword in specific_keywords if keyword in context_text
        ]
        if len(specific_keywords) >= 2:
            return len(matched_specific_keywords) >= 2

        return len(matched_specific_keywords) >= 1

    matched_meaningful_keywords = [
        keyword for keyword in meaningful_keywords if keyword in context_text
    ]

    return len(matched_meaningful_keywords) >= max(1, len(meaningful_keywords) // 2)


def is_strong_context(question: str, results: list[dict[str, float | str]]) -> bool:
    """Decide whether retrieval is strong enough for a knowledge-base answer."""
    chunks = [str(result["chunk"]) for result in results]

    if not has_strong_topic_coverage(question, chunks):
        return False

    if len(results) >= 2:
        return True

    if len(results) == 1 and float(results[0]["score"]) < 0.8:
        return True

    return False


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
    """Retrieve chunks and generate a detailed answer for a user question."""
    question = request.question.strip()

    if not question:
        # A blank question cannot be embedded or searched in a useful way.
        raise HTTPException(status_code=400, detail="question cannot be empty")

    if embedder is None or vector_store is None:
        # This should not happen after startup, but it keeps the error clear.
        raise HTTPException(status_code=503, detail="RAG index is not ready")

    if not is_genai_question(question):
        return QueryResponse(
            question=question,
            answer=build_out_of_scope_answer(),
            answer_source="not_enough_context",
            chunks=[],
            used_context="",
            count=0,
        )

    # Embed the question and search for the closest chunks in FAISS.
    question_embedding = embedder.embed_query(question)
    results = vector_store.search_with_scores(question_embedding, top_k=request.top_k)

    # Keep only chunks that pass a basic relevance check.
    relevant_results = [
        result
        for result in results
        if is_relevant(question, str(result["chunk"]), float(result["score"]))
    ]
    relevant_chunks = [str(result["chunk"]) for result in relevant_results]
    used_context = build_context(relevant_chunks)

    if not relevant_chunks:
        answer = build_not_enough_context_answer()
        answer_source = "not_enough_context"
    elif is_strong_context(question, relevant_results):
        answer = build_detailed_answer(question, relevant_chunks)

        if answer:
            answer_source = "knowledge_base"
        else:
            answer = build_not_enough_context_answer()
            answer_source = "not_enough_context"
    else:
        answer = build_not_enough_context_answer()
        answer_source = "not_enough_context"

    return QueryResponse(
        question=question,
        answer=answer,
        answer_source=answer_source,
        chunks=relevant_chunks,
        used_context=used_context,
        count=len(relevant_chunks),
    )


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> QueryResponse:
    """Alias endpoint for simple RAG retrieval."""
    # Reuse /query so both endpoints behave the same way.
    return query(request)
