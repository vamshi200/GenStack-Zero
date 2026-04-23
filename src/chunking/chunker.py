def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into small overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0

    # Move through the text in steps while keeping a small overlap.
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def chunk_documents(
    documents: list[str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split many documents into one list of chunks."""
    all_chunks: list[str] = []

    # Keep this loop simple so it is easy to follow and debug.
    for document in documents:
        all_chunks.extend(chunk_text(document, chunk_size, overlap))

    return all_chunks
