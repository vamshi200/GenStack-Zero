from pathlib import Path


def load_text_file(file_path: str) -> str:
    """Load a single text file and return its contents."""
    path = Path(file_path)

    # UTF-8 is a common default for text documents.
    return path.read_text(encoding="utf-8")


def load_documents(folder_path: str) -> list[str]:
    """Load all .txt files from a folder."""
    folder = Path(folder_path)
    documents: list[str] = []

    # Sort files so results are predictable every time you run the code.
    for file_path in sorted(folder.glob("*.txt")):
        documents.append(load_text_file(str(file_path)))

    return documents
