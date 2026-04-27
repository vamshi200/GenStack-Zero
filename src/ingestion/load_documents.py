from pathlib import Path

import fitz


def create_document(
    text: str,
    file_name: str,
    file_type: str,
    page_number: int | None = None,
) -> dict:
    """Create one simple document record with text and metadata."""
    return {
        "text": text,
        "metadata": {
            "file_name": file_name,
            "file_type": file_type,
            "page_number": page_number,
        },
    }


def load_text_file(file_path: str) -> list[dict]:
    """Load one .txt file and return it as a document record."""
    path = Path(file_path)

    # UTF-8 is a common default for text documents.
    text = path.read_text(encoding="utf-8")

    return [
        create_document(
            text=text,
            file_name=path.name,
            file_type=".txt",
        )
    ]


def load_pdf_file(file_path: str) -> list[dict]:
    """Load one PDF file and return one document record per page."""
    path = Path(file_path)
    documents: list[dict] = []

    # PyMuPDF opens the PDF so we can read each page separately.
    with fitz.open(path) as pdf_document:
        for page_index, page in enumerate(pdf_document, start=1):
            page_text = page.get_text().strip()

            if not page_text:
                continue

            documents.append(
                create_document(
                    text=page_text,
                    file_name=path.name,
                    file_type=".pdf",
                    page_number=page_index,
                )
            )

    return documents


def load_documents(folder_path: str) -> list[dict]:
    """Load all supported documents from a folder."""
    folder = Path(folder_path)
    documents: list[dict] = []

    # Sort files so results are predictable every time you run the code.
    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".txt":
            documents.extend(load_text_file(str(file_path)))
        elif file_path.suffix.lower() == ".pdf":
            documents.extend(load_pdf_file(str(file_path)))

    return documents
