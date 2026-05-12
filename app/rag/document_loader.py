from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.path_config import DOCUMENTS_DIR


@dataclass(frozen=True)
class LoadedDocument:
    source_file: str
    text: str


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pypdf is required to read PDF documents.") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[LoadedDocument]:
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {documents_dir}")

    supported = {".md", ".txt", ".pdf"}
    documents: list[LoadedDocument] = []

    for path in sorted(documents_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue

        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8")

        if text.strip():
            documents.append(LoadedDocument(source_file=path.name, text=text.strip()))

    if not documents:
        raise ValueError(f"No readable markdown, text, or PDF documents found in {documents_dir}")

    return documents
