from __future__ import annotations

import json
import os
from pathlib import Path

from config.path_config import FAISS_INDEX_FILE, RAG_CHUNKS_FILE, RAG_INDEX_DIR, RAG_METADATA_FILE
from app.rag.document_loader import load_documents
from app.rag.text_splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_documents

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index(
    embedding_model: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, object]:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "sentence-transformers and faiss-cpu are required. Install requirements.txt first."
        ) from exc

    embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    documents = load_documents()
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were created from the source documents.")

    model = SentenceTransformer(embedding_model)
    texts = [f"{chunk.section_title}\n{chunk.text}" for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir = Path(RAG_INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    with open(RAG_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump([chunk.to_dict() for chunk in chunks], f, ensure_ascii=False, indent=2)

    metadata = {
        "embedding_model": embedding_model,
        "similarity": "cosine",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
        "indexed_fields": ["section_title", "text"],
    }
    with open(RAG_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def main() -> None:
    metadata = build_index()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
