from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config.path_config import FAISS_INDEX_FILE, RAG_CHUNKS_FILE, RAG_METADATA_FILE
from app.rag.index_builder import DEFAULT_EMBEDDING_MODEL


class RagRetriever:
    def __init__(self) -> None:
        self._index = None
        self._chunks: list[dict[str, Any]] | None = None
        self._model = None
        self._embedding_model_name: str | None = None

    def _load(self) -> None:
        if self._index is not None and self._chunks is not None and self._model is not None:
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "sentence-transformers and faiss-cpu are required. Install requirements.txt first."
            ) from exc

        if not Path(FAISS_INDEX_FILE).exists() or not Path(RAG_CHUNKS_FILE).exists():
            raise FileNotFoundError(
                "RAG index not found. Please run: python -m app.rag.index_builder"
            )

        metadata: dict[str, Any] = {}
        if Path(RAG_METADATA_FILE).exists():
            with open(RAG_METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        embedding_model = metadata.get("embedding_model") or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )

        self._index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(RAG_CHUNKS_FILE, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        self._model = SentenceTransformer(embedding_model)
        self._embedding_model_name = embedding_model

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        self._load()
        assert self._index is not None
        assert self._chunks is not None
        assert self._model is not None

        top_k = max(1, min(int(top_k), 10))
        query_embedding = self._model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores, indices = self._index.search(query_embedding, top_k)

        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self._chunks):
                continue
            item = dict(self._chunks[index])
            item["score"] = float(score)
            results.append(item)
        return results


retriever = RagRetriever()
