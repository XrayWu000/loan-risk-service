from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.rag.document_loader import LoadedDocument

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_file: str
    section_title: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _source_prefix(source_file: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", source_file.rsplit(".", 1)[0]).strip("_")
    return prefix.lower() or "document"


def _sections_from_markdown(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "文件概述"
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = heading.group(1).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def _split_units(section_text: str) -> list[str]:
    units: list[str] = []
    buffer: list[str] = []

    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if buffer:
                units.append("\n".join(buffer).strip())
                buffer = []
            continue

        is_list_item = re.match(r"^([-*]|\d+\.)\s+", stripped) is not None
        if is_list_item:
            if buffer:
                units.append("\n".join(buffer).strip())
                buffer = []
            units.append(stripped)
        else:
            buffer.append(stripped)

    if buffer:
        units.append("\n".join(buffer).strip())

    return [unit for unit in units if unit]


def _window_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def split_documents(
    documents: list[LoadedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    counters: dict[str, int] = {}

    for document in documents:
        prefix = _source_prefix(document.source_file)
        counters.setdefault(prefix, 0)

        for section_title, section_text in _sections_from_markdown(document.text):
            units = _split_units(section_text)
            current = ""

            for unit in units:
                candidate = f"{current}\n{unit}".strip() if current else unit
                if len(candidate) <= chunk_size:
                    current = candidate
                    continue

                for piece in _window_text(current, chunk_size, chunk_overlap):
                    counters[prefix] += 1
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{prefix}_{counters[prefix]:03d}",
                            source_file=document.source_file,
                            section_title=section_title,
                            text=piece,
                        )
                    )
                current = unit

            if current:
                for piece in _window_text(current, chunk_size, chunk_overlap):
                    counters[prefix] += 1
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{prefix}_{counters[prefix]:03d}",
                            source_file=document.source_file,
                            section_title=section_title,
                            text=piece,
                        )
                    )

    return chunks
