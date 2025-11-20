from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .data import Document


@dataclass
class Chunk:
    document_index: int
    chunk_index: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.document_index}-{self.chunk_index}"


class BaseChunker:
    name: str = "base"
    description: str = "Base chunker interface."

    def chunk_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        raise NotImplementedError

    def chunk_speeches(self, speeches: Sequence[Document]) -> List[Chunk]:  # pragma: no cover - backward compat
        return self.chunk_documents(speeches)


class FullSpeechChunker(BaseChunker):
    name = "full"
    description = "Treat each document as a single chunk."

    def chunk_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for idx, document in enumerate(documents):
            text = (document.content or "").strip()
            if not text:
                continue
            chunks.append(Chunk(document_index=idx, chunk_index=0, text=text))
        return chunks


class ParagraphChunker(BaseChunker):
    name = "paragraphs"
    description = "Group paragraphs into chunks between ~450-1100 characters."

    def __init__(self, min_chars: int = 450, max_chars: int = 1100):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for idx, document in enumerate(documents):
            paragraphs = [
                paragraph.strip()
                for paragraph in (document.content or "").split("\n\n")
                if paragraph.strip()
            ]
            if not paragraphs:
                continue
            current: List[str] = []
            current_len = 0
            chunk_idx = 0

            def flush() -> None:
                nonlocal current, current_len, chunk_idx
                if not current:
                    return
                text = "\n\n".join(current).strip()
                if text:
                    chunks.append(Chunk(document_index=idx, chunk_index=chunk_idx, text=text))
                    chunk_idx += 1
                current = []
                current_len = 0

            for paragraph in paragraphs:
                paragraph_len = len(paragraph)
                next_len = current_len + (2 if current else 0) + paragraph_len

                if next_len <= self.max_chars or not current:
                    current.append(paragraph)
                    current_len = next_len
                    if current_len >= self.min_chars:
                        flush()
                else:
                    flush()
                    current.append(paragraph)
                    current_len = len(paragraph)

            if current:
                flush()

        return chunks


class SlidingWindowChunker(BaseChunker):
    name = "sliding_window"
    description = "Token-based sliding windows (150 words, 30 word overlap)."

    def __init__(self, window_size: int = 150, overlap: int = 30):
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")
        self.window_size = window_size
        self.overlap = overlap

    def chunk_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        step = self.window_size - self.overlap
        for idx, document in enumerate(documents):
            words = (document.content or "").split()
            if not words:
                continue
            chunk_idx = 0
            for start in range(0, len(words), step):
                window_words = words[start : start + self.window_size]
                if not window_words:
                    continue
                text = " ".join(window_words)
                chunks.append(Chunk(document_index=idx, chunk_index=chunk_idx, text=text))
                chunk_idx += 1
                if len(window_words) < self.window_size:
                    break
        return chunks


def get_chunker(name: str) -> BaseChunker:
    normalized = name.lower().strip()
    registry = _build_registry()
    if normalized not in registry:
        raise ValueError(f"Unknown chunker '{name}'. Available: {', '.join(sorted(registry))}")
    chunker_factory = registry[normalized]
    return chunker_factory()


def available_chunkers() -> Dict[str, str]:
    registry = _build_registry()
    return {name: factory().description for name, factory in registry.items()}


def _build_registry() -> Dict[str, Callable[[], BaseChunker]]:
    return {
        FullSpeechChunker.name: FullSpeechChunker,
        ParagraphChunker.name: ParagraphChunker,
        SlidingWindowChunker.name: SlidingWindowChunker,
    }
