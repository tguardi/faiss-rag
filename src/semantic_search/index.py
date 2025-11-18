from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Dict, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from .chunkers import Chunk, get_chunker
from .data import Speech


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class IndexArtifacts:
    index_path: Path
    metadata_path: Path
    chunk_metadata_path: Path
    model_name: str
    chunker_name: str


def build_faiss_index(
    speeches: Sequence[Speech],
    data_dir: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    chunker_name: str = "full",
) -> IndexArtifacts:
    """Encode the provided speeches and persist a FAISS index to disk."""
    chunker = get_chunker(chunker_name)
    chunks = chunker.chunk_speeches(speeches)
    if not chunks:
        raise ValueError("No chunks were produced from the provided speeches.")

    texts = [chunk.text for chunk in chunks]

    model = load_model(model_name)
    print(f"Encoding {len(texts)} chunks with {model_name} on CPU...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print("Encoding complete. Building FAISS index...")

    index = _create_index(embeddings)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = get_artifact_paths(data_dir, chunker_name)
    faiss.write_index(index, str(paths["index"]))
    print(f"FAISS index written to {paths['index']}")

    metadata = {
        "model_name": model_name,
        "vector_dim": embeddings.shape[1],
        "speech_count": len(speeches),
        "chunk_count": len(chunks),
        "chunker": chunker_name,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_chunk_metadata(paths["chunks"], chunks)

    return IndexArtifacts(
        index_path=paths["index"],
        metadata_path=paths["metadata"],
        chunk_metadata_path=paths["chunks"],
        model_name=model_name,
        chunker_name=chunker_name,
    )


def load_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    # Force CPU execution per requirements.
    return SentenceTransformer(model_name, device="cpu")


def search_index(
    query: str,
    index: faiss.Index,
    model: SentenceTransformer,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FAISS distances and indices for a single query string."""
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)
    return scores[0], indices[0]


def _create_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def get_artifact_paths(data_dir: Path, chunker_name: str) -> Dict[str, Path]:
    safe = _sanitize(chunker_name)
    return {
        "index": data_dir / f"speeches_{safe}.faiss",
        "metadata": data_dir / f"index_meta_{safe}.json",
        "chunks": data_dir / f"chunks_{safe}.json",
    }


def load_chunk_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk metadata missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_chunk_metadata(path: Path, chunks: Sequence[Chunk]) -> None:
    serialised = [
        {
            "chunk_id": chunk.chunk_id,
            "speech_index": chunk.speech_index,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    path.write_text(json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower())
    safe = safe.strip("_")
    return safe or "default"
