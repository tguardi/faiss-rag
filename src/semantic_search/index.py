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
from .data import Document


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class IndexArtifacts:
    index_path: Path
    metadata_path: Path
    chunk_metadata_path: Path
    model_name: str
    chunker_name: str
    corpus_prefix: str


def build_faiss_index(
    documents: Sequence[Document],
    data_dir: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    chunker_name: str = "full",
    corpus_prefix: str = "speeches",
) -> IndexArtifacts:
    """Encode the provided documents and persist a FAISS index to disk."""
    chunker = get_chunker(chunker_name)
    chunks = chunker.chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks were produced from the provided documents.")

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

    paths = get_artifact_paths(data_dir, chunker_name, corpus_prefix=corpus_prefix)
    faiss.write_index(index, str(paths["index"]))
    print(f"FAISS index written to {paths['index']}")

    metadata = {
        "model_name": model_name,
        "vector_dim": embeddings.shape[1],
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunker": chunker_name,
        "corpus_prefix": corpus_prefix,
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
        corpus_prefix=corpus_prefix,
    )


def load_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load the embedding model, checking for a local copy first.

    Priority:
    1. Check models/ directory in project root
    2. Fall back to Hugging Face cache (auto-download if needed)

    This allows for portable, offline-capable deployments.
    """
    from pathlib import Path

    # Check for local model in models/ directory
    project_root = Path(__file__).parent.parent.parent
    local_model_path = project_root / "models" / model_name.split("/")[-1]

    if local_model_path.exists() and local_model_path.is_dir():
        print(f"Loading model from local path: {local_model_path}")
        return SentenceTransformer(str(local_model_path), device="cpu")

    # Fall back to Hugging Face cache (will download if not cached)
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
    """
    Create a FAISS index using HNSW with cosine similarity.

    HNSW (Hierarchical Navigable Small World) provides fast approximate
    nearest neighbor search with good recall. Since embeddings are normalized,
    Inner Product (IP) is equivalent to cosine similarity.

    Parameters:
    - M=32: Number of bi-directional links per node (higher = better recall, more memory)
    - efConstruction=64: Size of dynamic candidate list during construction (higher = better quality)
    - efSearch=64: Size of dynamic candidate list during search (set at search time if needed)
    """
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    # Create HNSW index with Inner Product (cosine similarity for normalized vectors)
    M = 32  # Number of connections per layer
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)

    # Set construction parameters for quality
    index.hnsw.efConstruction = 64

    # Add vectors to the index
    index.add(embeddings)

    # Set search-time parameter (can be adjusted at query time for speed/quality tradeoff)
    index.hnsw.efSearch = 64

    return index


def get_artifact_paths(
    data_dir: Path,
    chunker_name: str,
    corpus_prefix: str = "speeches",
) -> Dict[str, Path]:
    chunker_safe = _sanitize(chunker_name)
    corpus_safe = _sanitize(corpus_prefix or "speeches")

    if corpus_safe == "speeches":
        index_name = f"speeches_{chunker_safe}.faiss"
        metadata_name = f"index_meta_{chunker_safe}.json"
        chunk_name = f"chunks_{chunker_safe}.json"
    else:
        index_name = f"{corpus_safe}_{chunker_safe}.faiss"
        metadata_name = f"index_meta_{corpus_safe}_{chunker_safe}.json"
        chunk_name = f"chunks_{corpus_safe}_{chunker_safe}.json"

    return {
        "index": data_dir / index_name,
        "metadata": data_dir / metadata_name,
        "chunks": data_dir / chunk_name,
    }


def load_chunk_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk metadata missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_chunk_metadata(path: Path, chunks: Sequence[Chunk]) -> None:
    serialised = [
        {
            "chunk_id": chunk.chunk_id,
            "document_index": chunk.document_index,
            "speech_index": chunk.document_index,  # backward compatibility for existing tooling
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
