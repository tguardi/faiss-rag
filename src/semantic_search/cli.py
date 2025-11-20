from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from . import data as data_module
from .chunkers import available_chunkers
from .index import (
    build_faiss_index,
    get_artifact_paths,
    load_chunk_metadata,
    load_index,
    load_model,
    search_index,
)
from .rag import BedrockRAGClient, RagSegment, DEFAULT_MODEL_ID


def _detect_repo_root() -> Path:
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fallback to module location (e.g., when installed elsewhere).
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _detect_repo_root()
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "speeches.json"
DEFAULT_CHUNKER = "paragraphs"
DEFAULT_CORPUS_PREFIX = "speeches"


@dataclass
class RagConfig:
    top_k: int
    model_id: str
    region: str | None
    max_output_tokens: int
    temperature: float


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over a FAISS-backed text corpus with optional Bedrock Q&A"
    )
    chunker_options = available_chunkers()
    chunker_help = (
        "Chunking strategy (default: paragraphs). Available: "
        + ", ".join(
            f"{name} ({desc})" for name, desc in chunker_options.items()
        )
        + "."
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the FAISS index from the current data file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return (default: 5).",
    )
    parser.add_argument(
        "--chunker",
        type=str,
        default=DEFAULT_CHUNKER,
        choices=sorted(chunker_options),
        help=chunker_help,
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the JSON corpus (default: data/speeches.json).",
    )
    parser.add_argument(
        "--data-key",
        type=str,
        help=(
            "JSON key that holds the list of documents when the file is an object "
            "(default: auto-detect)."
        ),
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        help=(
            "Prefix for FAISS/metadata filenames. Defaults to the data-file name "
            "(or 'speeches' for the built-in dataset)."
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Execute a single semantic search query.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive REPL for repeated search queries.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help=(
            "Call Amazon Bedrock (Anthropic Claude Sonnet) to craft a response "
            "from the retrieved chunks (requires AWS credentials)."
        ),
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=4,
        help="How many chunks to pass to the Bedrock model when --rag is enabled (default: 4).",
    )
    parser.add_argument(
        "--bedrock-model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Bedrock model ID to invoke (default: Anthropic Claude Sonnet 4.5).",
    )
    parser.add_argument(
        "--bedrock-region",
        type=str,
        help="AWS region for Bedrock (defaults to your AWS configuration).",
    )
    parser.add_argument(
        "--bedrock-max-output",
        type=int,
        default=600,
        help="Maximum output tokens for the Bedrock response (default: 600).",
    )
    parser.add_argument(
        "--bedrock-temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for the Bedrock response (default: 0.1).",
    )

    args = parser.parse_args(argv)

    if args.rag and args.rag_top_k <= 0:
        parser.error("--rag-top-k must be a positive integer.")
    if args.rag and not args.query:
        parser.error("--rag currently requires --query (non-interactive mode).")
    if args.rag and args.interactive:
        parser.error("--rag cannot be combined with --interactive mode yet.")

    data_file = _resolve_data_file(args.data_file)
    corpus_prefix = _determine_corpus_prefix(data_file, args.artifact_prefix)
    documents = _load_documents_or_exit(data_file, args.data_key)

    artifact_paths = ensure_data(
        rebuild_index=args.rebuild_index,
        chunker_name=args.chunker,
        documents=documents,
        data_file=data_file,
        corpus_prefix=corpus_prefix,
    )

    if not args.query and not args.interactive:
        print("Index is ready. Use --query or --interactive to search documents.")
        return

    chunks = load_chunk_metadata(artifact_paths["chunks"])
    index = load_index(artifact_paths["index"])
    model = load_model()

    rag_config = None
    if args.rag:
        rag_config = RagConfig(
            top_k=args.rag_top_k,
            model_id=args.bedrock_model,
            region=args.bedrock_region,
            max_output_tokens=args.bedrock_max_output,
            temperature=args.bedrock_temperature,
        )

    if args.query:
        _run_single_query(
            args.query,
            documents,
            chunks,
            model,
            index,
            args.top_k,
            rag_config=rag_config,
        )

    if args.interactive:
        _interactive_loop(documents, chunks, model, index, args.top_k)


def ensure_data(
    *,
    rebuild_index: bool,
    chunker_name: str,
    documents,
    data_file: Path,
    corpus_prefix: str,
):
    """Ensure FAISS index exists for the requested corpus and chunker."""

    artifact_paths = get_artifact_paths(
        DATA_DIR,
        chunker_name,
        corpus_prefix=corpus_prefix,
    )

    if not data_file.exists():
        raise SystemExit(
            f"Data file not found at {data_file}. "
            "Provide --data-file pointing to a valid JSON corpus."
        )

    index_missing = (
        not artifact_paths["index"].exists()
        or not artifact_paths["chunks"].exists()
    )

    if rebuild_index or index_missing:
        action = "Rebuilding" if rebuild_index else "Building"
        print(f"{action} FAISS index from {data_file}...")
        build_faiss_index(
            documents,
            DATA_DIR,
            chunker_name=chunker_name,
            corpus_prefix=corpus_prefix,
        )

    return artifact_paths


def _resolve_data_file(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def _determine_corpus_prefix(data_file: Path, override: str | None) -> str:
    if override:
        trimmed = override.strip()
        if trimmed:
            return trimmed
    try:
        if data_file.resolve() == DEFAULT_DATASET_PATH.resolve():
            return DEFAULT_CORPUS_PREFIX
    except FileNotFoundError:
        # Path may not exist yet; fall through to stem-based prefix.
        pass
    return data_file.stem or DEFAULT_CORPUS_PREFIX


def _load_documents_or_exit(path: Path, dataset_key: str | None):
    try:
        return data_module.load_documents(path, dataset_key=dataset_key)
    except FileNotFoundError as exc:
        raise SystemExit(f"Data file not found: {path}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _run_single_query(
    query: str,
    documents,
    chunks,
    model,
    index,
    top_k: int,
    *,
    rag_config: RagConfig | None = None,
) -> None:
    print(f"Query: {query}")
    retrieval_k = top_k
    if rag_config is not None:
        retrieval_k = max(retrieval_k, rag_config.top_k)
    scores, indices = search_index(query, index, model, top_k=retrieval_k)
    display_scores = scores[:top_k]
    display_indices = indices[:top_k]
    _print_results(display_scores, display_indices, documents, chunks)

    if rag_config is not None:
        contexts = _prepare_rag_segments(
            scores,
            indices,
            documents,
            chunks,
            rag_config.top_k,
        )
        if not contexts:
            print(
                "No retrieved chunks available for RAG answer. "
                "Try increasing --rag-top-k."
            )
            return
        rag_client = BedrockRAGClient(
            model_id=rag_config.model_id,
            region=rag_config.region,
        )
        try:
            answer = rag_client.generate_response(
                question=query,
                contexts=contexts,
                temperature=rag_config.temperature,
                max_output_tokens=rag_config.max_output_tokens,
            )
        except Exception as exc:  # pragma: no cover - surface error to user
            print(f"Bedrock request failed: {exc}")
            return
        _print_rag_answer(answer)


def _interactive_loop(documents, chunks, model, index, top_k: int) -> None:
    print("Entering interactive mode. Type 'exit' or 'quit' to leave.")
    try:
        while True:
            try:
                query = input("query> ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            scores, indices = search_index(query, index, model, top_k=top_k)
            _print_results(scores, indices, documents, chunks)
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")


def _print_results(scores, indices, documents, chunks) -> None:
    print("-" * 60)
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        document_idx = chunk.get("document_index")
        if document_idx is None:
            document_idx = chunk.get("speech_index", 0)
        if document_idx < 0 or document_idx >= len(documents):
            continue
        document = documents[document_idx]
        snippet_text = chunk.get("text", document.content).strip()
        chunk_number = chunk.get("chunk_index", 0) + 1
        attribution = document.attribution
        date_label = document.display_date
        url = document.link
        print(
            f"[{rank}] {document.title} — {attribution} "
            f"({date_label}) [chunk {chunk_number}]"
        )
        print(f"Score: {score:.3f}")
        if url:
            print(f"URL: {url}")
        print("Snippet:")
        print(snippet_text)
        print("-" * 60)


def _prepare_rag_segments(
    scores,
    indices,
    documents,
    chunks,
    limit: int,
) -> list[RagSegment]:
    segments: list[RagSegment] = []
    if limit <= 0:
        return segments
    for score, idx in zip(scores, indices):
        if len(segments) >= limit:
            break
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        document_idx = chunk.get("document_index")
        if document_idx is None:
            document_idx = chunk.get("speech_index", 0)
        if document_idx < 0 or document_idx >= len(documents):
            continue
        document = documents[document_idx]
        text = chunk.get("text", document.content).strip()
        segments.append(
            RagSegment(
                title=document.title,
                attribution=document.attribution,
                date=document.display_date,
                url=document.link,
                chunk_index=chunk.get("chunk_index", 0),
                score=float(score),
                text=text,
            )
        )
    return segments


def _print_rag_answer(answer: str) -> None:
    print("\nBedrock Answer")
    print("-" * 60)
    print(answer.strip())
    print("-" * 60)


if __name__ == "__main__":
    main()
