#!/usr/bin/env python3
"""
Benchmark different chunking strategies and retrieval parameters.

This script runs the same query across different configurations and
compares the results, helping you understand how parameters affect
search quality.

Usage:
    uv run python examples/benchmark_search.py "inflation expectations"
    uv run python examples/benchmark_search.py "AI" --top-k 10
"""
import argparse
import sys
from pathlib import Path
from textwrap import shorten

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_search.data import load_documents
from semantic_search.chunkers import available_chunkers
from semantic_search.index import (
    build_faiss_index,
    get_artifact_paths,
    load_chunk_metadata,
    load_index,
    load_model,
    search_index,
)


def _resolve_path(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def _determine_prefix(data_path: Path, override: str | None, default: Path) -> str:
    if override:
        trimmed = override.strip()
        if trimmed:
            return trimmed
    try:
        if data_path.resolve() == default.resolve():
            return "speeches"
    except FileNotFoundError:
        pass
    return data_path.stem or "corpus"


def benchmark_chunkers(query, top_k=5, data_file=None, data_key=None, artifact_prefix=None):
    """Run the same query across all chunking strategies."""
    data_dir = Path(__file__).parent.parent / "data"
    default_data_file = data_dir / "speeches.json"
    data_path = _resolve_path(data_file or default_data_file)
    prefix = _determine_prefix(data_path, artifact_prefix, default_data_file)

    documents = load_documents(data_path, dataset_key=data_key)
    model = load_model()
    chunker_names = list(available_chunkers().keys())

    print("=" * 80)
    print(f"BENCHMARKING QUERY: '{query}'")
    print(f"Top-k results: {top_k}")
    print("=" * 80)

    results_by_chunker = {}

    for chunker_name in chunker_names:
        print(f"\n{'=' * 80}")
        print(f"CHUNKER: {chunker_name.upper()}")
        print("=" * 80)

        artifact_paths = get_artifact_paths(
            data_dir,
            chunker_name,
            corpus_prefix=prefix,
        )

        # Build index if it doesn't exist
        if (
            not artifact_paths["index"].exists()
            or not artifact_paths["chunks"].exists()
        ):
            print(f"Building index for {chunker_name} chunker...")
            build_faiss_index(
                documents,
                data_dir,
                chunker_name=chunker_name,
                corpus_prefix=prefix,
            )

        # Load index and search
        chunks = load_chunk_metadata(artifact_paths["chunks"])
        index = load_index(artifact_paths["index"])

        scores, indices = search_index(query, index, model, top_k=top_k)

        # Store and display results
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0 or idx >= len(chunks):
                continue

            chunk = chunks[idx]
            doc_idx = chunk.get("document_index")
            if doc_idx is None:
                doc_idx = chunk.get("speech_index", 0)
            if doc_idx < 0 or doc_idx >= len(documents):
                continue

            document = documents[doc_idx]
            snippet_text = chunk.get("text", document.content).replace(
                "\n", " "
            )
            snippet = shorten(snippet_text, width=150, placeholder="...")

            result = {
                "rank": rank,
                "score": score,
                "title": document.title,
                "source": document.attribution,
                "date": document.display_date,
                "snippet": snippet,
            }
            results.append(result)

            print(f"\n[{rank}] Score: {score:.3f}")
            print(f"    {document.title}")
            print(f"    {document.attribution} ({document.display_date})")
            print(f"    {snippet}")

        results_by_chunker[chunker_name] = results

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print(f"\nQuery: '{query}'")
    print(f"Top-{top_k} results per chunker:\n")

    for chunker_name, results in results_by_chunker.items():
        if results:
            avg_score = sum(r["score"] for r in results) / len(results)
            top_score = results[0]["score"] if results else 0

            print(f"{chunker_name.upper():20} | "
                  f"Avg Score: {avg_score:.3f} | "
                  f"Top Score: {top_score:.3f}")
            print(f"{'':20} | Top: {results[0]['title'][:50]}...")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark search across chunking strategies"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query to benchmark",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/speeches.json"),
        help="JSON file containing the corpus (default: data/speeches.json)",
    )
    parser.add_argument(
        "--data-key",
        type=str,
        help="JSON key that stores the list of documents (default: auto-detect)",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        help="Prefix for FAISS/metadata artifacts (default: derived from data-file)",
    )

    args = parser.parse_args()

    try:
        benchmark_chunkers(
            args.query,
            args.top_k,
            data_file=args.data_file,
            data_key=args.data_key,
            artifact_prefix=args.artifact_prefix,
        )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted.")
        return 1

    print("\n" + "=" * 80)
    print("INTERPRETING RESULTS:")
    print("=" * 80)
    print("""
Higher scores indicate better semantic match to your query.

- If FULL chunker has highest scores: Your query matches overall themes
- If PARAGRAPHS has highest scores: Balanced match (common case)
- If SLIDING_WINDOW has highest scores: Very specific phrase matching

Compare the actual snippets returned - sometimes a lower score with
more relevant content is better than a higher score with less relevance.

Experiment with:
1. Different queries (broad vs. specific)
2. Different top-k values (5, 10, 20)
3. Note which chunker works best for your use case
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
