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

from semantic_search.data import load_speeches
from semantic_search.chunkers import available_chunkers
from semantic_search.index import (
    build_faiss_index,
    get_artifact_paths,
    load_chunk_metadata,
    load_index,
    load_model,
    search_index,
)


def benchmark_chunkers(query, top_k=5):
    """Run the same query across all chunking strategies."""
    data_dir = Path(__file__).parent.parent / "data"
    speeches_path = data_dir / "speeches.json"

    speeches = load_speeches(speeches_path)
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

        artifact_paths = get_artifact_paths(data_dir, chunker_name)

        # Build index if it doesn't exist
        if (
            not artifact_paths["index"].exists()
            or not artifact_paths["chunks"].exists()
        ):
            print(f"Building index for {chunker_name} chunker...")
            build_faiss_index(speeches, data_dir, chunker_name=chunker_name)

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
            speech_idx = chunk.get("speech_index", 0)
            if speech_idx < 0 or speech_idx >= len(speeches):
                continue

            speech = speeches[speech_idx]
            snippet_text = chunk.get("text", speech.content).replace(
                "\n", " "
            )
            snippet = shorten(snippet_text, width=150, placeholder="...")

            result = {
                "rank": rank,
                "score": score,
                "title": speech.title,
                "speaker": speech.speaker,
                "date": speech.date,
                "snippet": snippet,
            }
            results.append(result)

            print(f"\n[{rank}] Score: {score:.3f}")
            print(f"    {speech.title}")
            print(f"    {speech.speaker} ({speech.date})")
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

    args = parser.parse_args()

    try:
        benchmark_chunkers(args.query, args.top_k)
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
