#!/usr/bin/env python3
"""
Compare different chunking strategies on a sample document.

This script helps visualize how different chunking strategies split text,
making it easier to understand their trade-offs.

Usage:
    uv run python examples/compare_chunking_strategies.py
    uv run python examples/compare_chunking_strategies.py --document-index 5
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_search.data import load_documents
from semantic_search.chunkers import (
    FullSpeechChunker,
    ParagraphChunker,
    SlidingWindowChunker,
)


def print_separator(char="=", length=80):
    print(char * length)


def print_chunk_stats(chunker_name, chunks):
    """Print statistics about chunks."""
    print(f"\n{chunker_name.upper()} CHUNKER")
    print_separator()
    print(f"Total chunks: {len(chunks)}")

    chunk_lengths = [len(chunk.text) for chunk in chunks]
    if chunk_lengths:
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)

        print(f"Average chunk length: {avg_length:.0f} characters")
        print(f"Min chunk length: {min_length} characters")
        print(f"Max chunk length: {max_length} characters")
    print()


def preview_chunks(chunks, max_preview=3):
    """Show a preview of the first few chunks."""
    print("Preview of first chunks:")
    print_separator("-")

    for i, chunk in enumerate(chunks[:max_preview], 1):
        preview_text = chunk.text[:200].replace("\n", " ")
        if len(chunk.text) > 200:
            preview_text += "..."

        print(f"\n[Chunk {i}] ({len(chunk.text)} chars)")
        print(preview_text)
        print_separator("-")

    if len(chunks) > max_preview:
        print(f"\n... and {len(chunks) - max_preview} more chunks")


def compare_chunkers(doc, show_previews=True):
    """Compare all available chunking strategies on a document."""
    print_separator("=")
    print(f"DOCUMENT: {doc.title}")
    print(f"Source: {doc.attribution}")
    print(f"Date: {doc.display_date}")
    print(f"Total length: {len(doc.content)} characters")
    print_separator("=")

    chunkers = {
        "full": FullSpeechChunker(),
        "paragraphs": ParagraphChunker(),
        "sliding_window": SlidingWindowChunker(),
    }

    for name, chunker in chunkers.items():
        chunks = chunker.chunk_documents([doc])
        print_chunk_stats(name, chunks)

        if show_previews and name != "full":
            preview_chunks(chunks, max_preview=2)
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on documents"
    )
    parser.add_argument(
        "--document-index",
        "--speech-index",
        dest="document_index",
        type=int,
        default=0,
        help="Index of document to analyze (default: 0 = first entry)",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Don't show chunk previews, just statistics",
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

    args = parser.parse_args()

    # Load documents
    data_path = args.data_file.expanduser()
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return 1

    documents = load_documents(data_path, dataset_key=args.data_key)

    if not documents:
        print("Error: No documents found in dataset")
        return 1

    if args.document_index >= len(documents):
        print(
            f"Error: Document index {args.document_index} out of range. "
            f"Dataset has {len(documents)} entries (0-{len(documents)-1})"
        )
        return 1

    document = documents[args.document_index]
    compare_chunkers(document, show_previews=not args.no_previews)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print(
        """
- FULL CHUNKER: Best for finding documents by overall topic/theme.
  Trade-off: Less precise matching on specific details.

- PARAGRAPHS CHUNKER (default): Balanced approach for most use cases.
  Trade-off: Good precision while maintaining context.

- SLIDING WINDOW CHUNKER: Best for finding specific phrases/details.
  Trade-off: May split semantic units, more chunks to search.

Try different chunkers with:
    uv run fed-faiss-search --chunker full --query "your query"
    uv run fed-faiss-search --chunker paragraphs --query "your query"
    uv run fed-faiss-search --chunker sliding_window --query "your query"
"""
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
