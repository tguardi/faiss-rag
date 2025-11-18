#!/usr/bin/env python3
"""
Compare different chunking strategies on a sample speech.

This script helps visualize how different chunking strategies split text,
making it easier to understand their trade-offs.

Usage:
    uv run python examples/compare_chunking_strategies.py
    uv run python examples/compare_chunking_strategies.py --speech-index 5
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_search.data import load_speeches
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


def compare_chunkers(speech, show_previews=True):
    """Compare all available chunking strategies on a speech."""
    print_separator("=")
    print(f"SPEECH: {speech.title}")
    print(f"Speaker: {speech.speaker}")
    print(f"Date: {speech.date}")
    print(f"Total length: {len(speech.content)} characters")
    print_separator("=")

    chunkers = {
        "full": FullSpeechChunker(),
        "paragraphs": ParagraphChunker(),
        "sliding_window": SlidingWindowChunker(),
    }

    for name, chunker in chunkers.items():
        # chunk_speeches expects a list of speeches
        chunks = chunker.chunk_speeches([speech])
        print_chunk_stats(name, chunks)

        if show_previews and name != "full":
            preview_chunks(chunks, max_preview=2)
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on Federal Reserve speeches"
    )
    parser.add_argument(
        "--speech-index",
        type=int,
        default=0,
        help="Index of speech to analyze (default: 0 = most recent)",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Don't show chunk previews, just statistics",
    )

    args = parser.parse_args()

    # Load speeches
    data_dir = Path(__file__).parent.parent / "data"
    speeches_path = data_dir / "speeches.json"

    if not speeches_path.exists():
        print(f"Error: {speeches_path} not found")
        return 1

    speeches = load_speeches(speeches_path)

    if not speeches:
        print("Error: No speeches found in dataset")
        return 1

    if args.speech_index >= len(speeches):
        print(
            f"Error: Speech index {args.speech_index} out of range. "
            f"Dataset has {len(speeches)} speeches (0-{len(speeches)-1})"
        )
        return 1

    speech = speeches[args.speech_index]
    compare_chunkers(speech, show_previews=not args.no_previews)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print(
        """
- FULL CHUNKER: Best for finding speeches by overall topic/theme.
  Trade-off: Less precise matching on specific details.

- PARAGRAPHS CHUNKER (default): Balanced approach for most use cases.
  Trade-off: Good precision while maintaining context.

- SLIDING WINDOW CHUNKER: Best for finding specific phrases/details.
  Trade-off: May split semantic units, more chunks to search.

Try different chunkers with:
    uv run fed-speech-search --chunker full --query "your query"
    uv run fed-speech-search --chunker paragraphs --query "your query"
    uv run fed-speech-search --chunker sliding_window --query "your query"
"""
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
