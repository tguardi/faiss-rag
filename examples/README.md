# Examples and Experimentation Guide

This directory contains tools and examples to help you understand and experiment with different aspects of semantic search.

## Prerequisites

Before running these examples, make sure you have:

1. **Installed uv**:
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or with pip
   pip install uv
   ```

2. **Installed dependencies**:
   ```bash
   uv sync
   ```

3. **Built the index** (at least once):
   ```bash
   uv run fed-faiss-search --rebuild-index
   ```

## Quick Start

### 1. Compare Chunking Strategies

See how different chunking strategies split the same speech:

```bash
uv run python examples/compare_chunking_strategies.py
```

This shows you:
- How many chunks each strategy creates
- Average, min, and max chunk sizes
- Preview of actual chunk content
- Trade-offs between strategies

Try different speeches:
```bash
uv run python examples/compare_chunking_strategies.py --speech-index 5
```

### 2. Benchmark Search Quality

Compare search results across all chunking strategies:

```bash
uv run python examples/benchmark_search.py "inflation expectations"
```

This helps you understand:
- Which chunker works best for different query types
- How similarity scores vary
- What content each strategy retrieves

Try different queries:
```bash
# Broad topic query
uv run python examples/benchmark_search.py "monetary policy"

# Specific detail query
uv run python examples/benchmark_search.py "interest rate decision"

# Get more results
uv run python examples/benchmark_search.py "AI" --top-k 10
```

### 3. Interactive Experimentation

Use the interactive mode to rapidly test different queries:

```bash
uv run fed-faiss-search --interactive --chunker paragraphs
```

Then try the same session with a different chunker:
```bash
uv run fed-faiss-search --interactive --chunker sliding_window
```

## Understanding Chunking Strategies

### Full Speech Chunker
**When to use:**
- Finding speeches by overall topic
- Broad thematic queries
- When you want whole-speech context

**Trade-offs:**
- ✅ Maximum context preserved
- ✅ Fewer, simpler results
- ❌ Less precise matching on details
- ❌ Large chunks may dilute signal

**Example queries:**
```bash
uv run fed-faiss-search --chunker full --query "economic outlook"
```

### Paragraph Chunker (Default)
**When to use:**
- General-purpose search
- Balance between context and precision
- Most use cases

**Trade-offs:**
- ✅ Respects natural text boundaries
- ✅ Good balance of context/precision
- ✅ Reasonable number of chunks
- ⚖️ May miss very specific phrases

**Example queries:**
```bash
uv run fed-faiss-search --chunker paragraphs --query "inflation expectations"
```

### Sliding Window Chunker
**When to use:**
- Finding specific phrases or technical details
- When context boundaries don't matter
- Maximum recall needed

**Trade-offs:**
- ✅ Captures all text with overlap
- ✅ Best for finding specific content
- ❌ More chunks = slower search
- ❌ May split coherent thoughts

**Example queries:**
```bash
uv run fed-faiss-search --chunker sliding_window --query "quantitative tightening"
```

## Experimentation Ideas

### 1. Query Type Testing

Test how different query types perform:

```bash
# Abstract concepts
uv run python examples/benchmark_search.py "financial stability"

# Specific terms
uv run python examples/benchmark_search.py "Basel III capital requirements"

# Questions
uv run python examples/benchmark_search.py "what are the risks to inflation"

# Names and people
uv run python examples/benchmark_search.py "Jerome Powell"
```

### 2. Top-k Tuning

Experiment with different numbers of results:

```bash
# Few results (high precision)
uv run fed-faiss-search --query "AI in banking" --top-k 3

# Many results (high recall)
uv run fed-faiss-search --query "AI in banking" --top-k 20
```

### 3. Custom Chunker Development

Want to create your own chunking strategy? Edit `src/semantic_search/chunkers.py`:

```python
class MyCustomChunker(BaseChunker):
    def chunk_speech(self, speech: Speech, speech_index: int) -> list[Chunk]:
        # Your custom chunking logic here
        # Return list of Chunk objects
        pass
```

Then register it in `available_chunkers()` and use it:
```bash
uv run fed-faiss-search --chunker my_custom --query "test"
```

## Measuring Success

### Quantitative Metrics
- **Similarity scores**: Higher = better semantic match (0.0 to 1.0)
- **Chunk counts**: Fewer chunks = faster search, more chunks = better coverage
- **Avg chunk size**: Smaller = more precise, larger = more context

### Qualitative Assessment
1. Read the actual snippets returned
2. Check if they answer your information need
3. Verify relevance vs. similarity score
4. Note false positives/negatives

## Advanced Experiments

### Dataset Analysis

Explore the default speech dataset (or your own documents):

```python
from pathlib import Path
from semantic_search.data import load_documents

documents = load_documents(Path("data/speeches.json"))

# Find speeches by speaker
bowman_speeches = [d for d in documents if (d.speaker or "").find("Bowman") >= 0]
print(f"Michelle Bowman appears in {len(bowman_speeches)} documents")

# Find recent speeches
recent = documents[:10]  # Already sorted by date
for doc in recent:
    print(f"{doc.date}: {doc.title}")

# Analyze topics
inflation_docs = [
    d for d in documents
    if "inflation" in d.title.lower()
]
print(f"{len(inflation_docs)} documents mention 'inflation' in the title")
```

### Index Inspection

Examine the FAISS index properties:

```python
from semantic_search.index import load_index, get_artifact_paths
from pathlib import Path

paths = get_artifact_paths(Path("data"), "paragraphs")
index = load_index(paths["index"])

print(f"Index dimension: {index.d}")
print(f"Number of vectors: {index.ntotal}")
print(f"Index type: {type(index).__name__}")
```

### Embedding Exploration

Understand what the model is learning:

```bash
# Try semantically similar queries
uv run fed-faiss-search --query "inflation"
uv run fed-faiss-search --query "price increases"
uv run fed-faiss-search --query "cost of living"

# Try different phrasings
uv run fed-faiss-search --query "lower interest rates"
uv run fed-faiss-search --query "cut rates"
uv run fed-faiss-search --query "accommodative policy"
```

## Tips for Experimentation

1. **Start broad, then narrow**: Begin with general queries, then refine
2. **Compare chunkers side-by-side**: Use `benchmark_search.py`
3. **Check multiple speeches**: Don't rely on just the top result
4. **Consider your use case**: What information are you trying to find?
5. **Document your findings**: Keep notes on what works for your queries

## Contributing Your Experiments

Found an interesting pattern or use case? Consider:
1. Adding a new example script to this directory
2. Documenting your custom chunker
3. Sharing benchmark results for specific query types

## Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Federal Reserve Speeches](https://www.federalreserve.gov/newsevents/speeches.htm)
