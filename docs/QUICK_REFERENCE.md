# Quick Reference Card

## Setup (First Time Only)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: pip install uv

# 2. Install dependencies
uv sync

# 3. Build index
uv run fed-faiss-search --rebuild-index
```

---

## Essential Commands

### Basic Search
```bash
# Single query
uv run fed-faiss-search --query "inflation expectations"

# Interactive mode
uv run fed-faiss-search --interactive

# More results
uv run fed-faiss-search --query "AI" --top-k 20
```

### Compare Chunking Strategies
```bash
# Try different chunkers
uv run fed-faiss-search --chunker full --query "monetary policy"
uv run fed-faiss-search --chunker paragraphs --query "monetary policy"
uv run fed-faiss-search --chunker sliding_window --query "monetary policy"
```

### Analysis Tools
```bash
# See how chunkers split text
uv run python examples/compare_chunking_strategies.py

# Benchmark all chunkers
uv run python examples/benchmark_search.py "your query"
```

### Index Management
```bash
# Rebuild index
uv run fed-faiss-search --rebuild-index

# Use different chunker
uv run fed-faiss-search --rebuild-index --chunker sliding_window
```

### Custom Corpus
```bash
# Point to your own JSON corpus
uv run fed-faiss-search \
  --data-file /path/to/my_docs.json \
  --artifact-prefix my_docs \
  --rebuild-index

# Search the same corpus (artifact-prefix keeps indexes separate)
uv run fed-faiss-search \
  --data-file /path/to/my_docs.json \
  --artifact-prefix my_docs \
  --query "supply chain stress"

# If the JSON stores documents under a nested key
uv run fed-faiss-search \
  --data-file corpora/custom.json \
  --data-key documents \
  --query "emerging risks"
```

---

## Chunker Selection Guide

| Query Type | Best Chunker | Example |
|------------|--------------|---------|
| Broad topic | `full` or `paragraphs` | "economic outlook" |
| Specific phrase | `sliding_window` | "Basel III requirements" |
| General search | `paragraphs` (default) | "inflation expectations" |
| Questions | `paragraphs` | "What are the risks?" |

---

## Top-k Selection Guide

| Use Case | Recommended k | Reason |
|----------|---------------|--------|
| Quick lookup | 3-5 | High precision |
| General search | 5-10 | Balanced |
| Research | 20-50 | High recall |
| Comprehensive | 50-100 | Maximum coverage |

---

## Similarity Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.7 - 1.0 | Highly relevant | Trust these results |
| 0.5 - 0.7 | Relevant | Good matches |
| 0.3 - 0.5 | Somewhat related | Review carefully |
| < 0.3 | Weakly related | Likely not useful |

---

## Common Workflows

### Finding Documents on a Topic
```bash
# Step 1: Broad search
uv run fed-faiss-search --query "financial stability" --top-k 10

# Step 2: Review top results
# Step 3: Refine query if needed
uv run fed-faiss-search --query "bank liquidity risks" --top-k 5
```

### Comparing Approaches
```bash
# Compare all chunkers at once
uv run python examples/benchmark_search.py "AI in finance"

# Review results and choose best chunker for your use case
```

### Exploring a New Topic
```bash
# Start interactive
uv run fed-faiss-search --interactive

# Then try variations:
query> monetary policy
query> interest rate decisions
query> quantitative tightening
query> balance sheet normalization
```

---

## Tuning Parameters

### In Code

**Change chunker parameters** (`src/semantic_search/chunkers.py`):
```python
class ParagraphChunker(BaseChunker):
    MIN_CHUNK_SIZE = 450  # Adjust these
    MAX_CHUNK_SIZE = 1100  # for your needs
```

**Change embedding model** (`src/semantic_search/index.py`):
```python
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Change this
    return SentenceTransformer(model_name, device="cpu")
```

After changing, rebuild: `uv run fed-faiss-search --rebuild-index`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Low scores (<0.4) | Try different phrasing or broader query |
| Too many irrelevant results | Lower top-k or use more specific query |
| Missing relevant content | Increase top-k or try sliding_window chunker |
| Slow search | Use full or paragraphs chunker (fewer chunks) |
| Index out of date | Run `--rebuild-index` |

---

## Dataset Info

- **Default corpus**: 198 recent Federal Reserve speeches (`data/speeches.json`)
- **Topics**: Monetary policy, banking regulation, financial stability, AI, etc.
- **Custom corpora**: Use `--data-file` to point at any JSON file (optionally `--data-key` if the documents live under a nested key).
- **Index isolation**: Supply `--artifact-prefix` (defaults to the file name) so each corpus keeps separate FAISS/chunk metadata files.

---

## File Locations

```
Project Files:
  speeches.json          data/speeches.json (default corpus)
  Custom corpora         anywhere on disk via --data-file
  FAISS indices         data/<prefix>_<chunker>.faiss
  Chunk metadata        data/chunks_<prefix>_<chunker>.json

Source Code:
  Main CLI              src/semantic_search/cli.py
  Chunking logic        src/semantic_search/chunkers.py
  Index/search          src/semantic_search/index.py

Documentation:
  This file            docs/QUICK_REFERENCE.md
  Chunking guide       docs/CHUNKING_STRATEGIES.md
  Parameters guide     docs/RETRIEVAL_PARAMETERS.md
  Examples README      examples/README.md
```

---

## Example Queries to Try

**Monetary Policy:**
- "inflation expectations"
- "interest rate decisions"
- "quantitative tightening"
- "monetary policy framework"

**Banking:**
- "stress testing methodology"
- "Basel III capital requirements"
- "bank liquidity management"
- "systemic risk"

**Economics:**
- "labor market dynamics"
- "economic outlook"
- "productivity growth"
- "wage-price spiral"

**Technology:**
- "AI in banking"
- "fintech innovation"
- "payment systems"
- "cybersecurity risks"

**Recent Topics:**
- "pandemic recovery"
- "supply chain disruptions"
- "climate risk"
- "cryptocurrency regulation"

---

## Key Concepts

**Chunking**: Splitting long documents into smaller pieces for better search precision

**Embedding**: Converting text to numerical vectors that capture semantic meaning

**Similarity**: How close two vectors are in embedding space (0-1 scale)

**Top-k**: Number of most similar results to return

**FAISS**: Facebook AI Similarity Search - fast vector similarity library

**Semantic Search**: Finding by meaning rather than exact keyword matching

---

## Next Steps

1. ✓ Try basic searches with default settings
2. ✓ Experiment with different chunkers using examples/compare_chunking_strategies.py
3. ✓ Use benchmark_search.py to find best chunker for your queries
4. ✓ Read docs/CHUNKING_STRATEGIES.md for deeper understanding
5. ✓ Create custom chunker for your specific use case
6. ✓ Share findings or contribute improvements!

---

**Quick Help**: Run any command with `--help` flag for options
```bash
uv run fed-faiss-search --help
uv run python examples/benchmark_search.py --help
```
