# Federal Reserve Semantic Search

This repository contains 198 Federal Reserve speeches in a static dataset at `data/speeches.json`. The speeches are embedded with the `sentence-transformers/all-MiniLM-L6-v2` model and indexed with FAISS HNSW for fast semantic search via a CLI tool.

The stack uses:

- [uv](https://github.com/astral-sh/uv) for dependency and project management.
- [sentence-transformers](https://www.sbert.net/) to download the Hugging Face model locally (CPU only).
- [FAISS](https://github.com/facebookresearch/faiss) HNSW index with cosine similarity as the in-memory vector store.

## Getting Started

**👋 New to this project?** See the **[Getting Started Guide](docs/GETTING_STARTED.md)** for a friendly introduction with examples and experiments.

**⚡ Quick Start:**

1. **Install uv** (if you don't have it):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or with pip
   pip install uv
   ```

2. Install dependencies: `uv sync`
3. Build the FAISS index: `uv run fed-speech-search --rebuild-index`
4. Search: `uv run fed-speech-search --query "inflation expectations"`
5. Interactive mode: `uv run fed-speech-search --interactive`

**📖 Quick Reference:** See **[Quick Reference Card](docs/QUICK_REFERENCE.md)** for command cheat sheet.

**⚡ Alternative: Using pip instead of uv**

If you prefer `pip` or are in a restrictive environment:

```bash
# Install with pip
pip install -r requirements.txt
pip install -e .

# Run commands WITHOUT "uv run" prefix:
fed-speech-search --rebuild-index
fed-speech-search --query "inflation expectations"
fed-speech-search --interactive

# Examples also work:
python examples/compare_chunking_strategies.py
python examples/benchmark_search.py "your query"
```

### Bedrock RAG Demo (Experimental)

You can optionally ask an Amazon Bedrock model (default: Anthropic Claude Sonnet 4.5) to synthesize an answer from the retrieved chunks.

1. Configure AWS credentials with Bedrock access (for example via `aws configure`) and set the desired region (e.g., `us-east-1`).
2. Run:
   ```bash
   uv run fed-speech-search \
     --query "How is AI affecting the banking system?" \
     --rag \
     --rag-top-k 4 \
     --bedrock-region us-east-1
   ```
3. Optional flags:
   - `--bedrock-model` to target another Bedrock foundation model.
   - `--bedrock-max-output` / `--bedrock-temperature` to tune the response length and creativity.

The CLI first prints the retrieved chunks and then the Bedrock answer.

### Notes

- The `--chunker` flag selects how speeches are split before embedding. Available options:
  - `paragraphs` (default): groups neighboring paragraphs into ~450–1100 character chunks.
  - `sliding_window`: token-based windows of ~150 words with 30-word overlap.
  - `full`: treats each speech as one chunk.
- Each chunker has its own FAISS/metadata files under `data/`. Supplying the same `--chunker` during queries ensures the CLI loads the matching index.
- The speech data is stored in `data/speeches.json` and contains 198 Federal Reserve speeches.
- All embeddings use the CPU-backed `sentence-transformers/all-MiniLM-L6-v2` checkpoint, which is cached locally by Hugging Face on first run.
- Re-running `fed-speech-search` without `--rebuild-index` reuses the existing FAISS index. The index is built automatically on first query if it doesn't exist.

### Custom Model Storage

By default, the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is downloaded to the Hugging Face cache directory (`~/.cache/huggingface/hub/`). You can use a custom model location in three ways:

**Option 1: Use Hugging Face Cache (Default)**
```bash
# Model automatically cached at:
# macOS/Linux: ~/.cache/huggingface/hub/
# Windows: C:\Users\<username>\.cache\huggingface\hub\
```

**Option 2: Specify Custom Cache Directory**

Set the `TRANSFORMERS_CACHE` environment variable:
```bash
export TRANSFORMERS_CACHE=/path/to/your/cache
uv run fed-speech-search --rebuild-index
```

**Option 3: Use Project-Local Models (Recommended for Portability)**

Place models in a `models/` directory within the project:
```bash
# Create models directory
mkdir -p models

# Place your model in models/all-MiniLM-L6-v2/
# The code will automatically check models/ before downloading
```

The system checks `models/` first, then falls back to the Hugging Face cache. This makes the project portable and works offline once models are in place.

### Data

The static dataset in `data/speeches.json` contains 198 Federal Reserve speeches covering topics such as:
- Monetary policy and interest rates
- Inflation and economic outlook
- Banking regulation and supervision
- Financial stability
- AI and technology in finance
- Community development

The dataset is part of the repository and does not require downloading from external sources.

## Troubleshooting Installation

### Network/Proxy Issues with `uv sync`

If you're in a restrictive environment (corporate network, firewall, etc.) and `uv sync` fails with download errors:

**Option 1: Configure Proxy Settings**

```bash
# Set proxy environment variables
export HTTP_PROXY="http://your-proxy:port"
export HTTPS_PROXY="http://your-proxy:port"
export NO_PROXY="localhost,127.0.0.1"

# Then try sync
uv sync
```

**Option 2: Use pip with requirements.txt**

```bash
# Use pip to install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Now run WITHOUT uv prefix:
fed-speech-search --rebuild-index
fed-speech-search --query "inflation expectations"
fed-speech-search --interactive

# Or with python -m:
python -m semantic_search.cli --query "inflation expectations"
```

**Note**: When using `pip install`, you run commands directly (e.g., `fed-speech-search`) instead of with `uv run` prefix.

**Option 3: Use Internal PyPI Mirror**

If your organization has an internal PyPI mirror, create `.uvrc`:

```toml
index-url = "https://your-internal-pypi/simple"
```

**Option 4: Offline Installation**

On a machine with internet access:
```bash
# Download all packages
uv export --format requirements-txt > requirements-full.txt
pip download -r requirements-full.txt -d packages/

# Transfer packages/ directory to restricted machine
# Then install:
pip install --no-index --find-links packages/ -r requirements-full.txt
```

**Option 5: Increase Timeout**

```bash
# For slow networks
uv sync --timeout 300
```

### Command Comparison: uv vs pip

| Task | With uv | With pip |
|------|---------|----------|
| Install dependencies | `uv sync` | `pip install -r requirements.txt && pip install -e .` |
| Run search | `uv run fed-speech-search --query "..."` | `fed-speech-search --query "..."` |
| Build index | `uv run fed-speech-search --rebuild-index` | `fed-speech-search --rebuild-index` |
| Interactive mode | `uv run fed-speech-search --interactive` | `fed-speech-search --interactive` |
| Run examples | `uv run python examples/benchmark_search.py` | `python examples/benchmark_search.py` |

**Key Difference**: With `uv`, use `uv run` prefix. With `pip`, run commands directly.

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `Failed to fetch` | Check proxy settings, try `pip install -r requirements.txt` |
| `Connection timeout` | Increase timeout: `uv sync --timeout 300` |
| `Certificate verify failed` | Add trusted host or configure SSL certs |
| `No matching distribution` | Check Python version (`python --version`), needs >=3.10 |

## Experimentation and Learning

This project is designed to be educational and easy to experiment with. Here are ways to explore and understand semantic search:

### Quick Experiments

**Compare chunking strategies** on the same query:
```bash
# Try all three chunkers
uv run fed-speech-search --chunker full --query "AI in banking"
uv run fed-speech-search --chunker paragraphs --query "AI in banking"
uv run fed-speech-search --chunker sliding_window --query "AI in banking"
```

**Adjust number of results**:
```bash
# Get more results for broader exploration
uv run fed-speech-search --query "inflation" --top-k 20

# Get fewer for focused reading
uv run fed-speech-search --query "inflation" --top-k 3
```

**Use interactive mode** for rapid testing:
```bash
uv run fed-speech-search --interactive
```

### Analysis Tools

**Visual chunking comparison**:
```bash
# See how chunkers split the same speech
uv run python examples/compare_chunking_strategies.py

# Try different speeches
uv run python examples/compare_chunking_strategies.py --speech-index 10
```

**Benchmark search quality**:
```bash
# Compare all chunkers side-by-side
uv run python examples/benchmark_search.py "monetary policy"

# Test with more results
uv run python examples/benchmark_search.py "financial stability" --top-k 10
```

### Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)**: Friendly introduction with step-by-step examples
- **[Quick Reference Card](docs/QUICK_REFERENCE.md)**: Command cheat sheet and quick tips
- **[Examples README](examples/README.md)**: Comprehensive experimentation guide with tips and ideas
- **[Chunking Strategies Deep Dive](docs/CHUNKING_STRATEGIES.md)**: Detailed explanation of each chunker, when to use them, and how to tune parameters
- **[Retrieval Parameters Guide](docs/RETRIEVAL_PARAMETERS.md)**: How to tune top-k, embedding models, similarity metrics, and more

### Common Experiments

1. **Test query types**: Try broad vs. specific queries to see how results differ
2. **Compare chunkers**: Use the benchmark script to find the best strategy for your queries
3. **Tune top-k**: Experiment with different result counts (3, 5, 10, 20, 50)
4. **Explore the dataset**: Search for different topics and speakers
5. **Create custom chunkers**: Modify `src/semantic_search/chunkers.py` to implement your own strategy

### Learning Resources

- See [examples/README.md](examples/README.md) for hands-on tutorials
- Read [docs/CHUNKING_STRATEGIES.md](docs/CHUNKING_STRATEGIES.md) to understand chunking trade-offs
- Check [docs/RETRIEVAL_PARAMETERS.md](docs/RETRIEVAL_PARAMETERS.md) for advanced tuning

## Project Structure

```
faiss/
├── src/semantic_search/     # Core source code
│   ├── cli.py              # Command-line interface
│   ├── data.py             # Data loading
│   ├── index.py            # FAISS indexing and search
│   ├── chunkers.py         # Chunking strategies
│   └── rag.py              # Amazon Bedrock helper for RAG demos
├── data/                    # Data and indices
│   ├── speeches.json       # 198 Federal Reserve speeches
│   └── *.faiss            # Generated FAISS indices
├── examples/               # Experimentation tools
│   ├── README.md          # Experimentation guide
│   ├── compare_chunking_strategies.py
│   └── benchmark_search.py
├── docs/                   # Documentation
│   ├── GETTING_STARTED.md      # Beginner-friendly guide
│   ├── QUICK_REFERENCE.md      # Command cheat sheet
│   ├── CHUNKING_STRATEGIES.md  # Chunking deep dive
│   └── RETRIEVAL_PARAMETERS.md # Parameter tuning guide
└── README.md              # This file
```
