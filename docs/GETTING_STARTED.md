# Getting Started with Federal Reserve Semantic Search

Welcome! This guide will help you get started with semantic search and experimentation.

## 🚀 5-Minute Quickstart

### 0. Install uv (First Time Only)

If you don't have `uv` installed yet:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip (if you have Python already)
pip install uv
```

After installation, you may need to restart your terminal or run `source ~/.bashrc` (Linux) or `source ~/.zshrc` (macOS).

**What is uv?** It's a fast Python package and project manager that handles dependencies and virtual environments automatically. Learn more at [astral.sh/uv](https://astral.sh/uv).

### 1. Install Dependencies and Build Index (First Time Only)

```bash
# Install project dependencies
uv sync

# Build the search index
uv run fed-faiss-search --rebuild-index
```

This will take ~2 minutes. You only need to do this once (or when changing chunkers).

### 2. Try Your First Search

```bash
uv run fed-faiss-search --query "inflation expectations"
```

You should see results like:
```
Query: inflation expectations
------------------------------------------------------------
[1] Inflation Expectations and Monetary Policymaking — Governor Adriana D. Kugler (April 02, 2025) [chunk 30]
Score: 0.704
URL: https://www.federalreserve.gov/newsevents/speech/kugler20250402a.htm
Snippet: 10. See David Lebow and Ekaterina Peneva (2024), " Inflation Perceptions during...
------------------------------------------------------------
```

### 3. Try Interactive Mode

```bash
uv run fed-faiss-search --interactive
```

Then type different queries:
```
query> monetary policy
query> AI in banking
query> financial stability risks
query> exit   # to quit
```

**Congrats!** You're doing semantic search. 🎉

---

## 🧪 Your First Experiments

### Experiment 1: Compare Chunking Strategies

See how different chunkers affect results:

```bash
# Default paragraphs chunker
uv run fed-faiss-search --query "stress testing" --top-k 3

# Full speech chunker
uv run fed-faiss-search --chunker full --query "stress testing" --top-k 3

# Sliding window chunker
uv run fed-faiss-search --chunker sliding_window --query "stress testing" --top-k 3
```

**What to notice:**
- Different chunks returned
- Different similarity scores
- Different levels of specificity

### Experiment 2: Visualize Chunking

See exactly how chunkers split text:

```bash
uv run python examples/compare_chunking_strategies.py
```

You'll see:
- Number of chunks created
- Chunk size statistics
- Preview of actual chunks

**Try different speeches:**
```bash
uv run python examples/compare_chunking_strategies.py --speech-index 5
uv run python examples/compare_chunking_strategies.py --speech-index 10
```

### Experiment 3: Benchmark Search Quality

Compare all chunkers side-by-side:

```bash
uv run python examples/benchmark_search.py "AI and financial services"
```

This runs your query through all three chunkers and shows:
- Top results from each
- Similarity scores
- Which chunker works best for this query

**Try different query types:**
```bash
# Broad topic
uv run python examples/benchmark_search.py "economic outlook"

# Specific phrase
uv run python examples/benchmark_search.py "quantitative tightening balance sheet"

# Question
uv run python examples/benchmark_search.py "what are inflation risks"
```

---

## 📊 Understanding Your Results

### Similarity Scores

```
0.7 - 1.0  ⭐⭐⭐  Excellent match
0.5 - 0.7  ⭐⭐    Good match
0.3 - 0.5  ⭐     Okay match
< 0.3             Weak match
```

**Example interpretation:**
```
[1] Score: 0.724  ← This is a great match!
[2] Score: 0.612  ← This is relevant
[3] Score: 0.458  ← This might be tangentially related
[4] Score: 0.289  ← This is probably not what you want
```

### Reading Results

Each result shows:
```
[1] Speech Title — Speaker (Date) [chunk X]
Score: 0.704
URL: https://...
Snippet: Preview of matched text...
```

- **Rank [1]**: Position in results (1 = most similar)
- **Score**: How similar to your query (0.0 - 1.0)
- **Chunk X**: Which part of the speech matched
- **Snippet**: Preview of the matched content

---

## 🎯 When to Use Each Chunker

### Use `--chunker full` when:

✅ Finding speeches by overall theme
✅ Broad exploratory search
✅ Want complete context

❌ Avoid for specific details

**Example queries:**
```bash
uv run fed-faiss-search --chunker full --query "economic outlook 2025"
uv run fed-faiss-search --chunker full --query "monetary policy stance"
```

### Use `--chunker paragraphs` (default) when:

✅ General-purpose search
✅ Balanced precision and context
✅ Most use cases

**Example queries:**
```bash
uv run fed-faiss-search --query "inflation expectations"
uv run fed-faiss-search --query "bank stress testing"
```

### Use `--chunker sliding_window` when:

✅ Finding specific phrases
✅ Technical term search
✅ Maximum recall needed

❌ Slower, more results to review

**Example queries:**
```bash
uv run fed-faiss-search --chunker sliding_window --query "Basel III capital buffers"
uv run fed-faiss-search --chunker sliding_window --query "liquidity coverage ratio"
```

---

## 🔧 Tuning Your Search

### Adjust Number of Results

```bash
# Get just the top match
uv run fed-faiss-search --query "AI risks" --top-k 1

# Get a few relevant results (default)
uv run fed-faiss-search --query "AI risks" --top-k 5

# Get many results for research
uv run fed-faiss-search --query "AI risks" --top-k 20
```

**Rule of thumb:**
- **k=3-5**: Quick lookup
- **k=10**: General search
- **k=20-50**: Research/analysis

### Improve Your Queries

**Be specific:**
```
❌ "banks"
✓ "community bank regulation"
```

**Use domain terminology:**
```
❌ "money printing"
✓ "quantitative easing asset purchases"
```

**Natural language works:**
```
✓ "How does the Fed view labor market strength?"
✓ "What are the main risks to financial stability?"
```

**Multiple related terms:**
```
✓ "AI artificial intelligence machine learning automation"
```

---

## 📚 Example Search Sessions

### Session 1: Learning about Inflation

```bash
uv run fed-faiss-search --interactive
```

```
query> inflation expectations
# Review top results, note speakers and dates

query> inflation drivers
# Compare perspectives

query> price stability mandate
# Understand policy context

query> PCE core inflation
# Technical details
```

### Session 2: Understanding Monetary Policy

```bash
# Try different chunkers to see what works
uv run fed-faiss-search --chunker paragraphs --query "rate cuts" --top-k 10
uv run fed-faiss-search --chunker full --query "rate cuts" --top-k 10

# Compare
uv run python examples/benchmark_search.py "interest rate policy"
```

### Session 3: Researching AI in Finance

```bash
# Cast a wide net
uv run fed-faiss-search --query "artificial intelligence" --top-k 30

# Get specific
uv run fed-faiss-search --query "AI risks financial system" --top-k 10

# Find specific topics
uv run fed-faiss-search --chunker sliding_window --query "machine learning credit decisions"
```

---

## 🎓 Next Steps

### Beginner
1. ✅ Try 10+ different queries
2. ✅ Experiment with different top-k values (3, 5, 10, 20)
3. ✅ Run `compare_chunking_strategies.py` on different speeches
4. ✅ Read similarity scores and understand relevance

### Intermediate
1. ✅ Use `benchmark_search.py` to compare chunkers
2. ✅ Try all three chunkers on your typical queries
3. ✅ Read [CHUNKING_STRATEGIES.md](CHUNKING_STRATEGIES.md)
4. ✅ Experiment with query phrasing

### Advanced
1. ✅ Read [RETRIEVAL_PARAMETERS.md](RETRIEVAL_PARAMETERS.md)
2. ✅ Modify chunker parameters in `src/semantic_search/chunkers.py`
3. ✅ Create a custom chunker
4. ✅ Try different embedding models
5. ✅ Build evaluation metrics for your use case

---

## 💡 Tips for Success

### Do:
✓ Start with default settings (paragraphs chunker, k=5)
✓ Read the actual snippets, not just scores
✓ Experiment with different phrasings
✓ Use benchmark_search.py to compare options
✓ Document what works for your queries

### Don't:
✗ Trust scores alone - read the content
✗ Use single-word queries
✗ Expect perfect results every time
✗ Change multiple parameters at once
✗ Skip the comparison tools

---

## 🆘 Common Questions

**Q: Why are my scores low (<0.5)?**
A: Query may be too vague or using different terminology than the speeches. Try being more specific or using Federal Reserve jargon.

**Q: Which chunker should I use?**
A: Start with `paragraphs` (default). Use `benchmark_search.py` to compare if unsure.

**Q: How many results should I request?**
A: Start with 5. Increase if you're not finding what you need. Decrease if too many irrelevant results.

**Q: Can I search for specific people?**
A: Yes! Try: `uv run fed-faiss-search --query "Jerome Powell policy views"`

**Q: Why do I need to rebuild the index?**
A: Only when changing chunkers or modifying chunker parameters. The index stores embeddings specific to each chunking strategy.

**Q: How do I add more speeches?**
A: The dataset is static. You can edit `data/speeches.json` to add speeches, then rebuild with `--rebuild-index`.

---

## 📖 Documentation Index

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Command cheat sheet
- **[CHUNKING_STRATEGIES.md](CHUNKING_STRATEGIES.md)**: Deep dive on chunking
- **[RETRIEVAL_PARAMETERS.md](RETRIEVAL_PARAMETERS.md)**: Tuning guide
- **[examples/README.md](../examples/README.md)**: Experimentation guide

---

## 🎯 Challenge Yourself

Try these to test your understanding:

1. Find 3 speeches about AI using different chunkers - which works best?
2. Compare results for "inflation" with k=5 vs k=20 - how does relevance change?
3. Find the best chunker for question-style queries
4. Identify query types where sliding_window outperforms paragraphs
5. Create a custom chunker and test it on your favorite query

---

**Ready to dive deeper?** Check out [examples/README.md](../examples/README.md) for more advanced experiments!

**Questions or issues?** Review the documentation or experiment with the comparison tools.

Happy searching! 🔍
