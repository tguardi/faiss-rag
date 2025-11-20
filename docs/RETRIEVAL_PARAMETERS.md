# Retrieval Parameters Guide

## Overview

This guide explains the key parameters that affect semantic search quality and how to tune them for your use case.

## Core Parameters

### 1. Top-k (Number of Results)

**What it does**: Controls how many results to return

**Default**: 5

**How to change**:
```bash
uv run fed-faiss-search --query "inflation" --top-k 10
```

#### Trade-offs

| k Value | Precision | Recall | Speed | Use Case |
|---------|-----------|--------|-------|----------|
| 1-3 | Highest | Lowest | Fastest | Quick fact lookup |
| 5-10 | High | Medium | Fast | General search |
| 20-50 | Medium | High | Medium | Research/exploration |
| 100+ | Low | Highest | Slower | Comprehensive analysis |

#### Recommendations

**Use k=1-3 when**:
- You want the single best answer
- High confidence in query phrasing
- Time/attention is limited

Example: "Who is the current Fed chair?"

**Use k=5-10 when**:
- General information seeking
- Want to compare perspectives
- Standard search behavior

Example: "What are views on inflation expectations?"

**Use k=20-50 when**:
- Research or analysis
- Want comprehensive coverage
- Building a dataset

Example: "All mentions of quantitative tightening"

**Use k=100+ when**:
- Exhaustive search
- Building training data
- Statistical analysis

Example: "Every speech mentioning AI"

### 2. Embedding Model

**What it does**: Converts text to vector representations

**Default**: `sentence-transformers/all-MiniLM-L6-v2`

**How to change**: Edit `src/semantic_search/index.py`:

```python
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Change this
    return SentenceTransformer(model_name, device="cpu")
```

#### Available Models

| Model | Dimensions | Speed | Quality | Size |
|-------|------------|-------|---------|------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | 80MB |
| all-mpnet-base-v2 | 768 | Medium | Better | 420MB |
| all-MiniLM-L12-v2 | 384 | Medium | Better | 120MB |
| multi-qa-mpnet-base | 768 | Medium | Best (QA) | 420MB |

See all models: https://www.sbert.net/docs/pretrained_models.html

#### Model Selection Guide

**Use MiniLM-L6-v2 (default) when**:
- Speed matters
- Resource-constrained environment
- Good enough quality for most queries

**Use all-mpnet-base-v2 when**:
- Quality is paramount
- Have GPU or powerful CPU
- Complex semantic matching needed

**Use multi-qa-mpnet-base when**:
- Queries are questions
- Question-answering use case
- Best quality needed

#### Switching Models

1. Edit `load_model()` in `src/semantic_search/index.py`
2. Rebuild index: `uv run fed-faiss-search --rebuild-index`
3. Model will download automatically on first use

**Warning**: Different models = different vectors = must rebuild index!

### 3. FAISS Index Type

**What it does**: Algorithm for similarity search

**Default**: `IndexHNSWFlat` (HNSW with Inner Product / Cosine Similarity)

**How to change**: Edit `src/semantic_search/index.py`:

```python
def _create_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    # ...
```

#### Index Types

**IndexHNSWFlat** (Current - Graph-based with Inner Product)
- Fast approximate search with excellent recall
- Works with normalized embeddings (cosine similarity)
- Efficient for datasets of any size
- No training required
- Parameters:
  - `M=32`: Connections per node (16-64 range)
  - `efConstruction=64`: Build quality (higher = better)
  - `efSearch=64`: Search quality (higher = more accurate)

```python
M = 32
index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 64
index.hnsw.efSearch = 64
```

**IndexFlatIP** (Alternative - Exact Brute Force)
- Exact search (100% recall)
- Works with normalized embeddings
- Good for small datasets (<10k vectors)
- No training required
- Slower than HNSW for large datasets

```python
index = faiss.IndexFlatIP(dim)
```

**IndexFlatL2** (L2 Distance)
- Exact search
- Euclidean distance
- Alternative to IP for non-normalized embeddings

```python
index = faiss.IndexFlatL2(dim)
```

**IndexIVFFlat** (Cluster-based Approximate)
- Faster for very large datasets (>1M vectors)
- Requires training
- More accuracy loss than HNSW

```python
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, 100)  # 100 clusters
index.train(embeddings)
```

#### Index Selection

For this project (198 speeches, ~5k chunks):
- **Current (IndexHNSWFlat)**: Excellent choice
- Fast, accurate, scales well
- No need to change for most use cases

#### HNSW Tuning Parameters

You can adjust HNSW parameters in `_create_index()`:

**M (connections per node)**:
- Default: 32
- Range: 16-64
- Higher = Better recall, more memory
- Lower = Faster build, less memory

**efConstruction (build quality)**:
- Default: 64
- Range: 40-500
- Higher = Better quality index, slower build
- Recommended: 64-200 for most cases

**efSearch (search quality)**:
- Default: 64
- Range: 10-500
- Higher = More accurate, slower search
- Lower = Faster search, slight recall loss
- Can be adjusted at search time

Example tuning:
```python
# Higher quality, slower
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128

# Faster, slightly lower quality
index.hnsw.efConstruction = 40
index.hnsw.efSearch = 32
```

For larger projects (>100k chunks):
- HNSW is ideal for most use cases
- Consider IndexIVFFlat only for >1M vectors
- See FAISS documentation for advanced tuning

### 4. Similarity Metric

**What it does**: How to measure vector similarity

**Options**:
1. **Cosine Similarity** (Current via normalized IP)
2. **Euclidean Distance** (L2)
3. **Dot Product** (unnormalized IP)

#### Current Implementation

```python
# In build_faiss_index()
embeddings = model.encode(texts, ..., normalize_embeddings=True)
# normalize_embeddings=True makes IP equivalent to cosine similarity
```

With normalized embeddings:
- Inner Product (IP) = Cosine Similarity
- Values range from -1 (opposite) to 1 (identical)
- 0 = orthogonal (unrelated)

#### Understanding Similarity Scores

```
0.8 - 1.0  : Extremely similar (rare, usually near-duplicates)
0.6 - 0.8  : Highly relevant
0.4 - 0.6  : Moderately relevant
0.2 - 0.4  : Tangentially related
0.0 - 0.2  : Barely related
< 0.0      : Different context/meaning
```

#### Typical Score Distributions

**Well-phrased query** ("inflation expectations"):
```
Top-1: 0.72 (highly relevant)
Top-5: 0.65-0.72 (all relevant)
Top-10: 0.55-0.72 (mostly relevant)
```

**Vague query** ("economic"):
```
Top-1: 0.45 (moderately relevant)
Top-5: 0.40-0.45 (mixed relevance)
Top-10: 0.35-0.45 (many false positives)
```

## Query Optimization

### Query Formulation Tips

**1. Be Specific**
```
❌ "economy"
✓ "economic outlook for 2025"
```

**2. Use Domain Language**
```
❌ "money printing"
✓ "quantitative easing"
```

**3. Natural Language Works**
```
✓ "What are the risks to inflation?"
✓ "How does the Fed view labor markets?"
```

**4. Multiple Terms**
```
✓ "AI artificial intelligence machine learning"
```

**5. Avoid Single Keywords**
```
❌ "inflation"
✓ "inflation expectations consumer prices"
```

### Query Expansion

For better recall, try multiple related queries:

```bash
# Instead of one query
uv run fed-faiss-search --query "inflation"

# Try variations
uv run fed-faiss-search --query "inflation expectations"
uv run fed-faiss-search --query "price stability"
uv run fed-faiss-search --query "consumer price index"
```

Or use the benchmark script:
```bash
uv run python examples/benchmark_search.py "inflation expectations"
```

## Advanced Tuning

### Re-ranking

After initial retrieval, you can re-rank results:

```python
# In cli.py or custom script
scores, indices = search_index(query, index, model, top_k=50)

# Re-rank by additional criteria
# Example: Boost recent speeches
reranked = sorted(
    zip(scores, indices),
    key=lambda x: (
        x[0] * 0.7 +  # Similarity score (70%)
        recency_score(speeches[chunks[x[1]]['speech_index']]) * 0.3  # Recency (30%)
    ),
    reverse=True
)[:10]
```

### Filtering

Add post-retrieval filters:

```python
# Filter by date
from datetime import datetime

def is_recent(speech, months=6):
    # Parse speech.date and check if within last N months
    pass

filtered_results = [
    (score, idx)
    for score, idx in zip(scores, indices)
    if is_recent(speeches[chunks[idx]['speech_index']])
]
```

### Multi-vector Search

Average multiple query embeddings:

```python
queries = [
    "inflation expectations",
    "price stability",
    "consumer prices"
]

embeddings = model.encode(queries, normalize_embeddings=True)
avg_embedding = embeddings.mean(axis=0)
avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Re-normalize

# Search with average embedding
scores, indices = index.search(avg_embedding.reshape(1, -1), k=10)
```

## Experimentation Workflow

### 1. Baseline

Establish baseline performance:

```bash
uv run fed-faiss-search --query "your typical query" --top-k 10
```

Document:
- Relevance of top-10
- First irrelevant result position
- Similarity scores

### 2. Parameter Sweep

Test different k values:

```bash
for k in 3 5 10 20 50; do
    echo "Testing k=$k"
    uv run fed-faiss-search --query "query" --top-k $k
done
```

### 3. Chunker Comparison

```bash
uv run python examples/benchmark_search.py "your query"
```

### 4. Iterate

Based on results:
- Adjust chunker if precision/recall off
- Tune k based on how many relevant results
- Refine query phrasing
- Consider custom chunker

### 5. Document

Keep notes:
```
Query: "monetary policy framework"
Chunker: paragraphs
Top-k: 10
Results: 8/10 relevant
Notes: Top results excellent, could reduce k to 5

Query: "AI in banking supervision"
Chunker: sliding_window
Top-k: 20
Results: 15/20 relevant
Notes: Good recall, but sliding_window may be overkill
```

## Common Issues and Solutions

### Issue: Low Similarity Scores (<0.4)

**Causes**:
- Query too vague
- Using wrong terminology
- Topic not in dataset

**Solutions**:
1. Be more specific
2. Use domain language
3. Try related queries
4. Expand dataset

### Issue: Too Many Results

**Causes**:
- k too high
- Query too broad
- Chunker too fine-grained

**Solutions**:
1. Reduce k
2. Refine query
3. Try coarser chunker (full or paragraphs)

### Issue: Missing Relevant Results

**Causes**:
- k too low
- Chunker too coarse
- Query-document vocabulary mismatch

**Solutions**:
1. Increase k
2. Try sliding_window chunker
3. Rephrase query
4. Use query expansion

### Issue: Results Out of Order

**Causes**:
- Similarity scores very close
- Chunker splits relevant content

**Solutions**:
1. Use finer chunker
2. Increase k and manually inspect
3. Consider re-ranking

## Performance Benchmarks

With default settings (all-MiniLM-L6-v2, paragraphs chunker, HNSW):

```
Index build: ~2 minutes
Index size: ~500KB + 400KB metadata
Search latency: <100ms per query (faster than flat index)
Memory usage: ~120MB
Recall: >99% (excellent approximation quality)

With 198 speeches (~4,500 chunks):
- Fast approximate search (IndexHNSWFlat)
- No GPU needed
- Runs on laptop
- Scales to much larger datasets
```

## Further Optimization

### For Production Use

1. **GPU acceleration**: Change `device="cpu"` to `device="cuda"`
2. **Batch queries**: Process multiple queries together
3. **Cache embeddings**: Don't re-encode same queries
4. **Use approximate search**: IndexIVF or IndexHNSW for large datasets

### Code Example: Query Caching

```python
import hashlib
import pickle

query_cache = {}

def cached_search(query, index, model, top_k=5):
    cache_key = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()

    if cache_key in query_cache:
        return query_cache[cache_key]

    results = search_index(query, index, model, top_k)
    query_cache[cache_key] = results
    return results
```

## Summary Checklist

- [ ] Understand your query patterns (broad vs specific)
- [ ] Test with default settings first
- [ ] Use benchmark script to compare chunkers
- [ ] Tune k based on precision/recall needs
- [ ] Consider model upgrade only if needed
- [ ] Document what works for your use case
- [ ] Iterate based on actual results

## Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
