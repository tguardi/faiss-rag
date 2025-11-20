# Semantic Search Notebook for SageMaker

## Quick Start Guide

The `semantic_search_notebook.ipynb` is a self-contained notebook that reads text files and provides semantic search functionality.

### Features

✓ **No pre-built data** - Reads from your own `.txt` files
✓ **FAISS HNSW index** - Fast approximate search with cosine similarity
✓ **Sentence Transformers** - High-quality embeddings
✓ **Interactive search** - Query loop for exploration
✓ **SageMaker ready** - Works in any Jupyter environment

### Setup in SageMaker

1. **Upload the notebook**:
   - Upload `semantic_search_notebook.ipynb` to your SageMaker notebook instance

2. **Run all cells** (first time):
   - Click "Run All" or execute cells sequentially
   - Cell 5 will automatically create a `documents/` folder
   - First run will download the embedding model (~80MB)
   - You'll see a warning that no `.txt` files were found

3. **Add your text files**:
   - Upload or create `.txt` files in the `documents/` folder that was created
   - Any plain text files will work

4. **Re-run cells 5-9**:
   - This loads your documents and builds the index
   - Now you're ready to search!

### Usage

```python
# After running all cells, search interactively:
Query> your search query here

# Or programmatically:
results = search("your query", top_k=5)
display_results("your query", results)
```

### Configuration Options

Edit cell 3 to customize:

```python
DOCUMENTS_DIR = "documents"     # Your folder path
MODEL_NAME = "all-MiniLM-L6-v2" # Fast, good quality
CHUNK_SIZE = 500                # Characters per chunk
CHUNK_OVERLAP = 100             # Overlap between chunks
TOP_K = 5                       # Default results count
```

### Alternative Models

For better quality (slower):
```python
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
```

For multilingual:
```python
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

### HNSW Parameters

Tune for speed vs quality in cell 3:

```python
HNSW_M = 32              # 16-64, higher = better recall
HNSW_EF_CONSTRUCTION = 64 # 40-500, higher = better quality
HNSW_EF_SEARCH = 64      # 10-500, higher = more accurate
```

### Tips for Colleagues

**First time users:**
1. Place 3-5 sample `.txt` files in `documents/`
2. Run all cells (takes ~2-3 minutes on first run)
3. Try the interactive search in the last cell
4. Experiment with different queries

**For production:**
- Increase `CHUNK_SIZE` for broader context (e.g., 1000)
- Decrease `CHUNK_SIZE` for precise matching (e.g., 200)
- Adjust `TOP_K` based on use case (3-20 typical)

**Troubleshooting:**
- If no documents found: Check folder path is correct
- If model download fails: Check internet connection
- If out of memory: Reduce `CHUNK_SIZE` or process fewer files

### What's Included

The notebook contains:

1. **Installation** - Auto-installs dependencies
2. **Document loading** - Reads all `.txt` files from folder
3. **Chunking** - Splits documents into overlapping segments
4. **Embedding** - Creates vector representations
5. **FAISS indexing** - Builds HNSW index for fast search
6. **Search functions** - Semantic search with scoring
7. **Interactive mode** - Query loop for exploration
8. **Export** - Save results to JSON (optional)

### Example Queries

Depending on your documents, try:
- "What are the main findings?"
- "key conclusions and recommendations"
- "technical implementation details"
- "risks and challenges"
- "future work and next steps"

### Export Results

```python
query = "your query"
results = search(query)
export_results(query, results, "results.json")
```

### Notebook Cells Overview

| Cell | Purpose | Run When |
|------|---------|----------|
| 1 | Install dependencies | First time only |
| 2-4 | Setup & config | Always |
| 5-6 | Load & chunk documents | When documents change |
| 7-8 | Load model & create embeddings | When documents change |
| 9 | Build FAISS index | When documents change |
| 10-12 | Search examples | Anytime |
| 13 | Interactive search | For exploration |
| 14 | Export results | Optional |

### Performance

**Small dataset (10-50 files):**
- Index build: 1-2 minutes
- Search: <100ms per query
- Memory: ~200MB

**Medium dataset (100-500 files):**
- Index build: 5-10 minutes
- Search: <200ms per query
- Memory: ~500MB-1GB

**Large dataset (1000+ files):**
- Index build: 20-60 minutes
- Search: <500ms per query
- Memory: 2-4GB

### Sharing with Colleagues

**Option 1: Share notebook + instructions**
```
semantic_search_notebook.ipynb
README_NOTEBOOK.md (this file)
```

**Option 2: Share with sample data**
```
semantic_search_notebook.ipynb
documents/
  ├── sample1.txt
  ├── sample2.txt
  └── sample3.txt
```

**Option 3: Pre-built index (advanced)**
- Run notebook once to build index
- Share notebook + documents + saved index
- Modify notebook to load pre-built index

### License

Same as parent project. This notebook is self-contained and can be used independently.
