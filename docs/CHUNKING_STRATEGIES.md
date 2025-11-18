# Chunking Strategies Deep Dive

## Why Chunking Matters

When performing semantic search over long documents, chunking is crucial because:

1. **Embedding models have limits**: Most models work best on shorter text (sentences to paragraphs)
2. **Specificity vs. context trade-off**: Large chunks preserve context but dilute specific signals
3. **Search granularity**: You want to find the relevant *part* of a document, not just relevant documents

## Available Strategies

### 1. Full Speech Chunker

**Implementation**: `FullSpeechChunker`

Treats each entire speech as a single chunk.

#### How it works
```python
# Pseudocode
def chunk_speech(speech):
    return [Chunk(text=speech.content, chunk_index=0)]
```

#### Parameters
None - one chunk per speech

#### Best for
- Finding speeches by overall topic/theme
- When you need complete context
- Exploratory searches ("What did they say about X?")

#### Example Results

Query: "monetary policy framework"

```
✓ Returns: Entire speeches about monetary policy frameworks
✓ Context: Full speech preserved
✗ Precision: May return speeches that mention topic briefly
✗ Ranking: Hard to distinguish highly relevant parts
```

#### Statistics (typical)
- Chunks per speech: 1
- Avg chunk size: 15,000-30,000 characters
- Total chunks (198 speeches): ~198

---

### 2. Paragraph Chunker (Default)

**Implementation**: `ParagraphChunker`

Groups consecutive paragraphs into semantically coherent chunks of 450-1100 characters.

#### How it works
```python
# Pseudocode
def chunk_speech(speech):
    paragraphs = speech.content.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk + para) > MAX_SIZE:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += para

    return chunks
```

#### Parameters
- `MIN_CHUNK_SIZE`: 450 characters
- `MAX_CHUNK_SIZE`: 1100 characters

These are tuned for the `all-MiniLM-L6-v2` model which works well on paragraph-length text.

#### Best for
- General-purpose semantic search
- Balanced precision and context
- Most use cases

#### Example Results

Query: "inflation expectations"

```
✓ Returns: Specific paragraphs discussing inflation expectations
✓ Context: Related sentences grouped together
✓ Precision: Highly relevant passages
✓ Ranking: Clear distinction between highly/moderately relevant
```

#### Statistics (typical)
- Chunks per speech: 15-30
- Avg chunk size: 700-800 characters
- Total chunks (198 speeches): ~4,500-5,000

---

### 3. Sliding Window Chunker

**Implementation**: `SlidingWindowChunker`

Creates overlapping windows of ~150 words with 30-word overlap.

#### How it works
```python
# Pseudocode
def chunk_speech(speech):
    words = speech.content.split()
    chunks = []

    for i in range(0, len(words), STRIDE):
        window = words[i:i+WINDOW_SIZE]
        chunks.append(" ".join(window))

    return chunks
```

#### Parameters
- `WINDOW_SIZE`: 150 words (~900-1000 characters)
- `STRIDE`: 120 words (= 30 word overlap)

#### Best for
- Finding specific phrases or technical terms
- Maximum recall (don't miss anything)
- When semantic boundaries aren't important

#### Example Results

Query: "stress testing methodology"

```
✓ Returns: Precise passages with those exact concepts
✓ Coverage: Overlapping ensures phrases aren't split
✗ Context: May split mid-thought
✗ Volume: Many more results to sift through
```

#### Statistics (typical)
- Chunks per speech: 80-120
- Avg chunk size: 900-1000 characters
- Total chunks (198 speeches): ~18,000-20,000

---

## Comparative Analysis

### Retrieval Characteristics

| Chunker | Precision | Recall | Speed | Context |
|---------|-----------|--------|-------|---------|
| Full | Low | Medium | Fast | Maximum |
| Paragraphs | High | High | Medium | Good |
| Sliding Window | Very High | Maximum | Slower | Limited |

### Query Type Performance

#### Broad Thematic Queries
Example: "economic outlook"

**Best**: Full or Paragraphs
- Full: Returns complete speeches on topic
- Paragraphs: Returns relevant sections with good context

**Worst**: Sliding Window
- Too granular, misses forest for trees

#### Specific Technical Queries
Example: "quantitative tightening balance sheet runoff"

**Best**: Sliding Window or Paragraphs
- Sliding Window: Captures specific phrases reliably
- Paragraphs: Good if phrase appears in coherent section

**Worst**: Full
- Signal diluted in large text

#### Question-like Queries
Example: "what are the risks to financial stability"

**Best**: Paragraphs
- Natural question-answer structure often paragraph-based

**Medium**: Sliding Window, Full
- May work but less optimized

### Computational Trade-offs

```
Index Build Time:
  Full: ~30 seconds (fewest chunks)
  Paragraphs: ~2 minutes (medium chunks)
  Sliding Window: ~8 minutes (most chunks)

Search Time (per query):
  Full: <0.1s
  Paragraphs: ~0.2s
  Sliding Window: ~0.5s

Storage:
  Full: ~30KB index + ~15KB metadata
  Paragraphs: ~350KB index + ~400KB metadata
  Sliding Window: ~1.4MB index + ~2MB metadata
```

## Tuning Parameters

### For Paragraph Chunker

Edit `src/semantic_search/chunkers.py`:

```python
class ParagraphChunker(BaseChunker):
    MIN_CHUNK_SIZE = 450  # Increase for more context
    MAX_CHUNK_SIZE = 1100  # Decrease for more precision
```

**Effects**:
- **Increase MIN/MAX**: Larger chunks, more context, less precision
- **Decrease MIN/MAX**: Smaller chunks, more precision, less context

**Recommendations**:
- Short queries → smaller chunks (300-800)
- Complex topics → larger chunks (600-1500)
- Technical terms → smaller chunks
- Abstract concepts → larger chunks

### For Sliding Window Chunker

```python
class SlidingWindowChunker(BaseChunker):
    WINDOW_SIZE = 150  # words per chunk
    OVERLAP = 30       # words of overlap
```

**Effects**:
- **Increase WINDOW_SIZE**: More context, fewer chunks
- **Decrease WINDOW_SIZE**: More precision, more chunks
- **Increase OVERLAP**: Better coverage, more redundancy
- **Decrease OVERLAP**: Faster, may miss split phrases

**Recommendations**:
- Phrase search → more overlap (50-75 words)
- Speed priority → less overlap (10-20 words)
- Long documents → larger windows (200-300 words)

## Creating Custom Chunkers

### Template

```python
from semantic_search.chunkers import BaseChunker, Chunk
from semantic_search.data import Speech

class MyChunker(BaseChunker):
    def chunk_speech(self, speech: Speech, speech_index: int) -> list[Chunk]:
        """
        Custom chunking logic.

        Args:
            speech: Speech object with .content, .title, etc.
            speech_index: Index of speech in dataset

        Returns:
            List of Chunk objects
        """
        chunks = []

        # Your chunking logic here
        # Example: Split by sentence count
        sentences = speech.content.split('. ')

        for i in range(0, len(sentences), 5):  # 5 sentences per chunk
            chunk_sentences = sentences[i:i+5]
            chunk_text = '. '.join(chunk_sentences)

            chunks.append(Chunk(
                text=chunk_text,
                speech_index=speech_index,
                chunk_index=i // 5
            ))

        return chunks
```

### Register Your Chunker

Add to `available_chunkers()` in `chunkers.py`:

```python
def available_chunkers() -> dict[str, str]:
    return {
        "full": "one chunk per speech",
        "paragraphs": "paragraph-based chunks (450-1100 chars)",
        "sliding_window": "overlapping 150-word windows",
        "my_custom": "your description here",  # Add this
    }

def get_chunker(name: str) -> BaseChunker:
    chunkers = {
        "full": FullSpeechChunker(),
        "paragraphs": ParagraphChunker(),
        "sliding_window": SlidingWindowChunker(),
        "my_custom": MyChunker(),  # Add this
    }
    return chunkers[name]
```

### Advanced Chunker Ideas

1. **Semantic Sentence Chunker**: Use NLP to group semantically related sentences
2. **Topic-based Chunker**: Cluster paragraphs by topic using embeddings
3. **Fixed-token Chunker**: Use proper tokenizer for exact token counts
4. **Section-based Chunker**: Split on headers/sections if structured
5. **Hybrid Chunker**: Combine strategies (e.g., paragraphs within sections)

## Evaluation Framework

### Qualitative Metrics

For a set of test queries, measure:

1. **Relevance@k**: How many of top-k results are relevant?
2. **MRR (Mean Reciprocal Rank)**: Where does first relevant result appear?
3. **Coverage**: Does strategy find all relevant passages?

### Example Evaluation

```python
test_queries = [
    "inflation expectations",
    "stress testing methodology",
    "interest rate decision",
]

for chunker in ["full", "paragraphs", "sliding_window"]:
    print(f"\nEvaluating {chunker}")

    for query in test_queries:
        results = search(query, chunker=chunker, top_k=5)

        # Manual labeling: Are results relevant?
        relevant = count_relevant_results(results)

        print(f"  {query}: {relevant}/5 relevant")
```

## Best Practices

1. **Start with paragraphs chunker**: Good default for most cases
2. **Use benchmark script**: Compare strategies objectively
3. **Consider your queries**: Match strategy to query type
4. **Tune incrementally**: Change one parameter at a time
5. **Measure real performance**: Use your actual queries
6. **Document your findings**: Keep notes on what works

## Further Reading

- [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)
- [OpenAI Cookbook: Question Answering](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
