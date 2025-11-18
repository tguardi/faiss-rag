# Local Model Storage

This directory is for storing embedding models locally to enable:
- **Offline usage**: Run without internet connection
- **Portability**: Package the project with models included
- **Version control**: Pin specific model versions
- **Faster startup**: Skip download time

## Usage

The code automatically checks this directory before downloading from Hugging Face.

### Option 1: Copy from Hugging Face Cache

If you already have models cached:

```bash
# Find your cached model
ls ~/.cache/huggingface/hub/

# Copy to local models directory
cp -r ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/* \
  models/all-MiniLM-L6-v2/
```

### Option 2: Download Directly

```python
from sentence_transformers import SentenceTransformer

# Download and save to models/
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('models/all-MiniLM-L6-v2')
```

### Option 3: Manual Download

1. Download model files from [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
2. Place all files in `models/all-MiniLM-L6-v2/`

Required files:
- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `tokenizer_config.json`
- `vocab.txt`
- `special_tokens_map.json`
- Other model-specific files

## Expected Structure

```
models/
├── README.md                    # This file
└── all-MiniLM-L6-v2/           # Default model
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ...
```

## Using Different Models

To use a different model:

1. Download/copy model to `models/<model-name>/`
2. Update `MODEL_NAME` in `src/semantic_search/index.py`
3. Rebuild index: `uv run fed-speech-search --rebuild-index`

## Note

Models in this directory are excluded from git (see `.gitignore`). This README is the only tracked file.

To share models with others, document the model name and let them download separately.
