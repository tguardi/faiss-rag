# Documents Folder

This folder is for the **Jupyter notebook** (`semantic_search_notebook.ipynb`).

## Purpose

Place your `.txt` files here to use with the standalone Jupyter notebook for semantic search in SageMaker or other Jupyter environments.

## Usage

1. **Open the notebook**: `semantic_search_notebook.ipynb`
2. **Run all cells** (first time)
3. **Add your `.txt` files** to this folder
4. **Re-run cells 5-9** to load and index your documents
5. **Search** using the notebook's search cells

## File Format

- **Extension**: `.txt` (plain text files)
- **Encoding**: UTF-8
- **Content**: Any text documents you want to search

## Example

```
documents/
├── README.md (this file)
├── report_2024.txt
├── meeting_notes.txt
└── research_paper.txt
```

## Not Using the Notebook?

If you're using the CLI tool (`faiss-search`), you **don't need** this folder. The CLI uses JSON files specified with `--data-file` flag.

## Note

This folder is **not used** by the main CLI tool. It's specifically for the Jupyter notebook workflow.
