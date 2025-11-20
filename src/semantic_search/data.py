from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Iterable


@dataclass
class Document:
    """Represents a single document with optional structured metadata."""

    title: str
    content: str
    speaker: str | None = None
    author: str | None = None
    location: str | None = None
    date: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def attribution(self) -> str:
        candidates = (
            self.speaker,
            self.author,
            self.metadata.get("speaker"),
            self.metadata.get("author"),
            self.metadata.get("source"),
            self.metadata.get("organization"),
        )
        result = _first_non_empty(candidates)
        return result or "Unknown source"

    @property
    def display_date(self) -> str:
        result = _first_non_empty((self.date, self.metadata.get("date")))
        return result or "Unknown date"

    @property
    def link(self) -> str:
        return _first_non_empty((self.url, self.metadata.get("url"))) or ""


def load_documents(path: Path, dataset_key: str | None = None) -> list[Document]:
    """Load documents from a JSON file, auto-detecting the payload list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = _extract_entries(data, dataset_key)
    documents: list[Document] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue

        metadata_field = entry.get("metadata")
        metadata = metadata_field if isinstance(metadata_field, dict) else {}

        extras = {
            key: value
            for key, value in entry.items()
            if key not in _RESERVED_FIELDS
        }
        metadata = {**extras, **metadata}

        title = _first_non_empty(
            (
                entry.get("title"),
                entry.get("name"),
                metadata.get("title"),
            ),
            default=f"Document {idx + 1}",
        )
        content = _first_non_empty(
            (
                entry.get("content"),
                entry.get("text"),
                entry.get("body"),
                metadata.get("content"),
            )
        )
        if not content:
            continue

        documents.append(
            Document(
                title=title,
                content=content,
                speaker=_first_non_empty((entry.get("speaker"), metadata.get("speaker"))),
                author=_first_non_empty(
                    (
                        entry.get("author"),
                        entry.get("authors"),
                        metadata.get("author"),
                    )
                ),
                location=_first_non_empty((entry.get("location"), metadata.get("location"))),
                date=_first_non_empty((entry.get("date"), metadata.get("date"))),
                url=_first_non_empty((entry.get("url"), metadata.get("url"))),
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            "No documents were loaded from the provided file. "
            "Double-check the JSON structure or use --data-key to specify the list field."
        )
    return documents


def load_speeches(path: Path) -> list[Document]:
    """Backward-compatible alias for the previous API."""

    return load_documents(path, dataset_key="speeches")


def _extract_entries(data: Any, dataset_key: str | None) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object or array at the top level")
    if dataset_key:
        entries = data.get(dataset_key)
        if not isinstance(entries, list):
            raise ValueError(f"JSON key '{dataset_key}' does not contain a list of documents")
        return [entry for entry in entries if isinstance(entry, dict)]
    for candidate in ("documents", "speeches", "items", "records", "data"):
        entries = data.get(candidate)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    raise ValueError(
        "Unable to find a list of documents in the JSON payload. "
        "Pass --data-key <key> to select the correct field."
    )


def _first_non_empty(values: Iterable[Any], *, default: str | None = None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


_RESERVED_FIELDS = {
    "title",
    "name",
    "content",
    "text",
    "body",
    "speaker",
    "author",
    "authors",
    "location",
    "date",
    "url",
    "metadata",
}
