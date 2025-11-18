from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import List


@dataclass
class Speech:
    """Represents a single speech along with rendered text content."""

    title: str
    speaker: str
    location: str
    date: str
    url: str
    content: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_speeches(path: Path) -> list[Speech]:
    """Load speeches from the static JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Speech(**raw) for raw in data.get("speeches", [])]
