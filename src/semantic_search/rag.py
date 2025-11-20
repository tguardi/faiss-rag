from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import json

import boto3


DEFAULT_MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"
ANTHROPIC_VERSION = "bedrock-2023-05-31"


@dataclass
class RagSegment:
    title: str
    attribution: str
    date: str
    url: str
    chunk_index: int
    score: float
    text: str

    def to_prompt_block(self, position: int) -> str:
        url_line = f"URL: {self.url}\n" if self.url else ""
        return (
            f"[{position}] {self.title} — {self.attribution} ({self.date}) "
            f"(chunk {self.chunk_index + 1}, score={self.score:.3f})\n"
            f"{url_line}Excerpt:\n{self.text.strip()}\n"
        )


class BedrockRAGClient:
    """Simple helper for calling Amazon Bedrock with Anthropic Claude Sonnet."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str | None = None):
        self.model_id = model_id
        session = boto3.session.Session(region_name=region)
        self._client = session.client("bedrock-runtime")

    def generate_response(
        self,
        question: str,
        contexts: Sequence[RagSegment],
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 600,
    ) -> str:
        if not contexts:
            raise ValueError("No context chunks supplied for RAG response.")

        prompt = self._build_prompt(question, contexts)
        body = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        }

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        return self._extract_text(payload).strip()

    @staticmethod
    def _build_prompt(question: str, contexts: Sequence[RagSegment]) -> str:
        instructions = (
            "You are assisting with a semantic search demo over a curated document collection. "
            "Answer the user's question using only the provided excerpts. "
            "Cite the relevant document titles or sources inline. "
            "If the excerpts do not contain the answer, say so explicitly."
        )
        context_blocks = "\n".join(
            segment.to_prompt_block(i + 1) for i, segment in enumerate(contexts)
        )
        return (
            f"{instructions}\n\n"
            f"Context:\n{context_blocks}\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    @staticmethod
    def _extract_text(payload: dict) -> str:
        """Extracts concatenated text output from a Bedrock response payload."""
        output_sections = payload.get("output", [])
        texts: list[str] = []
        for section in output_sections:
            for chunk in section.get("content", []):
                if chunk.get("type") == "text":
                    texts.append(chunk.get("text", ""))
        if texts:
            return "\n".join(texts)

        # Fallback for different schema variants.
        if "content" in payload:
            for block in payload["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
        if not texts:
            raise ValueError(f"Unable to parse Bedrock response: {payload}")
        return "\n".join(texts)
