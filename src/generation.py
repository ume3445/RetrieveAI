from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from . import config
from .retrieval import Chunk

_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "return_answer",
        "description": "Return the answer and the source passages that support it.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A concise, complete answer to the question.",
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer"},
                            "excerpt": {
                                "type": "string",
                                "description": "Verbatim short excerpt from the source chunk.",
                            },
                        },
                        "required": ["page", "excerpt"],
                    },
                    "description": "Up to 3 source passages that directly support the answer.",
                },
            },
            "required": ["answer", "citations"],
        },
    },
}


@dataclass
class Answer:
    text: str
    citations: list[dict]  # [{page: int, excerpt: str}]


def generate(query: str, chunks: list[Chunk]) -> Answer:
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    context = "\n\n---\n\n".join(
        f"[{i}] (page {c.page})\n{c.text}" for i, c in enumerate(chunks, 1)
    )

    completion = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context passages. "
                    "If the context is insufficient, say so."
                ),
            },
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"},
        ],
        tools=[_ANSWER_TOOL],
        tool_choice={"type": "function", "function": {"name": "return_answer"}},
        temperature=0.0,
    )

    payload = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    return Answer(text=payload["answer"], citations=payload.get("citations", []))
