from __future__ import annotations

import ollama

import config
from interfaces import CompressionStage


class SemanticRewriter(CompressionStage):
    """
    LLM-based semantic compression using Ollama.
    """

    name = "semantic_rewriter"

    def run(self, text: str) -> str:
        if not text.strip():
            return text

        try:
            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a prompt compression engine. "
                            "Rewrite prompts using fewer words while preserving meaning. "
                            "Do not add information. "
                            "Eliminate unnecessary information."
                            "Reduce the prompt size as much as possible."
                            "Return only the rewritten prompt."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
            )

            rewritten = response["message"]["content"].strip()

            if not rewritten:
                return text

            return rewritten

        except Exception as e:
            print(f"SemanticRewriter error: {e}")
            return text