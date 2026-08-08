from __future__ import annotations

import ollama

import config
from interfaces import CompressionStage
from evaluator import cosine_similarity
from tokenizer_utils import count_tokens


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
                            "You are a prompt compression engine.\n\n"
                            "Primary goal: preserve meaning.\n"
                            "Secondary goal: reduce token count.\n\n"
                            "Never remove:\n"
                            "- instructions\n"
                            "- constraints\n"
                            "- negations\n"
                            "- examples\n"
                            "- numbers\n"
                            "- placeholders\n"
                            "- code snippets\n\n"
                            "Only remove redundancy and verbosity.\n"
                            "Do not add information.\n"
                            "Eliminate unnecessary information.\n"
                            "Reduce the prompt size as much as possible.\n"
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

            # Reject negative compression
            if count_tokens(rewritten) >= count_tokens(text):
                return text

            # Reject semantic drift
            similarity = cosine_similarity(text, rewritten)
            if similarity < 0.7:
                return text

            return rewritten

        except Exception as e:
            print(f"SemanticRewriter error: {e}")
            return text