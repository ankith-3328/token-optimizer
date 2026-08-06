from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from interfaces import CompressionResult, EvalResult


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


def cosine_similarity(a: str, b: str) -> float:
    """
    Measure semantic similarity between two texts.

    Returns:
        1.0 -> identical meaning
        0.0 -> unrelated
       -1.0 -> opposite direction (rare in embeddings)
    """
    if not a.strip() or not b.strip():
        return 0.0

    embeddings = _embedder().encode(
        [a, b],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return float(np.dot(embeddings[0], embeddings[1]))


class Evaluator:
    def evaluate(
        self,
        result: CompressionResult,
    ) -> EvalResult:

        similarity = cosine_similarity(
            result.original_text,
            result.compressed_text,
        )

        return EvalResult(
            strategy_name=result.strategy_name,
            token_reduction_pct=result.reduction_pct,
            semantic_similarity=similarity,
        )