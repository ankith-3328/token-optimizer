from __future__ import annotations

import pytest

from interfaces import CompressionStage


class PassThroughStage(CompressionStage):
    name = "passthrough"

    def run(self, text: str) -> str:
        return text


class HalvingStage(CompressionStage):
    """Deterministic fake compressor for pipeline tests."""
    name = "halving"

    def run(self, text: str) -> str:
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])


@pytest.fixture
def passthrough() -> CompressionStage:
    return PassThroughStage()


@pytest.fixture
def word_counter():
    return lambda text: len(text.split())


@pytest.fixture(scope="session")
def pruner():
    """Real distilGPT2 stage — session-scoped so the model loads once."""
    from stages.perplexity_pruning import PerplexityPruningStage
    return PerplexityPruningStage(target_ratio=0.5)
