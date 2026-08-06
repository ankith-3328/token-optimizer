from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def pruner():
    """Real distilGPT2 stage — session-scoped so the model loads once."""
    from stages.perplexity_pruning import PerplexityPruningStage
    return PerplexityPruningStage(target_ratio=0.7)

