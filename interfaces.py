"""Shared contract. Locked on Day 1 — do not change without telling the other two."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class CompressionStage(ABC):
    """Every stage implements exactly this. Input str, output str."""

    #: Human-readable label used in the metrics waterfall.
    name: str = "stage"

    @abstractmethod
    def run(self, text: str) -> str:
        """Compress `text`. Must be safe to call on raw, uncleaned input."""
        raise NotImplementedError


@dataclass
class StageMetrics:
    stage_name: str
    tokens_before: int
    tokens_after: int

    @property
    def reduction_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)


@dataclass
class CompressionResult:
    strategy_name: str
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    stage_metrics: list[StageMetrics] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)


@dataclass
class EvalResult:
    strategy_name: str
    token_reduction_pct: float
    semantic_similarity: float