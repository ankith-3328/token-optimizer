from __future__ import annotations

from typing import Callable, Sequence

from interfaces import CompressionStage, CompressionResult, StageMetrics
from tokenizer_utils import count_tokens


class CompressionPipeline:

    def __init__(
        self,
        stages: Sequence[CompressionStage],
        token_counter: Callable[[str], int] = count_tokens,
        name: str = "hybrid",
    ) -> None:
        self.stages = list(stages)
        self.token_counter = token_counter
        self.name = name

    def compress(self, text: str) -> CompressionResult:
        original_tokens = self.token_counter(text)
        current = text
        metrics: list[StageMetrics] = []

        for stage in self.stages:
            before = self.token_counter(current)
            try:
                current = stage.run(current)
            except Exception as exc:  # a broken stage must not kill the run
                print(f"[pipeline] {stage.name} failed: {exc!r} — passing through")
            after = self.token_counter(current)
            metrics.append(StageMetrics(stage.name, before, after))

        return CompressionResult(
            strategy_name=self.name,
            original_text=text,
            compressed_text=current,
            original_tokens=original_tokens,
            compressed_tokens=self.token_counter(current),
            stage_metrics=metrics,
        )