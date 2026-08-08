from __future__ import annotations

import csv
import statistics
from pathlib import Path

from benchmark_prompts import BENCHMARK_PROMPTS
from evaluator import Evaluator
from pipeline import CompressionPipeline

from stages.rule_cleanup import RuleCleanupStage
from stages.perplexity_pruning import PerplexityPruningStage
from stages.semantic_rewriter import SemanticRewriter

RESULTS_DIR = Path(__file__).parent / "results"

def build_strategies():
    return {
        "rule_only": [
            RuleCleanupStage(),
        ],
        "perplexity_only": [
            PerplexityPruningStage(),
        ],
        "rewrite_only": [
            SemanticRewriter(),
        ],
        "hybrid": [
            RuleCleanupStage(),
            PerplexityPruningStage(),
            SemanticRewriter(),
        ],
    }

def run() -> None:
    evaluator = Evaluator()

    rows = []

    print(
        f"{'Strategy':<18}"
        f"{'Reduction':>12}"
        f"{'Similarity':>12}"
    )

    print("-" * 42)

    for strategy_name, stages in build_strategies().items():

        pipeline = CompressionPipeline(
            stages=stages,
            name=strategy_name,
        )

        reductions = []
        similarities = []

        for prompt in BENCHMARK_PROMPTS:

            result = pipeline.compress(prompt.text)

            evaluation = evaluator.evaluate(result)

            reductions.append(
                evaluation.token_reduction_pct
            )

            similarities.append(
                evaluation.semantic_similarity
            )

            rows.append(
                {
                    "strategy": strategy_name,
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "tokens_before": result.original_tokens,
                    "tokens_after": result.compressed_tokens,
                    "reduction_pct": round(
                        evaluation.token_reduction_pct,
                        4,
                    ),
                    "semantic_similarity": round(
                        evaluation.semantic_similarity,
                        4,
                    ),
                }
            )

        print(
            f"{strategy_name:<18}"
            f"{statistics.mean(reductions):>12.1%}"
            f"{statistics.mean(similarities):>12.4f}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        print("No benchmark results generated.")
        return

    output_file = RESULTS_DIR / "benchmark.csv"

    with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_file}")

if __name__ == "__main__":
    run()