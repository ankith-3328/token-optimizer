from evaluator import Evaluator
from pipeline import CompressionPipeline

from stages.rule_cleanup import RuleCleanupStage
from stages.perplexity_pruning import PerplexityPruningStage
from stages.semantic_rewriter import SemanticRewriter

SAMPLE_PROMPT = """
Please carefully review this Java code in order to identify bugs,
performance bottlenecks, security issues,
and provide detailed recommendations for improvement.
"""

def main() -> None:
    pipeline = CompressionPipeline(
        [
            RuleCleanupStage(),
            PerplexityPruningStage(),
            SemanticRewriter(),
        ]
    )

    result = pipeline.compress(SAMPLE_PROMPT)

    evaluation = Evaluator().evaluate(result)

    print("\nOriginal Text")
    print("-" * 60)
    print(result.original_text)

    print("\nCompressed Text")
    print("-" * 60)
    print(result.compressed_text)

    print("\nMetrics")
    print("-" * 60)
    print(f"Strategy: {evaluation.strategy_name}")
    print(f"Original Tokens: {result.original_tokens}")
    print(f"Compressed Tokens: {result.compressed_tokens}")
    print(f"Reduction: {evaluation.token_reduction_pct:.2%}")
    print(f"Similarity: {evaluation.semantic_similarity:.4f}")

    print("\nStage Breakdown")
    print("-" * 60)

    for metric in result.stage_metrics:
        print(
            f"{metric.stage_name:<20}"
            f"{metric.tokens_before:>8}"
            f"{metric.tokens_after:>8}"
            f"{metric.reduction_pct:>10.1%}"
        )

if __name__ == "__main__":
    main()