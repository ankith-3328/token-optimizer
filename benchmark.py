from interfaces import CompressionResult
from evaluator import Evaluator
from stages.semantic_rewriter import SemanticRewriter
from tokenizer_utils import count_tokens


SAMPLE_PROMPT = """
Review this Java code,
identify bugs,
suggest performance improvements,
and explain your reasoning.
"""


def run_stage(stage, text: str) -> CompressionResult:
    compressed = stage.run(text)

    return CompressionResult(
        strategy_name=stage.name,
        original_text=text,
        compressed_text=compressed,
        original_tokens = count_tokens(text),
        compressed_tokens = count_tokens(compressed)
    )


def main():
    evaluator = Evaluator()

    stage = SemanticRewriter()

    result = run_stage(stage, SAMPLE_PROMPT)

    evaluation = evaluator.evaluate(result)

    print("\nOriginal:")
    print(result.original_text)

    print("\nCompressed:")
    print(result.compressed_text)

    print("\nMetrics")
    print("-" * 40)
    print(f"Strategy: {evaluation.strategy_name}")
    print(f"Reduction: {evaluation.token_reduction_pct:.2%}")
    print(f"Similarity: {evaluation.semantic_similarity:.4f}")


if __name__ == "__main__":
    main()