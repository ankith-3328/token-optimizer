from evaluator import cosine_similarity
from evaluator import Evaluator
from interfaces import CompressionResult

def test_identical_sentences():
    score = cosine_similarity(
        "Review the code for bugs.",
        "Review the code for bugs.",
    )

    assert score > 0.95


def test_similar_sentences():
    score = cosine_similarity(
        "Review the code for bugs.",
        "Check the code for issues.",
    )

    assert score > 0.60


def test_unrelated_sentences():
    score = cosine_similarity(
        "Review the code for bugs.",
        "Banana apple orange mango.",
    )

    assert score < 0.40


def test_empty_string():
    score = cosine_similarity(
        "",
        "Review the code for bugs.",
    )

    assert score == 0.0


def test_whitespace_string():
    score = cosine_similarity(
        "   ",
        "Review the code for bugs.",
    )

    assert score == 0.0


def test_similarity_ordering():
    identical = cosine_similarity(
        "Review the code for bugs.",
        "Review the code for bugs.",
    )

    similar = cosine_similarity(
        "Review the code for bugs.",
        "Check the code for issues.",
    )

    unrelated = cosine_similarity(
        "Review the code for bugs.",
        "Banana apple orange mango.",
    )

    assert identical > similar
    assert similar > unrelated


def test_evaluator_returns_eval_result():
    evaluator = Evaluator()

    compression_result = CompressionResult(
        strategy_name="test_strategy",
        original_text="Review the code for bugs.",
        compressed_text="Check the code for issues.",
        original_tokens=100,
        compressed_tokens=60,
    )

    result = evaluator.evaluate(compression_result)

    assert result.strategy_name == "test_strategy"
    assert result.token_reduction_pct == 0.4
    assert result.semantic_similarity > 0.6


def test_evaluator_no_reduction():
    evaluator = Evaluator()

    compression_result = CompressionResult(
        strategy_name="test_strategy",
        original_text="Review the code for bugs.",
        compressed_text="Review the code for bugs.",
        original_tokens=100,
        compressed_tokens=100,
    )

    result = evaluator.evaluate(compression_result)

    assert result.token_reduction_pct == 0.0
    assert result.semantic_similarity > 0.95