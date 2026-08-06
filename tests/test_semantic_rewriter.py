from unittest.mock import patch

from evaluator import Evaluator
from interfaces import CompressionResult
from stages.semantic_rewriter import SemanticRewriter


@patch("stages.semantic_rewriter.ollama.chat")
def test_empty_string(mock_chat):
    rewriter = SemanticRewriter()

    assert rewriter.run("") == ""

    mock_chat.assert_not_called()


@patch("stages.semantic_rewriter.ollama.chat")
def test_whitespace_string(mock_chat):
    rewriter = SemanticRewriter()

    assert rewriter.run("   ") == "   "

    mock_chat.assert_not_called()


@patch("stages.semantic_rewriter.ollama.chat")
def test_returns_string(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": "Review code for bugs."
        }
    }

    rewriter = SemanticRewriter()

    result = rewriter.run(
        "Please review the code and identify bugs."
    )

    assert isinstance(result, str)


@patch("stages.semantic_rewriter.ollama.chat")
def test_non_empty_output(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": "Review code for bugs."
        }
    }

    rewriter = SemanticRewriter()

    result = rewriter.run(
        "Please review the code and identify bugs."
    )

    assert len(result.strip()) > 0


@patch("stages.semantic_rewriter.ollama.chat")
def test_rewriter_does_not_expand_excessively(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": (
                "Review Java code, identify bugs, "
                "suggest performance improvements."
            )
        }
    }

    rewriter = SemanticRewriter()

    original = (
        "Please review the following Java code, "
        "identify bugs, suggest performance improvements, "
        "and explain your reasoning."
    )

    rewritten = rewriter.run(original)

    assert len(rewritten.split()) <= len(original.split()) + 5


@patch("stages.semantic_rewriter.ollama.chat")
def test_semantic_rewriter_evaluator_flow(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": (
                "Review Java code, identify bugs, "
                "suggest performance improvements."
            )
        }
    }

    rewriter = SemanticRewriter()

    original = (
        "Please review the following Java code, "
        "identify bugs, suggest performance improvements."
    )

    rewritten = rewriter.run(original)

    result = CompressionResult(
        strategy_name="semantic_rewriter",
        original_text=original,
        compressed_text=rewritten,
        original_tokens=len(original.split()),
        compressed_tokens=len(rewritten.split()),
    )

    evaluation = Evaluator().evaluate(result)

    assert evaluation.strategy_name == "semantic_rewriter"
    assert 0.0 <= evaluation.semantic_similarity <= 1.0