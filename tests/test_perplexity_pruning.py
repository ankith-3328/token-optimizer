from __future__ import annotations

import math
import re

import pytest

from stages.perplexity_pruning import (
    PerplexityPruningStage,
    _aggregate_to_words,
    _budget_prune,
    _reconstruct,
    ScoredUnit,
)


# ── Correctness of the protection layer ──────────────────────────────────────
def test_negation_survives_aggressive_pruning(pruner):
    """The flagship correctness case: deleting 'not' inverts the instruction."""
    stage = PerplexityPruningStage(target_ratio=0.3)
    out = stage.run("Summarize the ticket below. Do not include any PII whatsoever.")
    assert "not" in out.lower().split()


@pytest.mark.parametrize("word", ["never", "must", "only", "without", "except"])
def test_constraint_words_survive(word):
    stage = PerplexityPruningStage(target_ratio=0.3)
    out = stage.run(
        f"You should carefully consider the following and {word} respond in "
        f"a manner that is fully consistent with the given guidelines."
    )
    assert word in out.lower()


def test_template_placeholders_intact():
    stage = PerplexityPruningStage(target_ratio=0.4)
    out = stage.run(
        "You are helping {user_name} with a {issue_type} issue and you should "
        "respond in a {tone} tone that is appropriate for the situation."
    )
    for placeholder in ("{user_name}", "{issue_type}", "{tone}"):
        assert placeholder in out


def test_numbers_and_units_preserved():
    stage = PerplexityPruningStage(target_ratio=0.4)
    out = stage.run(
        "Please make sure that the response you write is limited to exactly "
        "250 words and costs no more than $1.50 per request."
    )
    assert "250" in out and "1.50" in out


# ── Budget controller ────────────────────────────────────────────────────────
def test_output_within_budget(pruner):
    text = (
        "Please carefully review the following code and provide a detailed "
        "analysis of any potential bugs, performance bottlenecks, or areas "
        "where meaningful improvements can be made to the implementation."
    )
    stage = PerplexityPruningStage(target_ratio=0.5)
    out = stage.run(text)
    assert len(out.split()) <= len(text.split())
    assert stage.last_achieved_ratio <= 1.0


def test_ratio_one_is_passthrough():
    stage = PerplexityPruningStage(target_ratio=1.0)
    text = "Analyze this code for bugs."
    assert stage.run(text) == text


def test_lower_ratio_removes_more(pruner):
    text = (
        "Please carefully review the following code and provide a detailed "
        "analysis of any potential bugs, performance bottlenecks, security "
        "vulnerabilities, or areas where improvements can be made."
    )
    loose = PerplexityPruningStage(target_ratio=0.8).run(text)
    tight = PerplexityPruningStage(target_ratio=0.4).run(text)
    assert len(tight.split()) < len(loose.split())


def test_budget_prune_uses_original_indices():
    """Regression: sorting by score must not renumber units."""
    units = [
        ScoredUnit("alpha", 0, 5, 0, 9.0, False, 0),
        ScoredUnit("beta", 6, 10, 1, 1.0, False, 0),   # lowest score, index 1
        ScoredUnit("gamma", 11, 16, 2, 8.0, False, 0),
    ]
    surviving = _budget_prune(units, target=2)
    assert surviving == {0, 2}          # 'beta' dropped, NOT the sorted-position-0


def test_protected_units_never_dropped_even_when_lowest():
    units = [
        ScoredUnit("not", 0, 3, 0, 0.01, True, 0),     # lowest score, protected
        ScoredUnit("review", 4, 10, 1, 5.0, False, 0),
        ScoredUnit("code", 11, 15, 2, 6.0, False, 0),
    ]
    assert 0 in _budget_prune(units, target=1)


# ── Ordering and reconstruction ──────────────────────────────────────────────
def test_original_order_preserved(pruner):
    text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet."
    out = PerplexityPruningStage(target_ratio=0.5).run(text)
    original = [w.strip(".,").lower() for w in text.split()]
    kept = [w.strip(".,").lower() for w in out.split()]
    positions = [original.index(w) for w in kept if w in original]
    assert positions == sorted(positions)


def test_reconstruct_uses_original_slices():
    text = "alpha beta gamma"
    units = [
        ScoredUnit("alpha", 0, 5, 0, 1.0, False, 0),
        ScoredUnit("beta", 6, 10, 1, 2.0, False, 0),
        ScoredUnit("gamma", 11, 16, 2, 3.0, False, 0),
    ]
    assert _reconstruct(text, units, {0, 2}) == "alpha gamma"


def test_no_orphaned_subwords(pruner):
    """BPE regression: output must contain no word fragments."""
    text = (
        "Identify security vulnerabilities and performance bottlenecks in the "
        "implementation, considering internationalization and accessibility."
    )
    out = PerplexityPruningStage(target_ratio=0.5).run(text)
    for word in re.findall(r"[A-Za-z]+", out):
        assert len(word) > 1 or word.lower() in {"a", "i"}


# ── Determinism, edge cases, robustness ──────────────────────────────────────
def test_deterministic(pruner):
    text = "Please carefully analyze the following code for any potential bugs."
    stage = PerplexityPruningStage(target_ratio=0.5)
    assert stage.run(text) == stage.run(text)


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "hello", "a"])
def test_degenerate_inputs_do_not_crash(text):
    PerplexityPruningStage(target_ratio=0.5).run(text)


def test_long_input_exceeds_context_window():
    """distilGPT2 caps at 1024 tokens — the sliding window must handle more."""
    text = ("The quick brown fox jumps over the lazy dog near the riverbank. " * 120)
    out = PerplexityPruningStage(target_ratio=0.5).run(text)
    assert out and len(out) < len(text)


def test_invalid_ratio_rejected():
    with pytest.raises(ValueError):
        PerplexityPruningStage(target_ratio=0.0)
    with pytest.raises(ValueError):
        PerplexityPruningStage(target_ratio=1.5)


def test_scores_are_exposed_for_demo(pruner):
    stage = PerplexityPruningStage(target_ratio=0.5)
    stage.run("Analyze this code for potential bugs and performance issues.")
    assert stage.last_units
    assert all(isinstance(u.surprisal, float) for u in stage.last_units)
    assert math.isinf(stage.last_units[0].surprisal)   # first unit always kept


# ── Aggregation ──────────────────────────────────────────────────────────────
def test_aggregate_merges_subwords():
    text = "vulnerabilities exist"
    offsets = [(0, 6), (6, 15), (15, 21)]        # vulner|abilities| exist
    scores = [3.0, 5.0, 7.0]
    words = _aggregate_to_words(text, offsets, scores)
    assert [w[0] for w in words] == ["vulnerabilities", "exist"]
    assert words[0][3] == pytest.approx(4.0)     # mean of 3.0 and 5.0