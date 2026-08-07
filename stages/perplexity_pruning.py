from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from interfaces import CompressionStage

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_CHARS = re.compile(r"\w")


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScoredUnit:
    """One prunable unit — a whole word, with its position in the source text."""

    text: str
    start: int          # char offset into the original string
    end: int
    index: int          # position in document order
    surprisal: float
    protected: bool
    sentence_id: int


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — model loading (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_scorer():
    """Load distilGPT2 once per process. ~350 MB, CPU, no API cost."""
    tokenizer = AutoTokenizer.from_pretrained(config.SCORER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.SCORER_MODEL)
    model.eval()
    model.to("cpu")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Layers 2 + 3 — surprisal scoring with a strided sliding window
# ─────────────────────────────────────────────────────────────────────────────
def _token_surprisals(ids: torch.Tensor, model) -> list[float]:
    """Return -log P(t_i | t_<i) for every token.

    distilGPT2's context is 1024 tokens. For longer inputs we slide a window
    forward by STRIDE and keep only the scores for tokens that window is the
    first to reach — so every token is scored with the most left-context
    available rather than a cold start.

    Token 0 has no left context by construction; we give it +inf so it is
    always kept.
    """
    n = int(ids.size(0))
    scores = [math.inf] * n
    max_ctx, stride = config.SCORER_MAX_CTX, config.SCORER_STRIDE

    start, highest_scored = 0, 0
    while highest_scored < n - 1:
        end = min(start + max_ctx, n)
        window = ids[start:end].unsqueeze(0)

        with torch.no_grad():
            logits = model(window).logits

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        # logits[:, :-1] predicts ids[:, 1:]  ← the shift-by-one alignment
        targets = window[0, 1:].unsqueeze(-1)
        window_scores = -log_probs[0, :-1].gather(-1, targets).squeeze(-1)

        for k in range(int(window_scores.size(0))):
            abs_i = start + 1 + k
            if abs_i > highest_scored:
                scores[abs_i] = float(window_scores[k])

        highest_scored = end - 1
        if end >= n:
            break
        start += stride

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — subword → word aggregation  ★ the readability-critical step
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate_to_words(
    text: str,
    offsets: list[tuple[int, int]],
    token_scores: list[float],
) -> list[tuple[str, int, int, float]]:
    """Group BPE tokens into whole words using character offsets.

    GPT-2's byte-level BPE encodes the leading space as part of the *next*
    token (e.g. " exist" is a single token), so consecutive tokens' offsets
    are contiguous at word boundaries, not gapped. A new word therefore
    starts either when there's a genuine offset gap, OR when the token's
    own text begins with a whitespace character — and that leading space is
    trimmed off the stored span so it doesn't become part of the word.

    Without this, "vulnerabilities" -> "vulner|abilities" can be pruned
    apart and the output becomes gibberish — or worse, entire sentences
    collapse into one unsplit "word" and pruning does nothing at all.
    """
    words: list[tuple[str, int, int, float]] = []
    cur_start = cur_end = None
    cur_scores: list[float] = []

    for (start, end), score in zip(offsets, token_scores):
        if start == end:                       # special/empty token
            continue

        piece_start = start
        if text[piece_start].isspace():
            piece_start += 1
        if piece_start >= end:                 # token was pure whitespace
            continue

        starts_new_word = (
            cur_end is None
            or start > cur_end
            or text[start].isspace()
        )
        if starts_new_word:
            if cur_start is not None:
                words.append((text[cur_start:cur_end], cur_start, cur_end,
                              _mean_inf(cur_scores)))
            cur_start, cur_end, cur_scores = piece_start, end, [score]
        else:
            cur_end = end
            cur_scores.append(score)

    if cur_start is not None:
        words.append((text[cur_start:cur_end], cur_start, cur_end,
                      _mean_inf(cur_scores)))
    return words


def _mean_inf(values: list[float]) -> float:
    """Mean that propagates infinity (an always-keep marker)."""
    if any(math.isinf(v) for v in values):
        return math.inf
    return sum(values) / len(values) if values else math.inf


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 — protection
# ─────────────────────────────────────────────────────────────────────────────
def _protected_char_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges that must survive regardless of surprisal."""
    spans: list[tuple[int, int]] = []
    for pattern in config.PROTECTED_PATTERNS:
        spans.extend((m.start(), m.end()) for m in pattern.finditer(text))
    return spans


def _is_protected(
    word: str,
    start: int,
    end: int,
    protected_spans: list[tuple[int, int]],
    sentence_spans: list[tuple[int, int]],
) -> bool:
    """Veto removal of negations, constraints, numbers, placeholders, proper nouns.

    Mid-sentence capitalization is a cheap proper-noun heuristic — but a word
    capitalized only because it starts its own sentence isn't a proper noun,
    it's just grammar. We exclude sentence-initial words from that check so
    we don't over-protect (and artificially inflate the achievable ratio).
    """
    normalized = re.sub(r"[^\w']", "", word).lower()
    if normalized in config.PROTECTED_WORDS:
        return True
    if any(s < end and start < e for s, e in protected_spans):
        return True

    sid = _sentence_id_for(start, sentence_spans)
    sentence_start = sentence_spans[sid][0]
    is_sentence_initial = start == sentence_start

    if word[:1].isupper() and not is_sentence_initial and _WORD_CHARS.search(word):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — sentence segmentation and the coarse pass
# ─────────────────────────────────────────────────────────────────────────────
def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for match in _SENTENCE_SPLIT.finditer(text):
        if match.start() > pos:
            spans.append((pos, match.start()))
        pos = match.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans or [(0, len(text))]


def _sentence_id_for(offset: int, spans: list[tuple[int, int]]) -> int:
    for i, (start, end) in enumerate(spans):
        if start <= offset < end:
            return i
    return len(spans) - 1


def _coarse_prune(units: list[ScoredUnit], n_sentences: int, target: int) -> set[int]:
    """Drop whole low-information sentences. Returns the set of dropped ids.

    We only drop a sentence if doing so keeps us at or above the target — the
    fine pass should still have work to do, and overshooting here throws away
    more than the budget asked for.
    """
    if not config.COARSE_PASS_ENABLED or n_sentences < 3:
        return set()

    by_sentence: dict[int, list[ScoredUnit]] = {}
    for unit in units:
        by_sentence.setdefault(unit.sentence_id, []).append(unit)

    def sentence_score(sid: int) -> float:
        members = by_sentence[sid]
        finite = [u.surprisal for u in members if not math.isinf(u.surprisal)]
        return sum(finite) / len(finite) if finite else math.inf   # length-normalized

    candidates = sorted(by_sentence, key=sentence_score)
    remaining, dropped = len(units), set()

    for sid in candidates:
        if config.COARSE_KEEP_FIRST and sid == 0:
            continue
        if config.COARSE_KEEP_LAST and sid == n_sentences - 1:
            continue
        members = by_sentence[sid]
        if any(u.protected for u in members):
            continue
        if remaining - len(members) < target:
            continue                                   # would overshoot
        dropped.add(sid)
        remaining -= len(members)

    return dropped


# ─────────────────────────────────────────────────────────────────────────────
# Layer 7 — budget controller + ordered reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def _budget_prune(units: list[ScoredUnit], target: int) -> set[int]:
    """Greedily drop the lowest-surprisal removable units until target is met.

    Returns the set of *surviving* unit indices.

    Two things this gets right that a naive version does not:
      * indices are the units' ORIGINAL document positions, carried in
        ScoredUnit.index — sorting by score does not renumber them;
      * protection is a hard veto, so if the target is unreachable we stop
        and report the achieved ratio rather than deleting a protected word
        or looping forever.
    """
    surviving = {u.index for u in units}
    to_remove = len(units) - target
    if to_remove <= 0:
        return surviving

    for unit in sorted(units, key=lambda u: u.surprisal):
        if to_remove <= 0:
            break
        if unit.protected or math.isinf(unit.surprisal):
            continue
        surviving.discard(unit.index)
        to_remove -= 1

    return surviving


def _reconstruct(text: str, units: list[ScoredUnit], surviving: set[int]) -> str:
    """Rebuild the string from original character slices, in original order."""
    parts: list[str] = []
    prev_end: int | None = None

    for unit in units:                       # already in document order
        if unit.index not in surviving:
            continue
        if prev_end is not None:
            gap = text[prev_end:unit.start]
            parts.append("\n" if "\n" in gap else " ")
        parts.append(text[unit.start:unit.end])
        prev_end = unit.end

    out = "".join(parts)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


# ─────────────────────────────────────────────────────────────────────────────
# The stage
# ─────────────────────────────────────────────────────────────────────────────
class PerplexityPruningStage(CompressionStage):
    """LLMLingua-style perplexity pruning with an internal budget controller."""

    name = "perplexity_pruning"

    def __init__(self, target_ratio: float | None = None) -> None:
        if target_ratio is not None and not 0.0 < target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        self.target_ratio = (
            config.TARGET_RATIO if target_ratio is None else target_ratio
        )
        #: Populated after each run — read by the notebook demo and benchmark.
        #: Exposing it here keeps the str -> str contract intact.
        self.last_units: list[ScoredUnit] = []
        self.last_achieved_ratio: float | None = None

    # -- public API ----------------------------------------------------------
    def run(self, text: str) -> str:
        if not text or not text.strip() or self.target_ratio >= 1.0:
            self.last_units, self.last_achieved_ratio = [], 1.0
            return text

        units = self.score(text)
        if len(units) < 2:
            self.last_achieved_ratio = 1.0
            return text

        target = max(1, math.ceil(len(units) * self.target_ratio))
        sentence_spans = _sentence_spans(text)

        dropped_sentences = _coarse_prune(units, len(sentence_spans), target)
        survivors_after_coarse = [
            u for u in units if u.sentence_id not in dropped_sentences
        ]

        surviving = _budget_prune(survivors_after_coarse, target)
        self.last_achieved_ratio = len(surviving) / len(units)

        return _reconstruct(text, units, surviving)

    def score(self, text: str) -> list[ScoredUnit]:
        """Score `text` into prunable word units. Public for the demo notebook."""
        tokenizer, model = _load_scorer()
        encoded = tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            truncation=False,
        )
        ids = encoded["input_ids"][0]
        offsets = [tuple(pair) for pair in encoded["offset_mapping"][0].tolist()]

        token_scores = _token_surprisals(ids, model)
        words = _aggregate_to_words(text, offsets, token_scores)

        sentence_spans = _sentence_spans(text)
        protected_spans = _protected_char_spans(text)

        units = [
            ScoredUnit(
                text=word,
                start=start,
                end=end,
                index=i,
                surprisal=score,
                protected=_is_protected(word, start, end, protected_spans, sentence_spans),
                sentence_id=_sentence_id_for(start, sentence_spans),
            )
            for i, (word, start, end, score) in enumerate(words)
        ]
        self.last_units = units
        return units
