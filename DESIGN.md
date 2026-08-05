# Token Optimizer — System Design & Developer Handbook

A multi-stage prompt compression pipeline combining rule-based cleanup, perplexity-guided
importance scoring, and LLM semantic rewriting. Pure Python library — no UI, no server, no CLI.

**Team:** 3 developers · **Timeline:** 3–4 days · **Python:** 3.12

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [System architecture](#2-system-architecture)
3. [Key design decisions](#3-key-design-decisions)
4. [Folder and file structure](#4-folder-and-file-structure)
5. [The shared contract — `interfaces.py`](#5-the-shared-contract--interfacespy)
6. [Dev A — foundations, glue, integration](#6-dev-a--foundations-glue-integration)
7. [Dev B — perplexity pruning (the flagship)](#7-dev-b--perplexity-pruning-the-flagship)
8. [Dev C — semantic rewrite, evaluation, benchmark](#8-dev-c--semantic-rewrite-evaluation-benchmark)
9. [Dependency graph and parallel workflow](#9-dependency-graph-and-parallel-workflow)
10. [Day-by-day schedule](#10-day-by-day-schedule)
11. [Test plan](#11-test-plan)
12. [Benchmark methodology](#12-benchmark-methodology)
13. [Known limitations](#13-known-limitations)
14. [Future extensions](#14-future-extensions)
15. [Setup](#15-setup)
16. [References](#16-references)

---

## 1. What this project is

Given a prompt, the system compresses it to use fewer tokens while measuring exactly how much
semantic fidelity is traded away — and reports that tradeoff numerically.

This is the same problem space as Microsoft's **LLMLingua** line of research. The interview
framing: *"I implemented a multi-stage prompt compression pipeline with LLMLingua-style
perplexity pruning, and measured the compression/fidelity tradeoff empirically."*

**The core insight (information theory, not string manipulation):** a token's *surprisal* —
`−log P(tᵢ | t₁…tᵢ₋₁)` under a language model — measures how much information it carries.
Predictable tokens ("the", "please", "in order to") carry little; unpredictable, content-bearing
tokens carry a lot. Prune the low-surprisal tokens first.

**Headline result the README should carry:**

| Method           | Reduction | Similarity |
|------------------|-----------|------------|
| Rule only        | ~18%      | ~0.98      |
| Perplexity only  | ~42%      | ~0.94      |
| Rewrite only     | ~48%      | ~0.96      |
| **Hybrid**       | **~55%**  | **~0.97**  |

*(Illustrative shape. Report your own measured numbers — including the cases where compression
fails. An honest negative result is worth more in an interview than four rows of wins.)*

---

## 2. System architecture

### 2.1 The pipeline

```
                        Input Prompt
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Stage 1 — Rule Cleanup      │   Dev A
              │  regex, deterministic, free  │
              │  strips filler & whitespace  │
              └──────────────┬───────────────┘
                             │  str
                             ▼
              ┌──────────────────────────────┐
              │  Stage 2 — Perplexity Pruning│   Dev B  ★
              │  ┌────────────────────────┐  │
              │  │ surprisal scoring      │  │
              │  │ subword → word agg.    │  │
              │  │ sentence coarse pass   │  │
              │  │ protection layer       │  │
              │  │ BUDGET CONTROLLER      │  │
              │  │ ordered reconstruction │  │
              │  └────────────────────────┘  │
              └──────────────┬───────────────┘
                             │  str
                             ▼
              ┌──────────────────────────────┐
              │  Stage 3 — Semantic Rewrite  │   Dev C
              │  one LLM call, densest form  │
              └──────────────┬───────────────┘
                             │
                             ▼
                     Compressed Prompt
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Evaluator                   │   Dev C
              │  reduction / cost / drift    │
              └──────────────────────────────┘
```

### 2.2 Why this order

Each stage is more expensive and more intelligent than the last, so cheap deterministic
filtering happens first and the expensive LLM call only ever touches what survived.

| Stage | Cost | Determinism | Intelligence | Typical reduction |
|-------|------|-------------|--------------|-------------------|
| Rule cleanup | free | fully deterministic | none | 10–18% |
| Perplexity pruning | ~350 MB model, local, no API | deterministic given a seed | information-theoretic | 30–45% |
| Semantic rewrite | ~1.0 GB model, local, no API | deterministic (beam search, no sampling) | full language understanding | 25–40% |

The rewrite runs last, on the smallest input, making it the cheapest possible LLM call for
the job. This ordering is itself a defensible engineering decision — say so in the interview.

### 2.3 The `StageMetrics` waterfall

`pipeline.py` measures tokens before and after each stage, giving a per-stage breakdown for
free with no extra instrumentation:

```
rule_cleanup       : 47 → 39 tokens  (-17.0%)
perplexity_pruning : 39 → 24 tokens  (-38.5%)
semantic_rewrite   : 24 → 19 tokens  (-20.8%)
──────────────────────────────────────────────
TOTAL              : 47 → 19 tokens  (-59.6%)
```

---

## 3. Key design decisions

These are the "why" questions an interviewer will ask. Have answers ready.

### 3.1 Why isn't the budget controller its own stage?

The stage contract is `run(text: str) -> str`. If the budget controller were a separate stage,
the perplexity stage would have to **throw away its per-token surprisal scores** to hand off a
plain string — and the budget controller would then have nothing to work with.

So the budget controller lives *inside* `perplexity_pruning.py`. The external contract stays
clean (`str → str`); the scoring→removal handoff stays internal, where it has access to the
actual scores. This is why the architecture diagram shows four boxes but the folder has three
stage files.

### 3.2 Why is the rewrite stage worth its compute if we're saving tokens?

The rewrite runs a ~1 GB seq2seq model locally — a second or two of CPU per prompt, and no
money. But the deeper point stands regardless of price: compression is a **one-time, offline,
template-authoring cost**, amortized over every future call that reuses the compressed prompt.
It would *not* be sensible to run the full pipeline on every live request — you'd pay
compression latency on every call to save tokens on that same call.

State this explicitly in the README. It is the first question a sharp interviewer asks, and
"the compression step is free and local" is a good answer, but "compression is amortized over
reuse" is the *right* one.

### 3.3 Why prune at word level rather than BPE-token level?

GPT-2 uses byte-pair encoding, so `"vulnerabilities"` may tokenize as `vulner|abilities` and
`"bottlenecks"` as three pieces. Pruning individual BPE tokens produces literal gibberish.

We aggregate subword surprisals into whole-word units and prune whole words. Selective Context
does the same thing (grouping tokens into noun phrases via dependency parsing) for exactly this
reason. **This single decision is the difference between output that reads as English and
output that doesn't.**

### 3.4 Why a coarse sentence pass before token-level pruning?

Fine-grained pruning alone causes "death by a thousand cuts" — grammar gets shredded across
every sentence uniformly. Dropping whole low-information sentences first (e.g. a redundant
restatement of an instruction) preserves structure in the sentences that survive. This mirrors
LLMLingua's budget controller, which allocates different compression rates to different prompt
sections before touching individual tokens.

### 3.5 Why a protection list?

A pure score-based approach will happily delete **"not"** from *"do not include PII"* — because
"not" is highly predictable and therefore low-surprisal. That silently inverts the instruction
and is invisible until someone finds it in an eval.

Guarding negations, constraint modals, numbers, and template placeholders is the single best
talking point in the codebase: it demonstrates you thought about *silent correctness failure*,
not just about the happy path.

### 3.6 Why two different tokenizers?

The perplexity scorer uses distilGPT2's BPE tokenizer (that's the model's own vocabulary — no
choice). Cost accounting uses a separate counter for the *target* model's pricing. They
disagree. See [§13.2](#132-the-tokenizer-mismatch) for how we handle it honestly.

---

## 4. Folder and file structure

```
token_optimizer/
│
├── main.py                       # entry point — hardcoded prompt, runs pipeline, prints report
├── pipeline.py                   # CompressionPipeline orchestrator + per-stage metrics
├── interfaces.py                 # CompressionStage ABC + shared dataclasses  ← ALL 3, DAY 1
├── config.py                     # protected words, ratios, model IDs, pricing
├── benchmark_prompts.py          # hardcoded test prompts (+ optional eval pairs)
├── tokenizer_utils.py            # token counting + cost calculation
│
├── stages/
│   ├── __init__.py
│   ├── rule_cleanup.py           # Dev A
│   ├── perplexity_pruning.py     # Dev B  ★ includes the budget controller
│   └── semantic_rewriter.py      # Dev C
│
├── evaluator.py                  # Dev C — scores compressed vs original
├── benchmark.py                  # Dev C — individual + hybrid runs, comparison tables
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # shared fixtures + PassThroughStage stub
│   ├── test_interfaces.py
│   ├── test_rule_cleanup.py      # Dev A
│   ├── test_perplexity_pruning.py# Dev B
│   ├── test_semantic_rewriter.py # Dev C
│   ├── test_evaluator.py         # Dev C
│   └── test_pipeline.py          # Dev A — integration
│
├── results/                      # benchmark CSV + optional matplotlib PNG
├── requirements.txt
├── .env.example
└── README.md
```

### File ownership

| File | Owner | Difficulty |
|------|-------|-----------|
| `interfaces.py` | **All 3 — Day 1 kickoff** | Trivial |
| `config.py` | A | Low |
| `tokenizer_utils.py` | A | Low |
| `stages/rule_cleanup.py` | A | Medium |
| `benchmark_prompts.py` | A | Low |
| `pipeline.py` | A | Low |
| `main.py` | A | Low |
| `stages/perplexity_pruning.py` | **B** | **Very high** |
| `stages/semantic_rewriter.py` | C | Medium |
| `evaluator.py` | C | Medium |
| `benchmark.py` | C | Medium |
| `README.md` | C drafts; **B writes the algorithm section** | — |

Load is balanced by **effort, not file count**. Dev B's single file is worth Dev A's six.

---

## 5. The shared contract — `interfaces.py`

**Write this together, first, before splitting up.** It is the only file that requires all
three of you, and once it's locked nobody blocks anybody.

```python
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
    cost_saved: float
    semantic_similarity: float
    accuracy_retained: Optional[float] = None
```

### Contract rules — agree on all five in the kickoff

1. **`run()` takes and returns a plain string.** No exceptions, no side channels.
2. **Every stage must tolerate raw, uncleaned input.** The benchmark runs each stage standalone
   on the original prompt to produce the "X only" rows, so no stage may assume it is receiving
   another stage's output.
3. **`target_ratio` is a constructor argument, not a global read inside `run()`.** This lets
   the benchmark sweep ratios without mutating config.
4. **Stages must never raise on valid input.** On internal failure, return the input unchanged
   and log — a stage that throws kills the whole benchmark run.
5. **`name` is set per stage class** so the waterfall table is readable.

---

## 6. Dev A — foundations, glue, integration

**Scope:** `config.py`, `tokenizer_utils.py`, `stages/rule_cleanup.py`, `benchmark_prompts.py`,
`pipeline.py`, `main.py`.

**Blocked by:** nothing. Dev A is unblocked from minute one — which is why Dev A goes first and
hardest on Day 1.

**Ships to others:** `count_tokens()` by lunch Day 1 (earliest hard dependency in the project).

**Extra role:** Dev A is the **integration owner**. A's scope finishes first, so A absorbs Day 3
integration debugging and, if B is running late, takes over B's windowing logic or test writing.

### 6.1 `config.py`

```python
"""Central configuration. Dev A owns the file; Dev B dictates PROTECTED_WORDS."""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv

# Reads .env from the project root into os.environ. Must run before anything
# overrides like HF_HOME take effect. Silently does nothing if .env is
# missing — the project runs correctly with no .env at all (see DESIGN 15.5).
load_dotenv()

# ── Perplexity scorer (local, no API cost, ~350 MB) ──────────────────────────
SCORER_MODEL = "distilgpt2"
SCORER_MAX_CTX = 1024          # distilGPT2 hard context limit
SCORER_STRIDE = 512            # sliding-window stride for long inputs

# ── Compression ──────────────────────────────────────────────────────────────
TARGET_RATIO = 0.5             # keep 50% of tokens by default
COARSE_PASS_ENABLED = True     # drop whole low-information sentences first

# ── Semantic rewrite (Dev C) — local seq2seq, no API, no key ─────────────────
# flan-t5-base is instruction-tuned, ~1.0 GB, and runs on CPU. Step up to
# "google/flan-t5-large" (~3 GB) for better rewrites if RAM allows, or down
# to "google/flan-t5-small" (~300 MB) if disk is tight.
REWRITER_MODEL = "google/flan-t5-base"
REWRITER_MAX_NEW_TOKENS = 128
REWRITER_NUM_BEAMS = 4            # beam search > sampling for a deterministic benchmark

# ── Embeddings for semantic-drift scoring (local, free) ──────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Cost projection ──────────────────────────────────────────────────────────
# We never call a paid API. These are PUBLISHED list prices, used to project
# "if you were paying for this prompt, compression would save $X per call."
# We price against GPT-4-family models because tiktoken's cl100k_base IS their
# real tokenizer — so the token counts are exact, not estimated.
TOKENIZER_ENCODING = "cl100k_base"          # GPT-4 / GPT-4o / GPT-3.5-turbo
PRICING = {                                  # USD per 1M tokens (input, output)
    "gpt-4o":         (2.50, 10.00),
    "gpt-4-turbo":   (10.00, 30.00),
    "gpt-3.5-turbo":  (0.50,  1.50),
}
COST_MODEL = "gpt-4o"          # which model's list price to project savings against

# ── Protection layer — Dev B dictates the contents ───────────────────────────
PROTECTED_WORDS: set[str] = {
    # negation and scope — deleting these silently inverts instructions
    "not", "no", "never", "none", "neither", "nor", "without",
    "except", "unless", "cannot", "cant", "dont", "doesnt", "wont", "isnt",
    # constraint modals
    "must", "always", "only", "required", "require", "shall", "should",
    "least", "most", "exactly", "all", "every", "each", "any",
    # comparatives that flip meaning
    "before", "after", "instead", "rather", "than",
}

PROTECTED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\{[^{}]*\}"),          # {template_placeholders}
    re.compile(r"`[^`]*`"),             # `inline code`
    re.compile(r"`{3}.*?`{3}", re.S),   # fenced code blocks
    re.compile(r'"[^"]*"'),             # "quoted strings"
    re.compile(r"'[^']*'"),
    re.compile(r"\b\d[\d,.\-/:%]*\b"),  # numbers, dates, percentages, ranges
    re.compile(r"\$\d[\d,.]*"),         # currency
]

# Sentences at these positions are never dropped by the coarse pass —
# the first and last lines are usually the actual instruction.
COARSE_KEEP_FIRST = True
COARSE_KEEP_LAST = True

# ── Local model cache ────────────────────────────────────────────────────────
# Override in .env if your C: drive is short on space. Must be set before
# transformers is imported anywhere, which is why it lives here in config.
if os.environ.get("HF_HOME"):
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # silences a noisy warning
```

### 6.2 `tokenizer_utils.py`

```python
"""Token counting and cost math. Dev A."""
from __future__ import annotations

from functools import lru_cache

import tiktoken

import config


@lru_cache(maxsize=4)
def _encoding(name: str | None = None):
    return tiktoken.get_encoding(name or config.TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Exact GPT-4-family token count. Local, deterministic, offline, free.

    cl100k_base IS the tokenizer GPT-4 / GPT-4o / GPT-3.5-turbo use, so this
    is not an estimate — it is the real count for the models we project costs
    against. Safe to call inside the pruning loop.
    """
    if not text:
        return 0
    return len(_encoding().encode(text))


def count_tokens_for(text: str, encoding: str) -> int:
    """Count under a different tokenizer, for the cross-tokenizer comparison.

    Useful values: "cl100k_base" (GPT-4 family), "o200k_base" (GPT-4o family),
    "p50k_base" (older GPT-3), "gpt2" (what our perplexity scorer uses).
    The same compressed text saves a different number of tokens per family —
    that contrast is a genuinely interesting result to report.
    """
    if not text:
        return 0
    return len(_encoding(encoding).encode(text))


def estimate_cost(n_tokens: int, model: str | None = None, output: bool = False) -> float:
    """Projected USD cost of n_tokens under a model's PUBLISHED list price.

    We never call a paid API — this is a projection, not a bill. Label it as
    such in any output.
    """
    model = model or config.COST_MODEL
    if model not in config.PRICING:
        raise KeyError(f"No pricing configured for {model!r}")
    price_in, price_out = config.PRICING[model]
    rate = price_out if output else price_in
    return (n_tokens / 1_000_000) * rate


def cost_saved(before_tokens: int, after_tokens: int, model: str | None = None) -> float:
    """Projected saving per call if this prompt were sent to `model`."""
    return estimate_cost(before_tokens, model) - estimate_cost(after_tokens, model)
```

### 6.3 `stages/rule_cleanup.py`

```python
"""Stage 1 — deterministic rule-based cleanup. Dev A."""
from __future__ import annotations

import re

from interfaces import CompressionStage

# Ordered longest-first so multi-word phrases match before their substrings.
FILLER_PHRASES = [
    r"please carefully",
    r"please kindly",
    r"in order to",
    r"it is important (?:to note )?that",
    r"i would like you to",
    r"i want you to",
    r"could you please",
    r"make sure (?:that )?you",
    r"be sure to",
    r"as (?:an|a) (?:AI|assistant)[, ]",
    r"take (?:a moment|your time) (?:to|and)",
    r"very carefully",
    r"in detail",
    r"detailed and comprehensive",
    r"\bplease\b",
    r"\bkindly\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bsimply\b",
    r"\bjust\b",
]

_FILLER_RE = re.compile("|".join(f"(?:{p})" for p in FILLER_PHRASES), re.I)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_PROTECT = re.compile(r"`{3}.*?`{3}|`[^`]*`|\{[^{}]*\}", re.S)


class RuleCleanupStage(CompressionStage):
    name = "rule_cleanup"

    def run(self, text: str) -> str:
        if not text or not text.strip():
            return text

        # Carve out regions we must not touch, clean the rest, then restore.
        protected: list[str] = []

        def _stash(m: re.Match) -> str:
            protected.append(m.group(0))
            return f"\x00{len(protected) - 1}\x00"

        working = _PROTECT.sub(_stash, text)
        working = _FILLER_RE.sub(" ", working)
        working = _SPACE_BEFORE_PUNCT.sub(r"\1", working)
        working = _MULTI_SPACE.sub(" ", working)
        working = _MULTI_NEWLINE.sub("\n\n", working)
        working = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], working)

        # Re-capitalize if we stripped a leading filler word.
        working = working.strip()
        if working and text.strip()[:1].isupper():
            working = working[0].upper() + working[1:]
        return working
```

### 6.4 `pipeline.py`

```python
"""Stage orchestrator with per-stage metrics. Dev A."""
from __future__ import annotations

from typing import Callable, Sequence

from interfaces import CompressionStage, CompressionResult, StageMetrics
from tokenizer_utils import count_tokens


class CompressionPipeline:
    """Runs stages in order, measuring tokens at every hop."""

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
```

### 6.5 `benchmark_prompts.py` — a quiet responsibility

This file decides your headline numbers. If it contains only verbose, filler-heavy prompts,
rule cleanup alone scores suspiciously well and perplexity pruning looks redundant.

**Required mix (15–20 prompts):**

| Category | Count | Why |
|----------|-------|-----|
| Verbose instruction prompts | 4 | Where rule cleanup shines |
| Terse, already-optimized prompts | 3 | **Compression should mostly fail here** — an honest negative |
| Few-shot heavy (examples inline) | 3 | High redundancy, where perplexity pruning shines |
| Prompts with `{placeholders}` | 3 | Exercises the protection layer |
| Prompts with negations/constraints | 3 | The "do not include PII" correctness case |
| Prompts with code blocks | 2 | Exercises protected regions |
| One prompt > 1024 GPT-2 tokens | 1 | Exercises Dev B's sliding window |

```python
"""Hardcoded benchmark prompt set. Dev A."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkPrompt:
    id: str
    category: str
    text: str


BENCHMARK_PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="verbose_code_review",
        category="verbose",
        text=(
            "Please carefully review the following code and provide a detailed "
            "analysis of any potential bugs, performance bottlenecks, security "
            "vulnerabilities, or areas where improvements can be made."
        ),
    ),
    BenchmarkPrompt(
        id="terse_classify",
        category="already_terse",
        text="Classify the sentiment as positive, negative, or neutral.",
    ),
    BenchmarkPrompt(
        id="negation_pii",
        category="negation",
        text=(
            "Summarize the ticket below. Do not include PII. Never reveal the "
            "customer's email address or phone number under any circumstances."
        ),
    ),
    BenchmarkPrompt(
        id="template_vars",
        category="placeholder",
        text=(
            "You are helping {user_name} with a {issue_type} issue. "
            "Respond in {tone} tone and keep it under {max_words} words."
        ),
    ),
    # ... 12–16 more, covering every row in the table above
]
```

### 6.6 `main.py`

```python
"""Entry point. Run this from PyCharm's green arrow. Dev A."""
from __future__ import annotations

import config
from evaluator import Evaluator
from pipeline import CompressionPipeline
from stages.perplexity_pruning import PerplexityPruningStage
from stages.rule_cleanup import RuleCleanupStage
from stages.semantic_rewriter import SemanticRewriterStage

DEMO_PROMPT = (
    "Please carefully review the following code and provide a detailed analysis "
    "of any potential bugs, performance bottlenecks, security vulnerabilities, "
    "or areas where improvements can be made. Do not include any PII in your "
    "response."
)


def main() -> None:
    pipeline = CompressionPipeline(
        stages=[
            RuleCleanupStage(),
            PerplexityPruningStage(target_ratio=config.TARGET_RATIO),
            SemanticRewriterStage(),
        ]
    )

    result = pipeline.compress(DEMO_PROMPT)
    evaluation = Evaluator().evaluate(result)

    print("\n=== ORIGINAL ===")
    print(result.original_text)
    print("\n=== COMPRESSED ===")
    print(result.compressed_text)

    print("\n=== PER-STAGE WATERFALL ===")
    print(f"{'Stage':<22}{'Before':>8}{'After':>8}{'Reduction':>12}")
    for m in result.stage_metrics:
        print(f"{m.stage_name:<22}{m.tokens_before:>8}{m.tokens_after:>8}"
              f"{m.reduction_pct:>11.1%}")
    print("-" * 50)
    print(f"{'TOTAL':<22}{result.original_tokens:>8}"
          f"{result.compressed_tokens:>8}{result.reduction_pct:>11.1%}")

    print("\n=== EVALUATION ===")
    print(f"Semantic similarity : {evaluation.semantic_similarity:.3f}")
    print(f"Cost saved          : ${evaluation.cost_saved:.6f} per call")


if __name__ == "__main__":
    main()
```

---

## 7. Dev B — perplexity pruning (the flagship)

**Scope:** `stages/perplexity_pruning.py` — one file, seven internal layers.

**Blocked by:** `interfaces.CompressionStage` (Day 1 kickoff) and `count_tokens()` from Dev A.
**Stub both locally on Day 1** so an A-side delay never touches you; swap to the real ones Day 3.

**Ships to others:** pruned text to Dev C's rewriter; the stage class to Dev C's benchmark for
the "Perplexity Only" row; `PROTECTED_WORDS` contents to Dev A on Day 2; the algorithm section
of the README.

**You are the critical path.** Everything else in this project is bounded, known work. Yours is
the only file where a hard problem can eat a day. Don't volunteer for integration or README
formatting until Day 4.

### 7.1 The seven layers

| # | Layer | What it does | Failure mode if skipped |
|---|-------|--------------|-------------------------|
| 1 | Model loading | Lazy singleton, `eval()` mode, CPU-pinned | Benchmark takes minutes instead of seconds |
| 2 | Surprisal scoring | `−log P(tᵢ \| t₍<ᵢ₎)`, shift-by-one alignment | Off-by-one = scores attached to wrong tokens |
| 3 | Sliding window | Strided re-scoring for inputs > 1024 tokens | Crash or silent truncation on long prompts |
| 4 | Subword→word aggregation | Prune whole words, not BPE pieces | **Gibberish output** |
| 5 | Coarse sentence pass | Drop whole low-info sentences first | "Death by a thousand cuts" — shredded grammar |
| 6 | Protection layer | Veto removal regardless of score | Silent instruction inversion ("not" deleted) |
| 7 | Budget controller + reconstruction | Greedy drop to target, rebuild in original order | Wrong tokens deleted; word order scrambled |

### 7.2 `stages/perplexity_pruning.py`

```python
"""Stage 2 — perplexity-based pruning with an internal budget controller.

Dev B. This is the algorithmic centerpiece.

Algorithm
---------
1.  Score every BPE token's surprisal  -log P(t_i | t_<i)  under distilGPT2,
    using a strided sliding window so inputs longer than the model's 1024-token
    context are still scored with real left-context.
2.  Aggregate subword surprisals up to whole-word units so we never delete half
    a BPE-split word.
3.  Coarse pass: rank sentences by length-normalized mean surprisal and drop
    whole low-information sentences while that keeps us above target.
4.  Fine pass (budget controller): sort surviving words by ascending surprisal
    and greedily drop the least informative, skipping anything the protection
    layer vetoes, until the token budget is met.
5.  Reconstruct from the ORIGINAL character offsets in ORIGINAL order.

Inspired by LLMLingua (Jiang et al., EMNLP 2023) and Selective Context
(Li et al., 2023). See DESIGN.md section 16 for references.
"""
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

    A new word starts when there is a gap (whitespace) between the previous
    token's end offset and this token's start offset. Score = mean of the
    word's subword surprisals; +inf propagates so an unscored first token keeps
    its whole word.

    Without this step, "vulnerabilities" -> "vulner|abilities" can be pruned
    apart and the output becomes gibberish.
    """
    words: list[tuple[str, int, int, float]] = []
    cur_start = cur_end = None
    cur_scores: list[float] = []

    for (start, end), score in zip(offsets, token_scores):
        if start == end:                       # special/empty token
            continue
        starts_new_word = cur_end is None or start > cur_end
        if starts_new_word:
            if cur_start is not None:
                words.append((text[cur_start:cur_end], cur_start, cur_end,
                              _mean_inf(cur_scores)))
            cur_start, cur_end, cur_scores = start, end, [score]
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
    word: str, start: int, end: int, protected_spans: list[tuple[int, int]]
) -> bool:
    """Veto removal of negations, constraints, numbers, placeholders, proper nouns."""
    normalized = re.sub(r"[^\w']", "", word).lower()
    if normalized in config.PROTECTED_WORDS:
        return True
    if any(s < end and start < e for s, e in protected_spans):
        return True
    # Mid-sentence capitalization is a cheap proper-noun heuristic.
    if word[:1].isupper() and start > 0 and _WORD_CHARS.search(word):
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
                protected=_is_protected(word, start, end, protected_spans),
                sentence_id=_sentence_id_for(start, sentence_spans),
            )
            for i, (word, start, end, score) in enumerate(words)
        ]
        self.last_units = units
        return units
```

### 7.3 Three research-backed upgrades (Day 4, if time allows)

These are what separate a homework exercise from a flagship. Priority order:

**(a) Iterative conditional scoring — ITPC.** The implementation above scores every token once
against the *original* context. But once you delete tokens, the surviving context has changed,
so every remaining score is stale. LLMLingua's **Iterative Token-level Prompt Compression**
segments the prompt (~100–200 tokens), compresses segment 1, then scores segment 2 *conditioned
on the already-compressed prefix*, and so on. It models token interdependence instead of
assuming independence. This is the strongest single thing you can say in an interview.

**(b) Contrastive perplexity — the fix for plain surprisal's biggest flaw.** Plain surprisal
keeps *surprising* tokens, which is not the same as *important* tokens — it will lovingly
preserve typos, rare proper nouns, and random noise because they are unpredictable.
LongLLMLingua's answer: if there is a question/task, score the **shift** in a token's surprisal
when the question is prepended versus not. Tokens whose surprisal drops sharply under the
question are question-relevant. Even if you don't implement it, *stating this limitation about
your own method* is what a senior interviewer is listening for.

**(c) Dynamic per-sentence budget allocation.** Rather than one global threshold, allocate
budget non-uniformly — sentences that ranked high in the coarse pass keep more of their tokens.
LongLLMLingua does this per-document.

**Explicitly out of scope: distribution alignment.** distilGPT2's notion of "predictable" isn't
the target model's. LLMLingua closes that gap by instruction-tuning the small scorer on the
target LLM's outputs. Naming this as a known limitation is a strong interview close.

### 7.4 Traps that will bite you

| Trap | Symptom | Fix |
|------|---------|-----|
| **Sorted-index bug** | Wrong words deleted | Carry the original index in `ScoredUnit.index`; never use the sorted-list position |
| Off-by-one alignment | Scores on wrong tokens | `logits[:, :-1]` predicts `ids[:, 1:]`; verify by hand on a 5-word sentence |
| BPE `Ġ` mangling | Broken spacing | Use `return_offsets_mapping=True` and slice the original string — never `"".join(tokens)` |
| Input > 1024 tokens | Crash or silent truncation | Sliding window (`_token_surprisals`) |
| Unreachable target | Loop forever, or protected words deleted | Stop and report `last_achieved_ratio` |
| Non-idempotence | Running twice at 0.5 ≠ once at 0.25 | Expected. Document it; don't chase it. |
| Model reload per call | Benchmark takes minutes | `@lru_cache(maxsize=1)` on `_load_scorer()` |
| Non-linear degradation | Output falls apart past ~65% removal | Expected — chart it, it's a good result |

### 7.5 Your four Day-1 kickoff questions

1. `target_ratio` as a constructor arg, not a global read inside `run()` — confirm.
2. Dev A hosts `PROTECTED_WORDS`, Dev B dictates its contents — confirm.
3. Is the budget counted in local tokens or provider-billable tokens? (See §13.2.)
4. How do you expose per-token scores for the demo without breaking `str → str`?
   → `self.last_units` on the instance, plus the public `score()` method.

---

## 8. Dev C — semantic rewrite, evaluation, benchmark

**Scope:** `stages/semantic_rewriter.py`, `evaluator.py`, `benchmark.py`.

**Blocked by:** `count_tokens()` from Dev A. Nothing else — **stub A's and B's stages as
pass-throughs and build the entire evaluator and benchmark against fake data on Day 1.**
By end of Day 1 both tables should print with garbage numbers.

**Ships to others:** the final numbers everyone puts on their resume.

> ⚠️ **Dev C's unique risk — start this at the beginning of Day 1.** C owns the two largest
> model downloads (`flan-t5-base` ~1.0 GB, `all-MiniLM-L6-v2` ~90 MB) *plus* a transitive torch
> install. Kick the downloads off first and build the evaluator against stubs while they run —
> otherwise you lose an afternoon to a progress bar. No API key, no account, no billing: every
> model is open-source and local.

### 8.1 `stages/semantic_rewriter.py`

```python
"""Stage 3 — semantic rewrite with a LOCAL seq2seq model. Dev C.

No API, no key, no network after the first model download. flan-t5-base is
instruction-tuned, ~1.0 GB, and runs on CPU in a second or two per prompt.

Runs last, on the smallest input in the pipeline — so it is both the cheapest
call and the one least likely to mangle something, because rule cleanup and
perplexity pruning have already removed the obvious redundancy.

Results are cached so repeated benchmark runs don't re-do identical work.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from interfaces import CompressionStage

INSTRUCTION = (
    "Rewrite the following instruction to be as short as possible. "
    "Keep every constraint, negation, number, and placeholder exactly as it "
    "appears. Output only the rewritten instruction.\n\n"
)

# A rewrite that drops any of these is worse than no rewrite at all.
_CRITICAL = re.compile(r"\{[^{}]*\}|\b\d[\d,.\-/:%]*\b|\bnot\b|\bnever\b|\bno\b", re.I)


@lru_cache(maxsize=1)
def _load_rewriter(model_name: str):
    """Load the seq2seq rewriter once per process."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


class SemanticRewriterStage(CompressionStage):
    name = "semantic_rewrite"

    def __init__(self, model: str | None = None, use_cache: bool = True) -> None:
        self.model_name = model or config.REWRITER_MODEL
        self.use_cache = use_cache
        self._cache: dict[str, str] = {}
        self.rejected = 0       # rewrites discarded by the safety checks below

    def run(self, text: str) -> str:
        if not text or not text.strip():
            return text

        key = hashlib.sha256(f"{self.model_name}\x00{text}".encode()).hexdigest()
        if self.use_cache and key in self._cache:
            return self._cache[key]

        try:
            rewritten = self._generate(text)
        except Exception as exc:                     # never kill the benchmark
            print(f"[rewriter] generation failed: {exc!r} — passing through")
            rewritten = text

        if not self._is_safe(text, rewritten):
            self.rejected += 1
            rewritten = text

        if self.use_cache:
            self._cache[key] = rewritten
        return rewritten

    # -- internals -----------------------------------------------------------
    def _generate(self, text: str) -> str:
        tokenizer, model = _load_rewriter(self.model_name)
        inputs = tokenizer(
            INSTRUCTION + text,
            return_tensors="pt",
            truncation=True,
            max_length=512,          # flan-t5 encoder limit
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.REWRITER_MAX_NEW_TOKENS,
                num_beams=config.REWRITER_NUM_BEAMS,
                do_sample=False,     # deterministic — the benchmark must reproduce
                early_stopping=True,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _is_safe(self, original: str, rewritten: str) -> bool:
        """Reject a rewrite that is empty, longer, or drops critical tokens.

        A small local model will sometimes hallucinate or truncate. Falling
        back to the input is always better than shipping a broken prompt —
        and `self.rejected` makes the failure rate a reportable number.
        """
        if not rewritten:
            return False
        if len(rewritten) >= len(original):
            return False
        required = {m.group(0).lower() for m in _CRITICAL.finditer(original)}
        present = {m.group(0).lower() for m in _CRITICAL.finditer(rewritten)}
        return required.issubset(present)
```

### 8.2 `evaluator.py`

```python
"""Scoring: reduction, cost, semantic drift. Dev C."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import config
from interfaces import CompressionResult, EvalResult
from tokenizer_utils import cost_saved


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def cosine_similarity(a: str, b: str) -> float:
    """Embedding cosine similarity — our 'did we break the meaning?' signal.

    Local, free, and offline — no external service needed to detect drift.
    """
    if not a.strip() or not b.strip():
        return 0.0
    vectors = _embedder().encode([a, b], normalize_embeddings=True)
    return float(np.dot(vectors[0], vectors[1]))


class Evaluator:
    def evaluate(self, result: CompressionResult) -> EvalResult:
        return EvalResult(
            strategy_name=result.strategy_name,
            token_reduction_pct=result.reduction_pct,
            cost_saved=cost_saved(result.original_tokens, result.compressed_tokens),
            semantic_similarity=cosine_similarity(
                result.original_text, result.compressed_text
            ),
        )
```

### 8.3 `benchmark.py`

```python
"""Runs every stage standalone plus the hybrid pipeline; prints both tables. Dev C."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import config
from benchmark_prompts import BENCHMARK_PROMPTS
from evaluator import Evaluator
from interfaces import CompressionStage
from pipeline import CompressionPipeline
from stages.perplexity_pruning import PerplexityPruningStage
from stages.rule_cleanup import RuleCleanupStage
from stages.semantic_rewriter import SemanticRewriterStage

RESULTS_DIR = Path(__file__).parent / "results"


def build_strategies() -> dict[str, list[CompressionStage]]:
    """Each stage standalone (on RAW input) plus the full hybrid pipeline."""
    return {
        "rule_only":       [RuleCleanupStage()],
        "perplexity_only": [PerplexityPruningStage()],
        "rewrite_only":    [SemanticRewriterStage()],
        "hybrid":          [RuleCleanupStage(),
                            PerplexityPruningStage(),
                            SemanticRewriterStage()],
    }


def run() -> None:
    evaluator = Evaluator()
    rows: list[dict] = []

    for name, stages in build_strategies().items():
        pipeline = CompressionPipeline(stages, name=name)
        reductions, similarities, savings = [], [], []

        for prompt in BENCHMARK_PROMPTS:
            result = pipeline.compress(prompt.text)
            evaluation = evaluator.evaluate(result)
            reductions.append(evaluation.token_reduction_pct)
            similarities.append(evaluation.semantic_similarity)
            savings.append(evaluation.cost_saved)
            rows.append({
                "strategy": name,
                "prompt_id": prompt.id,
                "category": prompt.category,
                "tokens_before": result.original_tokens,
                "tokens_after": result.compressed_tokens,
                "reduction_pct": round(evaluation.token_reduction_pct, 4),
                "similarity": round(evaluation.semantic_similarity, 4),
                "cost_saved": round(evaluation.cost_saved, 8),
            })

        print(f"{name:<18}{statistics.mean(reductions):>10.1%}"
              f"{statistics.mean(similarities):>13.3f}"
              f"{sum(savings):>14.6f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "benchmark.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    print(f"{'Strategy':<18}{'Reduction':>10}{'Similarity':>13}{'Cost saved':>14}")
    print("-" * 55)
    run()
```

---

## 9. Dependency graph and parallel workflow

```
              interfaces.py   (all 3, Day 1 — the ONLY true blocker)
                     │
         ┌───────────┼───────────┐
         │           │           │
       Dev A       Dev B       Dev C
         │           │           │
   count_tokens() ──►│──────────►│     earliest real dependency
         │           │           │     A ships it by lunch Day 1
         │      PROTECTED_WORDS  │
         │◄──────────┘           │     B dictates contents, A hosts
         │           │           │
         └──────► pipeline.py ◄──┘     Day 3 — first time all three meet
                       │
                  benchmark.py         Dev C — needs all stages real
```

### The stub discipline — this is what makes it parallel

| Dev | Stubs | With |
|-----|-------|------|
| A | nothing | — |
| B | `config`, `count_tokens` | local constants, `len(text.split())` |
| C | `RuleCleanupStage`, `PerplexityPruningStage` | `PassThroughStage` (`return text`) |

```python
# tests/conftest.py — the stub everyone develops against
from interfaces import CompressionStage


class PassThroughStage(CompressionStage):
    """A no-op stage. Lets each dev build against the contract, not the code."""
    name = "passthrough"

    def run(self, text: str) -> str:
        return text
```

### Cross-cutting requirement — agree on Day 1

The benchmark's "Rule Only / Perplexity Only / Rewrite Only" rows mean **each stage must run
standalone on the raw, uncleaned prompt**. No stage may assume it is receiving another stage's
output. Easy to satisfy if you know it up front; annoying to retrofit on Day 3.

---

## 10. Day-by-day schedule

| | Dev A | Dev B | Dev C |
|---|-------|-------|-------|
| **Day 1** | `interfaces.py` kickoff → `config.py` + `tokenizer_utils.py` **by lunch**, then rule cleanup + prompt set | Kickoff → **start the torch/distilGPT2 download first**, then raw per-token surprisal printing; verify shift-by-one by hand | **Start the model downloads first.** Then evaluator + benchmark running end-to-end on stub stages while they finish |
| **Day 2** | Finish rule cleanup, unit tests, draft `pipeline.py` | Word aggregation, coarse sentence pass, protection layer → send list to A | Semantic rewriter + caching; swap in A's real rule cleanup |
| **Day 3** | **Integration owner** — wire `main.py`, debug the joins, help B if behind | Budget controller, reconstruction, sliding window → integrate | Swap in B's real stage, get both tables printing real numbers |
| **Day 4** | Polish output formatting, final run | Threshold tuning, then ITPC if time; README algorithm section | Final benchmark run, README numbers + tables |

### Two checkpoints worth scheduling

**End of Day 1 — the stub integration.** Run the full pipeline with all three stages as
pass-throughs. It should print a table of zeros without crashing. This catches interface
mismatches on Day 1 instead of Day 3, and it takes fifteen minutes.

**Midday Day 3 — the real integration.** All three real stages in `pipeline.py`. Whatever
breaks here is where your remaining time goes.

### If someone falls behind

Prioritize: **B's perplexity stage > C's evaluator > A's rule cleanup > C's semantic rewriter.**
The rewriter is the most droppable — least novel, costs money, and the pipeline still tells a
complete story as rule cleanup → perplexity pruning. Losing B's stage leaves you with a regex
script.

---

## 11. Test plan

Testable, isolated modules read as *engineered*, not *hacked together*. Run with `pytest -q`.

### 11.1 `tests/conftest.py`

```python
"""Shared fixtures. Note: no test may make a network call."""
from __future__ import annotations

import pytest

from interfaces import CompressionStage


class PassThroughStage(CompressionStage):
    name = "passthrough"

    def run(self, text: str) -> str:
        return text


class HalvingStage(CompressionStage):
    """Deterministic fake compressor for pipeline tests."""
    name = "halving"

    def run(self, text: str) -> str:
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])


@pytest.fixture
def passthrough() -> CompressionStage:
    return PassThroughStage()


@pytest.fixture
def word_counter():
    return lambda text: len(text.split())


@pytest.fixture(scope="session")
def pruner():
    """Real distilGPT2 stage — session-scoped so the model loads once."""
    from stages.perplexity_pruning import PerplexityPruningStage
    return PerplexityPruningStage(target_ratio=0.5)
```

### 11.2 `tests/test_perplexity_pruning.py` — Dev B (the critical suite)

```python
"""Dev B's tests. Every case here maps to a real failure mode."""
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
```

### 11.3 `tests/test_rule_cleanup.py` — Dev A

```python
from stages.rule_cleanup import RuleCleanupStage


def test_removes_filler_phrases():
    stage = RuleCleanupStage()
    out = stage.run("Please carefully review the code in order to find bugs.")
    assert "please carefully" not in out.lower()
    assert "in order to" not in out.lower()
    assert "review" in out.lower() and "bugs" in out.lower()


def test_collapses_whitespace():
    assert RuleCleanupStage().run("Review    the\n\n\n\ncode.") == "Review the\n\ncode."


def test_idempotent():
    stage = RuleCleanupStage()
    once = stage.run("Please kindly review    the code in order to find bugs.")
    assert stage.run(once) == once


def test_code_blocks_untouched():
    fence = "`" * 3
    text = f"Review this:\n{fence}python\nplease just do the thing\n{fence}"
    out = RuleCleanupStage().run(text)
    assert "please just do the thing" in out


def test_placeholders_untouched():
    out = RuleCleanupStage().run("Please help {user_name} with {issue}.")
    assert "{user_name}" in out and "{issue}" in out


def test_never_returns_empty_for_nonempty_input():
    assert RuleCleanupStage().run("Please just simply basically actually.").strip()


def test_reduces_or_preserves_length():
    text = "Please carefully review the code in order to find any potential bugs."
    assert len(RuleCleanupStage().run(text)) <= len(text)
```

### 11.4 `tests/test_semantic_rewriter.py` — Dev C

```python
"""Generation is stubbed so the suite never downloads flan-t5 or uses a GPU."""
from __future__ import annotations

import pytest

from stages.semantic_rewriter import SemanticRewriterStage


def _stage_returning(reply: str, **kwargs) -> SemanticRewriterStage:
    """A stage whose _generate() is replaced by a canned reply."""
    stage = SemanticRewriterStage(**kwargs)
    stage.calls = 0

    def fake_generate(text: str) -> str:
        stage.calls += 1
        return reply

    stage._generate = fake_generate      # type: ignore[method-assign]
    return stage


def test_returns_rewritten_text():
    stage = _stage_returning("Review code for bugs.")
    out = stage.run("Please carefully review the following code to find bugs.")
    assert out == "Review code for bugs."


def test_cache_prevents_duplicate_generation():
    stage = _stage_returning("Review code for bugs.")
    text = "Please carefully review the following code and report any bugs found."
    stage.run(text)
    stage.run(text)
    assert stage.calls == 1


def test_cache_can_be_disabled():
    stage = _stage_returning("Review code for bugs.", use_cache=False)
    text = "Please carefully review the following code and report any bugs found."
    stage.run(text)
    stage.run(text)
    assert stage.calls == 2


def test_longer_rewrite_is_rejected():
    """A 'compression' that grew the prompt is a failed compression."""
    stage = _stage_returning("A" * 5000)
    text = "Short prompt."
    assert stage.run(text) == text
    assert stage.rejected == 1


def test_empty_generation_falls_back_to_input():
    text = "Analyze this code for potential bugs."
    stage = _stage_returning("")
    assert stage.run(text) == text
    assert stage.rejected == 1


def test_dropped_negation_is_rejected():
    """The safety net: a local model that loses 'not' must not be trusted."""
    original = "Summarize the ticket. Do not include any PII in the output."
    stage = _stage_returning("Summarize the ticket. Include PII.")
    assert stage.run(original) == original
    assert stage.rejected == 1


def test_dropped_placeholder_is_rejected():
    original = "Greet {user_name} warmly and answer their question fully."
    stage = _stage_returning("Greet the user warmly.")
    assert stage.run(original) == original
    assert stage.rejected == 1


def test_dropped_number_is_rejected():
    original = "Write a summary of the article in exactly 250 words please."
    stage = _stage_returning("Write a short summary.")
    assert stage.run(original) == original
    assert stage.rejected == 1


def test_safe_rewrite_is_accepted():
    original = "Please summarize the ticket. Do not include {email} or PII."
    stage = _stage_returning("Summarize ticket. Do not include {email} or PII.")
    assert stage.run(original) != original
    assert stage.rejected == 0


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_input_short_circuits(text):
    stage = _stage_returning("something")
    assert stage.run(text) == text
    assert stage.calls == 0


def test_generation_failure_passes_through():
    stage = SemanticRewriterStage()

    def boom(text: str) -> str:
        raise RuntimeError("model exploded")

    stage._generate = boom               # type: ignore[method-assign]
    text = "Analyze this code for bugs."
    assert stage.run(text) == text


@pytest.mark.slow
def test_real_model_produces_something(monkeypatch):
    """Opt-in check against the real flan-t5: pytest -m slow"""
    stage = SemanticRewriterStage()
    out = stage.run(
        "Please carefully review the following code and provide a detailed "
        "analysis of any bugs or performance problems you happen to find."
    )
    assert out.strip()
```

### 11.5 `tests/test_pipeline.py` — Dev A, integration

```python
from conftest import HalvingStage, PassThroughStage

from pipeline import CompressionPipeline


def test_stages_run_in_order(word_counter):
    order = []

    class Recording(PassThroughStage):
        def __init__(self, label):
            self.name = label

        def run(self, text):
            order.append(self.name)
            return text

    pipeline = CompressionPipeline(
        [Recording("first"), Recording("second"), Recording("third")],
        token_counter=word_counter,
    )
    pipeline.compress("alpha beta gamma")
    assert order == ["first", "second", "third"]


def test_metrics_captured_per_stage(word_counter):
    pipeline = CompressionPipeline(
        [HalvingStage(), HalvingStage()], token_counter=word_counter
    )
    result = pipeline.compress("one two three four five six seven eight")
    assert len(result.stage_metrics) == 2
    assert result.stage_metrics[0].tokens_before == 8
    assert result.stage_metrics[0].tokens_after == 4
    assert result.stage_metrics[1].tokens_after == 2


def test_all_passthrough_gives_zero_reduction(word_counter, passthrough):
    """The Day-1 stub integration check."""
    pipeline = CompressionPipeline([passthrough] * 3, token_counter=word_counter)
    result = pipeline.compress("alpha beta gamma delta")
    assert result.reduction_pct == 0.0
    assert result.compressed_text == "alpha beta gamma delta"


def test_broken_stage_does_not_kill_the_run(word_counter):
    class Exploding(PassThroughStage):
        name = "exploding"

        def run(self, text):
            raise RuntimeError("boom")

    pipeline = CompressionPipeline(
        [HalvingStage(), Exploding(), HalvingStage()], token_counter=word_counter
    )
    result = pipeline.compress("one two three four five six seven eight")
    assert result.compressed_tokens == 2      # both halvings still ran
```

### 11.6 `tests/test_interfaces.py` — all three (guards the shared contract)

```python
"""If these break, someone changed the contract without telling the others."""
from __future__ import annotations

import pytest

from interfaces import CompressionResult, CompressionStage, EvalResult, StageMetrics


def test_stage_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        CompressionStage()          # type: ignore[abstract]


def test_subclass_must_implement_run():
    class Incomplete(CompressionStage):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()                # type: ignore[abstract]


def test_valid_subclass_works():
    class Echo(CompressionStage):
        name = "echo"

        def run(self, text: str) -> str:
            return text

    assert Echo().run("hello") == "hello"


@pytest.mark.parametrize(
    "before,after,expected",
    [(100, 50, 0.5), (100, 100, 0.0), (100, 0, 1.0), (0, 0, 0.0)],
)
def test_stage_metrics_reduction(before, after, expected):
    metrics = StageMetrics("s", before, after)
    assert metrics.reduction_pct == pytest.approx(expected)


def test_compression_result_reduction_and_zero_guard():
    result = CompressionResult("hybrid", "abc", "a", 100, 40)
    assert result.reduction_pct == pytest.approx(0.6)
    assert CompressionResult("x", "", "", 0, 0).reduction_pct == 0.0


def test_stage_metrics_default_empty_on_result():
    assert CompressionResult("x", "a", "b", 1, 1).stage_metrics == []


def test_eval_result_accuracy_is_optional():
    assert EvalResult("x", 0.5, 0.001, 0.97).accuracy_retained is None
```

### 11.7 `tests/test_evaluator.py` — Dev C

```python
"""Embedder is stubbed by default so the suite stays offline and fast."""
from __future__ import annotations

import numpy as np
import pytest

import evaluator as evaluator_module
from evaluator import Evaluator, cosine_similarity
from interfaces import CompressionResult


class _FakeEmbedder:
    """Deterministic bag-of-words embedder — no model download."""

    def encode(self, texts, normalize_embeddings=True):
        vocab = sorted({w for t in texts for w in t.lower().split()})
        vectors = []
        for text in texts:
            words = text.lower().split()
            vector = np.array([float(words.count(w)) for w in vocab])
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.array(vectors)


@pytest.fixture(autouse=True)
def stub_embedder(monkeypatch):
    evaluator_module._embedder.cache_clear()
    monkeypatch.setattr(evaluator_module, "_embedder", lambda: _FakeEmbedder())
    yield
    evaluator_module._embedder.cache_clear()


def test_identical_text_is_maximally_similar():
    text = "review the code for bugs"
    assert cosine_similarity(text, text) == pytest.approx(1.0, abs=1e-6)


def test_overlapping_text_scores_higher_than_disjoint():
    original = "review the code for security bugs"
    close = "review code for bugs"
    far = "bake a chocolate cake tonight"
    assert cosine_similarity(original, close) > cosine_similarity(original, far)


@pytest.mark.parametrize("a,b", [("", "x"), ("x", ""), ("   ", "x")])
def test_blank_input_returns_zero(a, b):
    assert cosine_similarity(a, b) == 0.0


def test_evaluate_populates_every_field():
    result = CompressionResult(
        strategy_name="hybrid",
        original_text="review the code for bugs",
        compressed_text="review code bugs",
        original_tokens=100,
        compressed_tokens=40,
    )
    evaluation = Evaluator().evaluate(result)
    assert evaluation.strategy_name == "hybrid"
    assert evaluation.token_reduction_pct == pytest.approx(0.6)
    assert evaluation.cost_saved > 0
    assert 0.0 <= evaluation.semantic_similarity <= 1.0


def test_no_compression_saves_nothing():
    result = CompressionResult("noop", "same text", "same text", 50, 50)
    evaluation = Evaluator().evaluate(result)
    assert evaluation.token_reduction_pct == 0.0
    assert evaluation.cost_saved == pytest.approx(0.0)


@pytest.mark.slow
def test_real_embedder_agrees_on_direction(monkeypatch):
    """Opt-in sanity check against the real model: pytest -m slow"""
    monkeypatch.undo()
    evaluator_module._embedder.cache_clear()
    original = "Summarize the support ticket without including any PII."
    close = "Summarize the ticket, no PII."
    far = "Write a poem about the sea."
    assert cosine_similarity(original, close) > cosine_similarity(original, far)
```

Register the marker in `pyproject.toml` so `pytest -m "not slow"` is the default local run:

```toml
[tool.pytest.ini_options]
markers = ["slow: requires downloading the real embedding model"]
```

### 11.8 Test coverage matrix

| Concern | Test | Owner |
|---------|------|-------|
| Negation survives | `test_negation_survives_aggressive_pruning` | B |
| Constraint modals survive | `test_constraint_words_survive` | B |
| Placeholders intact | `test_template_placeholders_intact` | B |
| Numbers preserved | `test_numbers_and_units_preserved` | B |
| Budget respected | `test_output_within_budget` | B |
| Ratio monotonicity | `test_lower_ratio_removes_more` | B |
| **Sorted-index bug** | `test_budget_prune_uses_original_indices` | B |
| Protected veto | `test_protected_units_never_dropped_even_when_lowest` | B |
| Document order | `test_original_order_preserved` | B |
| No orphaned subwords | `test_no_orphaned_subwords` | B |
| Determinism | `test_deterministic` | B |
| Long input (>1024) | `test_long_input_exceeds_context_window` | B |
| Degenerate inputs | `test_degenerate_inputs_do_not_crash` | B |
| Pass-through at ratio 1.0 | `test_ratio_one_is_passthrough` | B |
| Subword aggregation | `test_aggregate_merges_subwords` | B |
| Filler removal | `test_removes_filler_phrases` | A |
| Cleanup idempotence | `test_idempotent` | A |
| Code blocks protected | `test_code_blocks_untouched` | A |
| Stage ordering | `test_stages_run_in_order` | A |
| Metrics waterfall | `test_metrics_captured_per_stage` | A |
| Stub integration | `test_all_passthrough_gives_zero_reduction` | A |
| Stage failure isolation | `test_broken_stage_does_not_kill_the_run` | A |
| Rewrite caching | `test_cache_prevents_duplicate_generation` | C |
| Rewrite grew → reject | `test_longer_rewrite_is_rejected` | C |
| Dropped negation → reject | `test_dropped_negation_is_rejected` | C |
| Dropped placeholder → reject | `test_dropped_placeholder_is_rejected` | C |
| Dropped number → reject | `test_dropped_number_is_rejected` | C |
| Safe rewrite accepted | `test_safe_rewrite_is_accepted` | C |
| Generation failure fallback | `test_generation_failure_passes_through` | C |
| Similarity math | `test_identical_text_is_maximally_similar` | C |
| Similarity ordering | `test_overlapping_text_scores_higher_than_disjoint` | C |
| Eval fields populated | `test_evaluate_populates_every_field` | C |
| Contract is abstract | `test_stage_cannot_be_instantiated_directly` | all |
| `run()` is mandatory | `test_subclass_must_implement_run` | all |
| Metrics math + zero-division | `test_stage_metrics_reduction` | all |

**Rule: no test makes a network call, and no test downloads a model by default.**
`test_semantic_rewriter.py` stubs `_generate()`; `test_evaluator.py` stubs the embedder via an
autouse fixture. The two tests that load real models are marked `@pytest.mark.slow`.
Default run:

```bash
pytest -q -m "not slow"      # offline, free, seconds
pytest -q                    # includes the real-model check
```

Note `test_perplexity_pruning.py` *does* load distilGPT2 — but locally, once per session
(the `pruner` fixture is session-scoped), with no network call after the first download.

---

## 12. Benchmark methodology

### 12.1 What gets measured

| Metric | How | Cost |
|--------|-----|------|
| Token reduction % | before/after via local counter | free |
| Cost saved | reduction × per-token price | free |
| Semantic drift | sentence-transformers cosine similarity | free, local |
| Latency delta | wall-clock per stage | free |
| Rewrite rejection rate | `SemanticRewriterStage.rejected` / prompts | free |
| Accuracy retention *(optional)* | run both prompts through a local instruct model (e.g. `flan-t5-base`) on a small labeled set | free, local |

Embedding cosine similarity is the "did we break the meaning?" signal, and it is **local, free,
and offline** — that's the point of choosing it over an LLM-as-judge.

**Report the rewrite rejection rate.** `_is_safe()` discards any rewrite that is empty, longer
than the input, or has dropped a negation / number / placeholder. That counter is a direct
measurement of how often a small local model fails at this task — a genuinely interesting
number, and one no paid-API version of this project would surface.

### 12.2 Two tables

**Table 1 — strategy comparison (the flagship result):**

```
Strategy          Reduction   Similarity    Cost saved
-------------------------------------------------------
rule_only             18.2%        0.981      0.000042
perplexity_only       42.1%        0.938      0.000098
rewrite_only          47.6%        0.961      0.000111
hybrid                55.3%        0.972      0.000129
```

**Table 2 — per-stage waterfall (where the savings came from):**

```
Stage                 Before   After    Reduction
--------------------------------------------------
rule_cleanup              47      39       -17.0%
perplexity_pruning        39      24       -38.5%
semantic_rewrite          24      19       -20.8%
--------------------------------------------------
TOTAL                     47      19       -59.6%
```

The individual-strategy rows are the **evidence that the hybrid design choice was justified** —
without them, "we built a pipeline" is an assertion, not a result.

### 12.3 Report the failures too

Break results out by prompt category. If `already_terse` prompts show ~5% reduction and falling
similarity, **say so**. A compression system that knows when *not* to compress is a better
result than one that claims uniform wins.

---

## 13. Known limitations

State these in the README. Naming your own limitations is what separates a student project from
an engineered one.

### 13.1 Perplexity ≠ importance

Plain surprisal keeps *surprising* tokens, not *important* ones. It will preserve typos, rare
proper nouns, and noise because they are unpredictable. LongLLMLingua's contrastive perplexity
is the fix — see §7.3(b).

### 13.2 The tokenizer question

Two tokenizers are in play, and it's worth knowing why that's fine:

| Purpose | Tokenizer | Status |
|---------|-----------|--------|
| Surprisal scoring | distilGPT2 BPE | The scorer's own vocabulary — no choice |
| Budget, waterfall, cost projection | `tiktoken` `cl100k_base` | **Exact**, not an estimate |

`cl100k_base` *is* the tokenizer GPT-4, GPT-4o and GPT-3.5-turbo use. Because we project costs
against those models specifically, the token counts in the report are the real counts for the
model being priced — there is no approximation to disclose.

**What you must not do** is quote GPT-4 token counts while citing another provider's per-token
price. Different families tokenize the same text differently (often 10–30% apart on code and
non-English text), so a count from one and a rate from another produces a number that is simply
wrong. If you want to report savings for a different model family, count with *that* family's
tokenizer — `count_tokens_for()` exists for exactly this, and §14 extension #4 turns it into a
proper cross-tokenizer comparison.

The remaining honest caveat: the pruning *budget* is denominated in `cl100k_base` tokens while
the pruning *decisions* are made over GPT-2 word units. The two don't map 1:1, so the achieved
ratio in `cl100k_base` terms drifts slightly from `target_ratio`. `last_achieved_ratio` reports
what actually happened — quote that, not the target.

### 13.3 Non-idempotence

Greedy pruning is not idempotent: running at 0.5 twice ≠ running once at 0.25. Document it; do
not chase it.

### 13.4 Non-linear degradation

Output quality degrades non-linearly with compression ratio — expect a cliff somewhere around
60–70% removal. Chart it; the shape of that curve is a genuinely interesting result.

### 13.5 Distribution misalignment

distilGPT2's notion of "predictable" is not the target LLM's. LLMLingua closes this gap by
instruction-tuning the small scorer against the target model's outputs. Out of scope for a
4-day build — but naming the gap and its known fix is a strong interview close.

### 13.6 The local rewriter is the weak link

`flan-t5-base` is a 250M-parameter model. It is *far* weaker at instruction-preserving
compression than a frontier LLM, and it will sometimes truncate, hallucinate, or drop a
constraint. `_is_safe()` catches the detectable failures — empty output, output longer than the
input, a dropped negation / number / placeholder — and falls back to the uncompressed text.

Expect the rewrite stage to underperform perplexity pruning, and possibly to reject a
double-digit percentage of its own outputs. **Report that number.** "Our local rewriter's
outputs failed the safety check 23% of the time, so the hybrid pipeline falls back to the
pruned text in those cases" is a stronger, more honest result than a table that quietly hides
it — and it makes perplexity pruning, the piece you actually built, the star.

### 13.7 Compression is offline, not per-request

Compression is a one-time, template-authoring cost — amortized over every future call that
reuses the compressed prompt. Running the full pipeline per live request would add latency to
the very call you are trying to make cheaper. See §3.2.

### 13.8 Cost figures are projections, not spend

We never call a paid API. Every dollar figure is `tokens_saved × published list price`, clearly
labelled as a projection. Do not present it as measured spend.

---

## 14. Future extensions

Ordered by value-per-hour if you find spare time.

| # | Extension | Effort | Why it's worth it |
|---|-----------|--------|-------------------|
| 1 | **ITPC** — iterative conditional scoring | 3–4 h | The single strongest interview talking point (§7.3a) |
| 2 | **Contrastive perplexity** | 2–3 h | Fixes perplexity's core flaw (§7.3b) |
| 3 | **Adaptive threshold search** | 2 h | Turns the budget controller into a real optimizer: search for the ratio that keeps similarity ≥ 0.95 |
| 4 | **Multi-tokenizer comparison** | 1–2 h | `count_tokens_for()` already exists — sweep `cl100k_base` / `o200k_base` / `gpt2` and show the same compressed text saves a different amount per family (§13.2). Free and local. |
| 5 | **Second scorer model** | 1 h | Swap distilGPT2 for `gpt2` or `gpt2-medium` — does pruning quality improve with a stronger scorer? Empirical, cheap, interesting |
| 6 | **Local accuracy-retention eval** | 3 h | 15–20 labeled pairs run through a local instruct model (`flan-t5-base`) → a proper retention curve instead of one data point. Still zero cost. |
| 7 | **Results chart** | 1 h | One matplotlib PNG: compression % vs. similarity across strategies |
| 8 | **spaCy noun-phrase units** | 2 h | Prune at phrase level like Selective Context — better readability at high ratios |
| 9 | **Persistent cache** | 1 h | Disk-backed rewrite cache (`shelve` / JSON) so benchmark reruns skip flan-t5 generation entirely — saves minutes, not money |
| 10 | **Demo notebook** | 1 h | Token-by-token surprisal walkthrough — the best way to screen-share this in an interview |
| 11 | **Quantized rewriter** | 1–2 h | Load flan-t5 in 8-bit via `bitsandbytes` — same quality, ~4× less RAM, matters on a student laptop |
| 12 | **Ollama rewriter backend** | 2 h | Swap `_generate()` for a local Ollama call (`qwen2.5:3b`, `phi3`) — still free and offline, but a much stronger rewriter than flan-t5-base. Would likely close the gap flagged in §13.6. |

---

## 15. Setup

### 15.1 Zero paid services — read this first

**This project costs nothing to run and needs no account, no API key, and no billing.**
Every model is open-source and runs locally on CPU:

| Role | Model | License | Size on disk |
|------|-------|---------|--------------|
| Perplexity scorer | `distilgpt2` | Apache 2.0 | ~350 MB |
| Semantic rewriter | `google/flan-t5-base` | Apache 2.0 | ~1.0 GB |
| Similarity embedder | `all-MiniLM-L6-v2` | Apache 2.0 | ~90 MB |
| Token counter | `tiktoken` (`cl100k_base`) | MIT | ~2 MB |

**Total ≈ 1.5 GB of models + ~1 GB of packages.** Downloaded once, cached, then fully offline.

The only thing we borrow from the commercial world is **published list prices**, used to
*project* what compression would save if you were paying. We never send a request. Say this
plainly in the README — a projected saving that is labelled as a projection is honest; one
presented as a bill is not.

### 15.2 `requirements.txt`

```
# ── Core: perplexity scoring (Dev B) ─────────────────────────────────────────
torch>=2.2,<3.0                 # CPU wheel is enough — see 15.3
transformers>=4.40,<5.0         # distilGPT2 + flan-t5 loading

# ── Core: token counting and cost projection (Dev A) ─────────────────────────
tiktoken>=0.6

# ── Core: similarity scoring (Dev C) ─────────────────────────────────────────
sentence-transformers>=2.7      # NOTE: pulls torch as a dependency
numpy>=1.26,<3.0

# ── Config ───────────────────────────────────────────────────────────────────
python-dotenv>=1.0              # optional local overrides, see 15.5

# ── Dev / test ───────────────────────────────────────────────────────────────
pytest>=8.0

# ── Optional: results chart only ─────────────────────────────────────────────
matplotlib>=3.8
```

**What each package is for, and who needs it:**

| Package | Needed by | Purpose | Drop it if… |
|---------|-----------|---------|-------------|
| `torch` | B, C | Runs distilGPT2 and flan-t5 on CPU | never — it's the backbone |
| `transformers` | B, C | Loads both HuggingFace models + tokenizers | never |
| `tiktoken` | A, all | Exact GPT-4-family token counts | never |
| `sentence-transformers` | C | Embedding cosine similarity | you drop semantic-drift scoring |
| `numpy` | A, C | Cosine math, benchmark aggregation | never |
| `python-dotenv` | all | Reads optional `.env` overrides | you hardcode paths in `config.py` |
| `pytest` | all | Test suite | never (tests are a grading criterion) |
| `matplotlib` | C | One static results PNG | you skip the chart |

> ⚠️ **`sentence-transformers` pulls in `torch` transitively.** Dev C therefore ends up with
> the same ~200 MB torch install as Dev B even though C never touches distilGPT2. Expect it;
> don't debug it.

### 15.3 The one install gotcha — torch on Windows

`pip install torch` on Windows defaults to the **CUDA build (~2.5 GB)**. You do not need it —
every model here runs on CPU. Install the CPU wheel and save ~2 GB and several minutes:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Order matters: install torch **first** from the CPU index, then the rest. Otherwise
`sentence-transformers` resolves torch from PyPI and pulls the CUDA build anyway.

Verify you got the right one:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.x.x+cpu False     ← correct for this project
```

### 15.4 Per-dev minimal install (Day 1 only)

So nobody sits waiting on a download they don't need on day one. Everyone installs the **full**
`requirements.txt` before the Day-3 integration.

| Dev | Day-1 install | Why |
|-----|---------------|-----|
| **A** | `pip install tiktoken numpy python-dotenv pytest` | ~10 MB, done in seconds. A is unblocked immediately and can ship `count_tokens()` by lunch. |
| **B** | `pip install torch --index-url https://download.pytorch.org/whl/cpu`<br>`pip install transformers pytest` | Needs torch + distilGPT2. **Start this download first thing** — it is the longest install on the team. |
| **C** | `pip install sentence-transformers numpy pytest` | Builds the evaluator and benchmark against stub stages; doesn't need flan-t5 until Day 2. |

### 15.5 `.env` — what goes in it and where to get it

**Short answer: nothing secret. There is no key to obtain.** Because the whole stack is local
and open-source, `.env` holds only optional machine-specific overrides. The project runs
correctly with **no `.env` file at all**.

#### `.env.example` (commit this)

```
# ─────────────────────────────────────────────────────────────────────────────
# token-optimizer — local overrides. ALL OPTIONAL.
# There are NO API keys in this project. Every model is open-source and local.
# Copy to `.env` only if you need one of the overrides below.
# ─────────────────────────────────────────────────────────────────────────────

# Where HuggingFace caches downloaded models.
# Default on Windows: C:\Users\<you>\.cache\huggingface  (~1.5 GB)
# Set this if your C: drive is short on space.
# HF_HOME=D:\models\huggingface

# Silences a noisy tokenizers warning. config.py already defaults this to
# false; override only if you are debugging tokenizer performance.
# TOKENIZERS_PARALLELISM=false

# Work fully offline after the first download (fails fast instead of hanging
# on a network call). Handy on flaky campus wifi.
# HF_HUB_OFFLINE=1

# Which model's published list price to project savings against.
# One of: gpt-4o | gpt-4-turbo | gpt-3.5-turbo
# COST_MODEL=gpt-4o
```

#### Variable reference

| Variable | Required? | Who cares | Where the value comes from |
|----------|-----------|-----------|---------------------------|
| `HF_HOME` | No | B, C | A path **you choose** on your own disk. Not obtained from anywhere. |
| `TOKENIZERS_PARALLELISM` | No | B, C | Literal `false`. Already defaulted in `config.py`. |
| `HF_HUB_OFFLINE` | No | B, C | Literal `1`. Set after models are downloaded. |
| `COST_MODEL` | No | A, C | One of the keys in `config.PRICING`. |

**There is deliberately no `API_KEY` row.** If a teammate adds one, that is a design change
worth discussing first — it reintroduces a cost, a signup, and a network dependency into a
project whose selling point is that it has none.

`config.py` calls `load_dotenv()` at import, so any of these picked up automatically — no
PyCharm plugin needed.

### 15.6 `.gitignore`

```
.venv/
.env
.idea/
*.iml
__pycache__/
*.py[cod]
.pytest_cache/
results/*.csv
results/*.png
```

Commit `.env.example`, never `.env`. Keep `results/.gitkeep` so the folder exists.

### 15.7 Model downloads

First run of each stage pulls its model from HuggingFace. **One time**, then fully offline.

| Model | Pulled by | Size | First-load time (CPU) |
|-------|-----------|------|----------------------|
| `distilgpt2` | Dev B's stage | ~350 MB | ~5 s |
| `google/flan-t5-base` | Dev C's stage | ~1.0 GB | ~10 s |
| `all-MiniLM-L6-v2` | Dev C's evaluator | ~90 MB | ~3 s |

Cache location on Windows: `C:\Users\<you>\.cache\huggingface\hub`
(override with `HF_HOME` — see §15.5).

**Pre-download everything before Day 3** so integration day isn't spent watching progress bars:

```python
# scripts/prefetch.py — run once per machine
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

for name in ("distilgpt2",):
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name)

for name in ("google/flan-t5-base",):
    AutoTokenizer.from_pretrained(name)
    AutoModelForSeq2SeqLM.from_pretrained(name)

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("All models cached.")
```

**Low on disk?** Swap `flan-t5-base` (~1.0 GB) for `google/flan-t5-small` (~300 MB) in
`config.REWRITER_MODEL`. Rewrites get noticeably worse — which is itself a reportable
finding for the README.

### 15.8 PyCharm setup (Windows)

1. **File → Settings → Project → Python Interpreter → Add → Virtualenv**, base interpreter
   Python 3.12, location `.venv`.
2. Open the terminal tab and run the two `pip install` commands from §15.3.
3. **Right-click the `token_optimizer` folder → Mark Directory as → Sources Root.** Without
   this, `from interfaces import ...` raises `ModuleNotFoundError` even though the file is
   right there.
4. Run `main.py` with the green arrow. No run-configuration environment variables needed —
   `config.py` loads `.env` itself.
5. Tests: right-click `tests/` → **Run 'pytest in tests'**. Set the default test runner to
   pytest under **Settings → Tools → Python Integrated Tools**.

### 15.9 Verify your setup — one check per dev

Each dev runs their own snippet on Day 1. If it prints, you are unblocked.

```python
# Dev A — token counting and cost projection
from tokenizer_utils import count_tokens, cost_saved
print(count_tokens("Please carefully review this code."))     # -> 7
print(f"${cost_saved(1000, 400):.6f} projected saving")       # -> $0.001500

# Dev B — surprisal scoring
from stages.perplexity_pruning import PerplexityPruningStage
stage = PerplexityPruningStage(target_ratio=0.5)
units = stage.score("Please carefully review this code for bugs.")
for u in sorted(units, key=lambda x: x.surprisal)[:5]:
    print(f"{u.text:<15}{u.surprisal:>8.2f}{'  [protected]' if u.protected else ''}")

# Dev C — embedding similarity and local rewrite
from evaluator import cosine_similarity
print(cosine_similarity("review the code", "check the code"))   # -> ~0.8
from stages.semantic_rewriter import SemanticRewriterStage
print(SemanticRewriterStage().run(
    "Please carefully review the following code and report any bugs."))
```

### 15.10 Full first run

```bash
git clone <repo> && cd token-optimizer
python -m venv .venv
.venv\Scripts\activate                                        # Windows
# .venv/bin/activate                                          # macOS / Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

copy .env.example .env        # OPTIONAL — nothing in it is required
python scripts/prefetch.py    # optional, pre-pulls ~1.5 GB of models

python main.py                # single-prompt demo + waterfall
python benchmark.py           # full comparison tables -> results/benchmark.csv
pytest -q -m "not slow"       # fast suite: offline, no model downloads
pytest -q                     # full suite: loads the real models
```

### 15.11 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `pip install torch` downloads ~2.5 GB | Default CUDA wheel | Use the CPU index URL (§15.3) |
| `ModuleNotFoundError: interfaces` | Project root not on `sys.path` | Mark the folder as Sources Root in PyCharm, or run from the project root |
| `OSError: distilgpt2 is not a local folder` | No network on first run, or a proxy blocking HuggingFace | Run `scripts/prefetch.py` on a good connection; unset `HF_HUB_OFFLINE` |
| `SSLError` during model download | Corporate/campus proxy MITM | Set `HF_ENDPOINT` to a mirror, or download on another network |
| Benchmark takes minutes | Model reloading per call | Confirm `@lru_cache(maxsize=1)` is on `_load_scorer()` / `_load_rewriter()` |
| `huggingface/tokenizers: parallelism` warning | Fork after tokenizer use | Harmless; `config.py` already sets `TOKENIZERS_PARALLELISM=false` |
| Disk full during download | ~1.5 GB of models on C: | Set `HF_HOME` to another drive (§15.5) |
| Rewriter output is garbage | flan-t5-base is small | Expected sometimes — `stage.rejected` counts it. Try `flan-t5-large`, and report the rejection rate |
| Tests try to download models | Ran the full suite | Use `pytest -m "not slow"` for the fast offline run |

---

## 16. References

- **LLMLingua** — Jiang et al., *LLMLingua: Compressing Prompts for Accelerated Inference of
  Large Language Models*, EMNLP 2023. Budget controller, iterative token-level compression,
  distribution alignment. https://arxiv.org/abs/2310.05736
- **LongLLMLingua** — Jiang et al., *Accelerating and Enhancing LLMs in Long Context Scenarios
  via Prompt Compression*, ACL 2024. Contrastive perplexity, document reordering, adaptive
  ratios. https://aclanthology.org/2024.acl-long.91.pdf
- **LLMLingua-2** — Pan et al., *Data Distillation for Efficient and Faithful Task-Agnostic
  Prompt Compression*, 2024. Token classification with a BERT-size encoder; 3–6× faster.
  https://arxiv.org/pdf/2403.12968
- **Selective Context** — Li et al., *Compressing Context to Enhance Inference Efficiency of
  Large Language Models*, 2023. Self-information scoring, phrase-level units via dependency
  parsing.
- **Survey** — *Prompt Compression for Large Language Models: A Survey*, 2024.
  https://arxiv.org/html/2410.12388v2
- **Reference implementation** — https://github.com/microsoft/LLMLingua

---

*Architecture and design notes for the token-optimizer project. Team of 3, 3–4 day build.*
