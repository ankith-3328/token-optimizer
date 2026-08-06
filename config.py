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
# Ollama model used by the `semantic_rewriter` stage. Kept here for tests.
OLLAMA_MODEL = "qwen2.5:3b"

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
