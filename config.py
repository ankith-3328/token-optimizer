from __future__ import annotations

import os
import re


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:3b"

SCORER_MODEL = "distilgpt2"
SCORER_MAX_CTX = 1024          # distilGPT2 hard context limit
SCORER_STRIDE = 512            # sliding-window stride for long inputs

TARGET_RATIO = 0.7            
COARSE_PASS_ENABLED = True     
COARSE_KEEP_FIRST = True
COARSE_KEEP_LAST = True

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