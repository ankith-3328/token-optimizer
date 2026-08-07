from __future__ import annotations
import re


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:3b"

SCORER_MODEL = "distilgpt2"
SCORER_MAX_CTX = 1024
SCORER_STRIDE = 512

TARGET_RATIO = 0.7            
COARSE_PASS_ENABLED = True     
COARSE_KEEP_FIRST = True
COARSE_KEEP_LAST = True

TOKENIZER_ENCODING = "cl100k_base"

PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}

COST_MODEL = "gpt-4o"

PROTECTED_WORDS: set[str] = {
    "not", "no", "never", "none", "neither", "nor", "without",
    "except", "unless", "cannot", "cant", "can't", "dont", "don't",
    "doesnt", "doesn't", "wont", "won't", "isnt", "isn't",
    "must", "always", "only", "required", "require", "shall", "should",
    "least", "most", "exactly", "all", "every", "each", "any",
    "before", "after", "instead", "rather", "than",
}

PROTECTED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\{[^{}]*\}"),
    re.compile(r"`[^`]*`"),
    re.compile(r"`{3}.*?`{3}", re.S),
    re.compile(r'"[^"]*"'),
    re.compile(r"'[^']*'"),
    re.compile(r"\b\d[\d,.\-/:%]*\b"),
    re.compile(r"\$\d[\d,.]*"),
]