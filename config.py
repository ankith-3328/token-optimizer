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


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"