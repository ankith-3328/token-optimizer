from __future__ import annotations

import re

from interfaces import CompressionStage


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

_FILLER_RE = re.compile(
    "|".join(f"(?:{p})" for p in FILLER_PHRASES),
    re.I,
)

_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")

_PROTECT = re.compile(
    r"`{3}.*?`{3}|`[^`]*`|\{[^{}]*\}",
    re.S,
)


class RuleCleanupStage(CompressionStage):
    name = "rule_cleanup"

    def run(self, text: str) -> str:
        if not text or not text.strip():
            return text

        protected: list[str] = []

        def _stash(m: re.Match) -> str:
            protected.append(m.group(0))
            return f"\x00{len(protected) - 1}\x00"

        working = _PROTECT.sub(_stash, text)

        working = _FILLER_RE.sub(" ", working)
        working = _SPACE_BEFORE_PUNCT.sub(r"\1", working)
        working = _MULTI_SPACE.sub(" ", working)
        working = _MULTI_NEWLINE.sub("\n\n", working)

        working = re.sub(
            r"\x00(\d+)\x00",
            lambda m: protected[int(m.group(1))],
            working,
        )

        working = working.strip()

        if working and text.strip()[:1].isupper():
            working = working[0].upper() + working[1:]

        return working