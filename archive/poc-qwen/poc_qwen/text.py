"""Sentence splitting and chunk grouping.

Each model call must stay well under the Metal watchdog (~40 s of audio per
call has been reported to kill MLX kernels), so long inputs are split on
sentence boundaries and grouped into chunks of at most `max_chars`.
"""

from __future__ import annotations

import re

# Split after . ! ? … (plus closing quotes/brackets) followed by whitespace.
# Abbreviations like "Dr." / "e.g." and decimals like "9.15" are protected
# because the pattern requires whitespace after the terminator and a
# capital/quote/digit start for the next sentence is not required — so only
# abbreviations followed by a space (e.g. "Dr. Smith") still split; that is an
# accepted limitation for a demo.
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|No)\.$", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    buf = ""
    for piece in _SENTENCE_END.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        buf = f"{buf} {piece}".strip() if buf else piece
        if _ABBREV.search(buf):
            continue
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return parts


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Group sentences into chunks of at most max_chars (a single longer sentence becomes its own chunk)."""
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks
