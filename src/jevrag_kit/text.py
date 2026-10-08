"""Canonical text normalization, used to match claim quotes against passage text.

Matching after normalize() is exact. There is deliberately no fuzzy matching: a reworded
quote must fail. If you store a normalized copy of passage text, produce it with this function.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC, straight quotes, every whitespace run collapsed to one space, trimmed."""
    text = unicodedata.normalize("NFKC", text)
    return _SPACE.sub(" ", text.translate(_QUOTES)).strip()
