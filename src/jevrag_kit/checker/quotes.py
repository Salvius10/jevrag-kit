"""Find a claim's quote in the supplied passages. Plain string matching, no model.

Matching is exact after normalize(). There is deliberately no fuzzy matching: a reworded quote fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from jevrag_kit.text import normalize
from jevrag_kit.types import Claim, LocateStatus, Passage


@dataclass(frozen=True)
class LocateResult:
    status: LocateStatus
    passage: Passage | None


def locate(claim: Claim, supplied: Mapping[str, Passage], min_quote_chars: int) -> LocateResult:
    needle = normalize(claim.quote)
    if len(needle) < min_quote_chars:
        return LocateResult("too_short", None)
    cited = supplied.get(claim.passage_id)
    if cited is not None and needle in cited.normalized_text:
        return LocateResult("found", cited)
    for passage in supplied.values():  # the quote is real but attributed to the wrong passage
        if needle in passage.normalized_text:
            return LocateResult("reattributed", passage)
    return LocateResult("missing", None)
