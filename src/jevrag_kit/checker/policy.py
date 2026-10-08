"""Claim verification and the release policy.

| Verdict      | Confidence              | Action (defaults)                                   |
|--------------|-------------------------|-----------------------------------------------------|
| verified     | at or above auto_accept | ship                                                |
| verified     | below auto_accept       | low_confidence_action: review (reason low_confidence) |
| unsupported  | any                     | unsupported_action: review (reason unsupported)     |
| contradicted | any                     | drop                                                |
| fabricated   | n/a (quote not found)   | drop, and no model call is made                     |

A claim whose relation check fails with an error is dropped: an unverifiable claim never ships.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Mapping

from jevrag_kit.checker.quotes import locate
from jevrag_kit.checker.relation import ClaimVerifier
from jevrag_kit.config import CheckerConfig
from jevrag_kit.types import Action, Claim, Draft, LocateStatus, Passage, Relation, Verdict

RELATION_TO_VERDICT: dict[str, Verdict] = {
    "supports": "verified",
    "contradicts": "contradicted",
    "says_nothing": "unsupported",
}

_DEFAULT = CheckerConfig()


def _held(action: str, reason: str) -> tuple[Action, str | None]:
    return ("review", reason) if action == "review" else ("drop", None)


def release(verdict: Verdict, confidence: float | None, policy: CheckerConfig = _DEFAULT) -> tuple[Action, str | None]:
    """(action, review reason) for one verdict."""
    if verdict == "verified":
        if confidence is not None and confidence >= policy.auto_accept:
            return "ship", None
        return _held(policy.low_confidence_action, "low_confidence")
    if verdict == "unsupported":
        return _held(policy.unsupported_action, "unsupported")
    return "drop", None


@dataclass
class ClaimCheck:
    round: int
    claim: Claim  # passage_id corrected when the quote was reattributed
    cited_passage_id: str
    locate: LocateStatus
    relation: Relation | None
    probabilities: dict[str, float] | None
    confidence: float | None
    verdict: Verdict
    action: Action
    review_reason: str | None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claim"] = self.claim.model_dump()
        return d


def check_claim(
    claim: Claim,
    supplied: Mapping[str, Passage],
    verifier: ClaimVerifier,
    policy: CheckerConfig = _DEFAULT,
    round_no: int = 1,
) -> ClaimCheck:
    started = time.perf_counter()
    located = locate(claim, supplied, policy.min_quote_chars)
    if located.passage is None:
        return ClaimCheck(round_no, claim, claim.passage_id, located.status, None, None, None, "fabricated", "drop", None)

    final_claim = claim.model_copy(update={"passage_id": located.passage.id})
    try:
        result = verifier.relation(claim.text, located.passage.text)
    except Exception as exc:  # noqa: BLE001 - an unverifiable claim never ships
        return ClaimCheck(
            round_no, final_claim, claim.passage_id, located.status, None, None, None, "unsupported", "drop", None,
            error=f"{type(exc).__name__}: {exc}", latency_ms=_ms(started),
        )
    verdict = RELATION_TO_VERDICT[result.choice]
    action, reason = release(verdict, result.confidence, policy)
    return ClaimCheck(
        round_no,
        final_claim,
        claim.passage_id,
        located.status,
        result.choice,
        result.probabilities,
        result.confidence,
        verdict,
        action,
        reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=_ms(started),
    )


def verify_draft(
    draft: Draft,
    supplied: Mapping[str, Passage],
    verifier: ClaimVerifier,
    policy: CheckerConfig = _DEFAULT,
    round_no: int = 1,
    max_workers: int | None = None,
) -> list[ClaimCheck]:
    """Check every claim concurrently; results keep the draft's order."""
    if not draft.claims:
        return []
    workers = policy.workers if max_workers is None else max_workers
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda c: check_claim(c, supplied, verifier, policy, round_no), draft.claims))


def needs_regeneration(checks: list[ClaimCheck]) -> bool:
    """Nothing shipped, and at least one claim was dropped or withheld."""
    return bool(checks) and not any(c.action == "ship" for c in checks)


_WHY = {
    "too_short": "the quote is too short to verify",
    "missing": "the quote was not found word for word in any supplied passage",
}


def feedback_for(checks: list[ClaimCheck]) -> list[str]:
    """One line per claim that did not ship, telling the LLM why, for the regeneration prompt."""
    lines = []
    for c in checks:
        if c.action == "ship":
            continue
        if c.verdict == "fabricated":
            why = _WHY[c.locate]
        elif c.error:
            why = "it could not be checked"
        elif c.verdict == "contradicted":
            why = f"passage [{c.claim.passage_id}] contradicts it"
        elif c.verdict == "unsupported":
            why = f"passage [{c.claim.passage_id}] does not address it"
        else:
            why = f"support in [{c.claim.passage_id}] was not clear enough"
        lines.append(f'{c.claim.id} ("{c.claim.text}"): {why}')
    return lines


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
