"""Build the released answer from shipped claims. The LLM's own prose is never shown."""

from __future__ import annotations

from typing import Sequence

from jevrag_kit.checker.policy import ClaimCheck
from jevrag_kit.config import AnswerConfig
from jevrag_kit.types import Answer, AnswerClaim, Draft

_DEFAULT = AnswerConfig()


def withheld_sentence(n: int, config: AnswerConfig = _DEFAULT) -> str:
    template = config.withheld_one if n == 1 else config.withheld_many
    return template.format(n=n)


def abstained(reason: str, missing: str | None = None, withheld: int = 0) -> Answer:
    return Answer(status="abstained", text=None, reason=reason, missing=missing, withheld_count=withheld)


def assemble_answer(checks: Sequence[ClaimCheck], draft: Draft, config: AnswerConfig = _DEFAULT) -> Answer:
    """Premise corrections first, then answers; one cited line per shipped claim."""
    shipped = [c for c in checks if c.action == "ship"]
    withheld = sum(1 for c in checks if c.action == "review")
    ordered = [c for c in shipped if c.claim.type == "premise_correction"] + [
        c for c in shipped if c.claim.type == "answer"
    ]

    if not ordered:
        if withheld:
            reason = "pending_review"
        elif draft.insufficient:
            reason = "insufficient_evidence"
        else:
            reason = "no_verified_claims"
        return abstained(reason, draft.missing, withheld)

    lines = [
        config.citation_template.format(
            text=c.claim.text, passage_id=c.claim.passage_id, claim_id=c.claim.id, type=c.claim.type
        )
        for c in ordered
    ]
    if withheld:
        lines.append(withheld_sentence(withheld, config))

    source_ids: list[str] = []
    for c in ordered:
        if c.claim.passage_id not in source_ids:
            source_ids.append(c.claim.passage_id)

    return Answer(
        status="partial" if withheld else "answered",
        text=config.line_separator.join(lines),
        claims=[
            AnswerClaim(
                id=c.claim.id,
                type=c.claim.type,
                text=c.claim.text,
                passage_id=c.claim.passage_id,
                quote=c.claim.quote,
                confidence=c.confidence,
            )
            for c in ordered
        ],
        source_ids=source_ids,
        withheld_count=withheld,
    )
