"""Data shared by the classifier, the LLM layer, and the claims checker.

The Literal vocabularies below are the contract between the three stages; configuration
changes wording and numbers, never these names.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from jevrag_kit.text import normalize

Route = Literal["accept", "conflict", "drop"]
ClaimType = Literal["answer", "premise_correction"]
Relation = Literal["supports", "contradicts", "says_nothing"]
LocateStatus = Literal["found", "reattributed", "missing", "too_short"]
Verdict = Literal["verified", "unsupported", "contradicted", "fabricated"]
Action = Literal["ship", "review", "drop"]
AnswerStatus = Literal["answered", "partial", "abstained"]

CLAIM_TYPES: tuple[str, ...] = ("answer", "premise_correction")
RELATIONS: tuple[str, ...] = ("supports", "contradicts", "says_nothing")


class Passage(BaseModel):
    """One unit of source text, as your retrieval returns it.

    Put project-specific values (source, date, jurisdiction, ...) in `metadata`, or subclass
    Passage to add typed fields. Either kind can be sent to TypeSafe (classifier.state_fields)
    and shown to the LLM (llm.prompt.passage_template).
    """

    id: str = Field(min_length=1)
    text: str
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def normalized_text(self) -> str:
        return normalize(self.text)

    def has(self, name: str) -> bool:
        """True for a declared field (including subclass fields) or a metadata key."""
        return name in type(self).model_fields or name in self.metadata

    def get(self, name: str) -> Any:
        """A declared field, else a metadata value, else None."""
        if name in type(self).model_fields:
            return getattr(self, name)
        return self.metadata.get(name)


class PassageScores(BaseModel):
    """The classifier's answers for one (query, passage) pair: question key -> probability."""

    scores: dict[str, float]
    input_tokens: int = 0
    output_tokens: int = 0

    def __getitem__(self, key: str) -> float:
        return self.scores[key]


class Claim(BaseModel):
    id: str
    type: ClaimType
    text: str
    passage_id: str
    quote: str


class Draft(BaseModel):
    """The LLM's structured output: claims, each citing one passage with a verbatim quote."""

    insufficient: bool
    missing: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    raw: dict | None = None
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0


class RelationResult(BaseModel):
    choice: Relation
    probabilities: dict[str, float]
    confidence: float
    input_tokens: int = 0
    output_tokens: int = 0


class AnswerClaim(BaseModel):
    id: str
    type: ClaimType
    text: str
    passage_id: str
    quote: str
    confidence: float | None


class Answer(BaseModel):
    """The released answer, assembled in code from shipped claims only."""

    status: AnswerStatus
    text: str | None
    reason: str | None = None
    missing: str | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    withheld_count: int = 0
