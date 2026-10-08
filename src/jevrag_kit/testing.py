"""Deterministic fakes for unit tests in projects that use jevrag-kit. No network; they record calls.

    from jevrag_kit import Engine
    from jevrag_kit.testing import FakeGenerator, FakeScorer, FakeVerifier, make_claim, make_draft, make_passage

    engine = Engine(FakeScorer({"p1": {"is_relevant": 0.9, ...}}), FakeGenerator([make_draft(...)]), FakeVerifier())
"""

from __future__ import annotations

from typing import Any, Mapping

from jevrag_kit.llm.prompt import Prompt
from jevrag_kit.types import Claim, Draft, Passage, PassageScores, RelationResult


def make_passage(pid: str, text: str, title: str | None = None, **metadata: Any) -> Passage:
    return Passage(id=pid, text=text, title=f"Doc: {pid}" if title is None else title, metadata=metadata)


def make_scores(values: Mapping[str, float] | None = None, *, input_tokens: int = 100, output_tokens: int = 5, **more: float) -> PassageScores:
    return PassageScores(scores={**(values or {}), **more}, input_tokens=input_tokens, output_tokens=output_tokens)


def grounded_scores(rel: float = 0.9, ev: float = 0.9, con: float = 0.05, inj: float = 0.02, ans: float = 0.5) -> PassageScores:
    """Scores for the five default questions."""
    return make_scores(
        is_relevant=rel,
        contains_answer_evidence=ev,
        contradicts_query_premise=con,
        contains_prompt_injection=inj,
        answers_query=ans,
    )


def make_claim(cid: str, text: str, pid: str, quote: str, ctype: str = "answer") -> Claim:
    return Claim(id=cid, type=ctype, text=text, passage_id=pid, quote=quote)


def make_draft(*claims: Claim, insufficient: bool = False, missing: str | None = None) -> Draft:
    return Draft(insufficient=insufficient, missing=missing, claims=list(claims), raw={"fake": True})


class FakeScorer:
    """Scores by passage id: a PassageScores, a {key: probability} mapping, or an exception to raise."""

    def __init__(self, table: Mapping[str, Any], default: PassageScores | Mapping[str, float] | None = None):
        self.table = dict(table)
        self.default = default if default is not None else grounded_scores(rel=0.1, ev=0.1)
        self.calls: list[str] = []

    def score(self, query: str, passage: Passage) -> PassageScores:
        self.calls.append(passage.id)
        value = self.table.get(passage.id, self.default)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, PassageScores):
            return value
        return make_scores(value)


class FakeGenerator:
    """Returns scripted drafts in order; records each prompt it was given."""

    def __init__(self, drafts: list[Draft]):
        self.drafts = list(drafts)
        self.prompts: list[Prompt] = []

    @property
    def calls(self) -> list[dict]:
        return [
            {"accepted": list(p.accepted_ids), "conflicting": list(p.conflicting_ids), "feedback": p.feedback}
            for p in self.prompts
        ]

    def generate(self, prompt: Prompt) -> Draft:
        self.prompts.append(prompt)
        if not self.drafts:
            raise AssertionError("generator called more times than scripted")
        return self.drafts.pop(0)


class FakeVerifier:
    """Relation by claim text; anything unlisted gets `default` (supports at 0.95)."""

    def __init__(self, table: Mapping[str, tuple[str, float]] | None = None, default: tuple[str, float] = ("supports", 0.95)):
        self.table = dict(table or {})
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def relation(self, claim: str, section: str) -> RelationResult:
        self.calls.append((claim, section))
        choice, confidence = self.table.get(claim, self.default)
        rest = (1 - confidence) / 2
        probabilities = {"supports": rest, "contradicts": rest, "says_nothing": rest, choice: confidence}
        return RelationResult(
            choice=choice, probabilities=probabilities, confidence=confidence, input_tokens=40, output_tokens=2
        )
