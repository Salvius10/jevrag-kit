"""The relation check: does the passage support, contradict, or say nothing about the claim?"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from jevrag_kit.config import RelationQuestion
from jevrag_kit.types import RelationResult

if TYPE_CHECKING:
    from jevrag_kit.config import JevConfig

RELATION_KEY = "relation"


class ClaimVerifier(Protocol):
    def relation(self, claim: str, section: str) -> RelationResult: ...


def build_relation_question(question: RelationQuestion | None = None) -> Any:
    from typesafe_sdk import Choice

    q = question or RelationQuestion()
    return Choice(
        instructions=q.instructions,
        criteria={"supports": q.supports, "contradicts": q.contradicts, "says_nothing": q.says_nothing},
    )


class TypeSafeClaimVerifier:
    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        question: RelationQuestion | None = None,
        *,
        retries: int = 2,
        timeout: float = 120.0,
        base_url: str | None = None,
        transport: Any = None,
    ):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        # The SDK retries with backoff on 408/429/5xx and connection errors.
        self._client = TypeSafeClient(
            api_key=api_key,
            model=model,
            timeout=timeout,
            retry=RetryPolicy(max_retries=retries),
            transport=transport,
            base_url=base_url,
        )
        self._questions = {RELATION_KEY: build_relation_question(question)}
        self.model = model

    @classmethod
    def from_config(cls, config: JevConfig, api_key: str, *, transport: Any = None) -> TypeSafeClaimVerifier:
        return cls(
            api_key,
            config.checker_model,
            config.checker.relation,
            retries=config.checker.retries,
            timeout=config.typesafe.timeout,
            base_url=config.typesafe.base_url,
            transport=transport,
        )

    def relation(self, claim: str, section: str) -> RelationResult:
        response = self._client.system_one(
            state={"claim": claim, "section": section}, questions=self._questions, model=self.model
        )
        answer = response.answers[RELATION_KEY]
        return RelationResult(
            choice=answer.choice,
            probabilities={k: float(v) for k, v in answer.probabilities.items()},
            confidence=float(answer.confidence),
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        )
