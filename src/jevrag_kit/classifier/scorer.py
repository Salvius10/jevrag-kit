"""Passage scoring: every configured question about one (query, passage) pair in one TypeSafe request.

None of the questions decides whether to keep a passage; router.decide() does that, in code.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

from pydantic_core import to_jsonable_python

from jevrag_kit.config import ClassifierConfig, Question
from jevrag_kit.types import Passage, PassageScores

if TYPE_CHECKING:
    from jevrag_kit.config import JevConfig

ATTEMPTS = 3
BACKOFF_SECONDS = 0.5
DEFAULT_STATE_FIELDS = ("id", "title", "text")


class PassageScorer(Protocol):
    def score(self, query: str, passage: Passage) -> PassageScores: ...


class TransientError(Exception):
    """Raise from a custom scorer for a failure worth retrying."""


def is_transient(exc: BaseException) -> bool:
    """Connection errors, timeouts, rate limits, and server errors are retried; nothing else is."""
    if isinstance(exc, TransientError):
        return True
    import typesafe_sdk as ts

    return isinstance(
        exc,
        (
            ts.TypeSafeAPIConnectionError,
            ts.TypeSafeAPITimeoutError,
            ts.TypeSafeRateLimitError,
            ts.TypeSafeInternalServerError,
        ),
    )


def build_questions(questions: Mapping[str, Question]) -> dict[str, Any]:
    """TypeSafe Noul question objects, keyed like the configuration."""
    from typesafe_sdk import Noul, NoulCriteria

    out: dict[str, Any] = {}
    for key, q in questions.items():
        criteria = {name: text for name, text in (("true", q.true_means), ("false", q.false_means)) if text is not None}
        out[key] = Noul(instructions=q.instructions, criteria=NoulCriteria(**criteria) if criteria else None)
    return out


def passage_state(query: str, passage: Passage, fields: Sequence[str] = DEFAULT_STATE_FIELDS) -> dict:
    """The state TypeSafe evaluates: the query and the listed passage fields that have a value."""
    values: dict[str, Any] = {}
    for name in fields:
        value = passage.get(name)
        if value is not None:
            values[name] = to_jsonable_python(value)
    return {"query": query, "passage": values}


class TypeSafePassageScorer:
    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        questions: Mapping[str, Question] | None = None,
        state_fields: Sequence[str] = DEFAULT_STATE_FIELDS,
        *,
        timeout: float = 120.0,
        base_url: str | None = None,
        transport: Any = None,
    ):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        # Retries are made by score_one, so attempts are counted (and testable) in one place.
        self._client = TypeSafeClient(
            api_key=api_key,
            model=model,
            timeout=timeout,
            retry=RetryPolicy(max_retries=0),
            transport=transport,
            base_url=base_url,
        )
        self._questions = build_questions(questions if questions is not None else ClassifierConfig().questions)
        self.state_fields = tuple(state_fields)
        self.model = model

    @classmethod
    def from_config(cls, config: JevConfig, api_key: str, *, transport: Any = None) -> TypeSafePassageScorer:
        return cls(
            api_key,
            config.classifier_model,
            config.classifier.questions,
            config.classifier.state_fields,
            timeout=config.typesafe.timeout,
            base_url=config.typesafe.base_url,
            transport=transport,
        )

    def score(self, query: str, passage: Passage) -> PassageScores:
        response = self._client.system_one(
            state=passage_state(query, passage, self.state_fields), questions=self._questions, model=self.model
        )
        return PassageScores(
            scores={key: float(response.answers[key].noul) for key in self._questions},
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        )


@dataclass
class ScoreOutcome:
    passage_id: str
    scores: PassageScores | None
    error: str | None
    attempts: int
    latency_ms: int

    @property
    def failed(self) -> bool:
        return self.scores is None


def score_one(
    scorer: PassageScorer,
    query: str,
    passage: Passage,
    attempts: int = ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    *,
    backoff_seconds: float = BACKOFF_SECONDS,
    transient: Callable[[BaseException], bool] = is_transient,
) -> ScoreOutcome:
    """Score one passage. Transient errors are retried with exponential backoff; any failure
    that remains is returned as an outcome with `error` set, never raised."""
    started = time.perf_counter()
    for attempt in range(1, attempts + 1):
        try:
            scores = scorer.score(query, passage)
            return ScoreOutcome(passage.id, scores, None, attempt, _ms(started))
        except Exception as exc:  # noqa: BLE001 - any failure marks the passage score_failed
            if not transient(exc) or attempt == attempts:
                return ScoreOutcome(passage.id, None, f"{type(exc).__name__}: {exc}", attempt, _ms(started))
            sleep(backoff_seconds * 2 ** (attempt - 1))
    raise ValueError("attempts must be at least 1")


def score_all(
    scorer: PassageScorer,
    query: str,
    passages: Sequence[Passage],
    max_workers: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    *,
    attempts: int = ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
    transient: Callable[[BaseException], bool] = is_transient,
) -> list[ScoreOutcome]:
    """One request per passage, run concurrently; outcomes keep the input order."""
    if not passages:
        return []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(
            pool.map(
                lambda p: score_one(
                    scorer, query, p, attempts, sleep, backoff_seconds=backoff_seconds, transient=transient
                ),
                passages,
            )
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
