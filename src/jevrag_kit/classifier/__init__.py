"""JEV passage classifier: TypeSafe scores each (query, passage) pair; configured rules route it."""

from jevrag_kit.classifier.router import RouteDecision, RoutingResult, decide, route_all, threshold_sides
from jevrag_kit.classifier.scorer import (
    ATTEMPTS,
    BACKOFF_SECONDS,
    PassageScorer,
    ScoreOutcome,
    TransientError,
    TypeSafePassageScorer,
    build_questions,
    is_transient,
    passage_state,
    score_all,
    score_one,
)

__all__ = [
    "ATTEMPTS",
    "BACKOFF_SECONDS",
    "PassageScorer",
    "RouteDecision",
    "RoutingResult",
    "ScoreOutcome",
    "TransientError",
    "TypeSafePassageScorer",
    "build_questions",
    "decide",
    "is_transient",
    "passage_state",
    "route_all",
    "score_all",
    "score_one",
    "threshold_sides",
]
