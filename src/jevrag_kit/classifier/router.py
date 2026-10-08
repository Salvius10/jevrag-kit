"""Routing in code. Every number and the rule order come from ClassifierConfig.

Rules are tested in order; the first match decides. A passage no rule matches takes the
fallback route. Accepted and conflicting passages are then ordered and capped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from jevrag_kit.classifier.scorer import ScoreOutcome
from jevrag_kit.config import Block, ClassifierConfig
from jevrag_kit.types import Passage, Route


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    decided_by: str | None  # the score key whose comparison settled the route


def decide(scores: Mapping[str, float], config: ClassifierConfig) -> RouteDecision:
    for rule in config.rules:
        if rule.matches(scores[rule.score]):
            return RouteDecision(rule.route, rule.reason, rule.score)
    fallback = config.fallback
    return RouteDecision(fallback.route, fallback.reason, fallback.decided_by)


def threshold_sides(scores: Mapping[str, float], config: ClassifierConfig) -> dict[str, dict]:
    """Which side of its threshold each rule's score falls on, by the rule's own comparison."""
    return {rule.key: {"threshold": rule.threshold, "side": rule.side(scores[rule.score])} for rule in config.rules}


@dataclass
class RoutingResult:
    accepted: list[Passage] = field(default_factory=list)
    conflicting: list[Passage] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)  # one per passage, in input order


def _order_value(scores: Mapping[str, float], block: Block) -> float:
    return scores[block.order_by] if block.order_by else 0.0


def route_all(
    passages: Sequence[Passage],
    outcomes: Sequence[ScoreOutcome],
    config: ClassifierConfig,
    ranks: Sequence[int] | None = None,
) -> RoutingResult:
    """Route every scored passage, then order and cap the accept and conflict blocks.

    `ranks` are retrieval ranks (1 = best), used to break ties; by default the input order.
    A passage without scores is dropped with reason `score_failed`.
    """
    passages = list(passages)
    ranks = list(ranks) if ranks is not None else list(range(1, len(passages) + 1))
    if len(ranks) != len(passages):
        raise ValueError(f"got {len(ranks)} ranks for {len(passages)} passages")
    ids = [p.id for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("passage ids must be unique")

    required = config.required_scores()
    by_id = {o.passage_id: o for o in outcomes}
    records: dict[str, dict] = {}
    accepted: list[tuple[float, int, Passage]] = []
    conflicting: list[tuple[float, int, Passage]] = []

    for passage, rank in zip(passages, ranks):
        pid = passage.id
        outcome = by_id.get(pid)
        if outcome is None or outcome.scores is None:
            records[pid] = {
                "passage_id": pid,
                "route": "drop",
                "reason": "score_failed",
                "decided_by": None,
                "included": False,
                "sides": None,
            }
            continue
        values = outcome.scores.scores
        missing = required - values.keys()
        if missing:
            raise ValueError(
                f"scores for passage '{pid}' lack {sorted(missing)}; the scorer must answer every question the rules use"
            )
        d = decide(values, config)
        records[pid] = {
            "passage_id": pid,
            "route": d.route,
            "reason": d.reason,
            "decided_by": d.decided_by,
            "included": False,
            "sides": threshold_sides(values, config),
        }
        if d.route == "accept":
            accepted.append((_order_value(values, config.accept), rank, passage))
        elif d.route == "conflict":
            conflicting.append((_order_value(values, config.conflict), rank, passage))

    accepted.sort(key=lambda a: (-a[0], a[1], a[2].id))
    conflicting.sort(key=lambda c: (-c[0], c[1], c[2].id))
    result = RoutingResult()
    for block, items, limit in (
        (result.accepted, accepted, config.accept.limit),
        (result.conflicting, conflicting, config.conflict.limit),
    ):
        for position, (_, _, passage) in enumerate(items):
            record = records[passage.id]
            if position < limit:
                block.append(passage)
                record["included"] = True
                record["position"] = position + 1
            else:
                record["capped"] = True
    result.records = [records[p.id] for p in passages]
    return result
