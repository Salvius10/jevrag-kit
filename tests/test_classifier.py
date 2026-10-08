import json
from datetime import date

import httpx2
import pytest

from helpers import noul_response, typesafe_transport
from jevrag_kit.classifier import (
    ScoreOutcome,
    TransientError,
    TypeSafePassageScorer,
    build_questions,
    decide,
    passage_state,
    route_all,
    score_all,
    score_one,
    threshold_sides,
)
from jevrag_kit.config import ClassifierConfig, parse_config
from jevrag_kit.types import Passage, PassageScores


def S(rel=0.9, ev=0.9, con=0.1, inj=0.05, ans=0.5) -> dict[str, float]:
    return {
        "is_relevant": rel,
        "contains_answer_evidence": ev,
        "contradicts_query_premise": con,
        "contains_prompt_injection": inj,
        "answers_query": ans,
    }


def P(pid: str, **metadata) -> Passage:
    return Passage(id=pid, title=f"T {pid}", text=f"text {pid}", metadata=metadata)


@pytest.fixture
def c(config) -> ClassifierConfig:
    return config.classifier


def route(scores, cfg):
    return decide(scores, cfg).route


# --- decide(): one test per default rule ------------------------------------------------------


def test_injection_drops(c):
    assert decide(S(inj=0.71), c).reason == "injection"
    assert route(S(inj=0.71), c) == "drop"


def test_contradiction_goes_to_conflict(c):
    assert route(S(con=0.71), c) == "conflict"


def test_irrelevant_drops(c):
    d = decide(S(rel=0.44), c)
    assert (d.route, d.reason, d.decided_by) == ("drop", "not_relevant", "is_relevant")


def test_evidence_accepts(c):
    d = decide(S(ev=0.56), c)
    assert (d.route, d.decided_by) == ("accept", "contains_answer_evidence")


def test_no_evidence_falls_back_to_drop(c):
    d = decide(S(ev=0.55), c)
    assert (d.route, d.reason, d.decided_by) == ("drop", "no_evidence", "contains_answer_evidence")


def test_boundaries_use_the_configured_comparisons(c):
    assert route(S(inj=0.70), c) == "accept"  # "above" is strict
    assert route(S(con=0.70), c) == "accept"
    assert route(S(rel=0.45), c) == "accept"  # "below" is strict, so the floor is inclusive


def test_precedence_follows_rule_order(c):
    assert route(S(inj=0.95, ev=0.99, rel=0.99, ans=0.99), c) == "drop"
    assert decide(S(inj=0.95, con=0.95), c).reason == "injection"
    assert route(S(con=0.95, ev=0.99), c) == "conflict"
    assert route(S(con=0.95, rel=0.1, ev=0.1), c) == "conflict"


def test_threshold_sides_match_the_rule_comparisons(c):
    sides = threshold_sides(S(rel=0.45, ev=0.55, con=0.71, inj=0.2), c)
    assert list(sides) == ["contains_prompt_injection", "contradicts_query_premise", "is_relevant", "contains_answer_evidence"]
    assert sides["is_relevant"]["side"] == "above"
    assert sides["contains_answer_evidence"]["side"] == "below"
    assert sides["contradicts_query_premise"]["side"] == "above"
    assert sides["contains_prompt_injection"] == {"threshold": 0.70, "side": "below"}


def test_inclusive_comparisons_and_named_rules():
    cfg = parse_config(
        "classifier:\n  rules:\n"
        "    - {score: is_relevant, when: at_most, threshold: 0.3, route: drop, reason: off_topic, name: floor}\n"
        "    - {score: contains_answer_evidence, when: at_least, threshold: 0.6, route: accept, reason: evidence}\n"
        "  fallback: {route: drop, reason: weak, decided_by: null}\n"
    ).classifier
    assert decide(S(rel=0.3), cfg).reason == "off_topic"
    assert decide(S(rel=0.31, ev=0.6), cfg).route == "accept"
    d = decide(S(rel=0.31, ev=0.59), cfg)
    assert (d.route, d.reason, d.decided_by) == ("drop", "weak", None)
    sides = threshold_sides(S(rel=0.3, ev=0.6), cfg)
    assert sides == {"floor": {"threshold": 0.3, "side": "below"}, "contains_answer_evidence": {"threshold": 0.6, "side": "above"}}


# --- route_all(): ordering, caps, records ----------------------------------------------------


def _outcome(pid, scores):
    return ScoreOutcome(pid, PassageScores(scores=scores) if scores else None, None if scores else "boom", 1, 1)


def test_route_all_orders_caps_and_records(config):
    c = config.with_overrides({"classifier.accept.limit": 2, "classifier.conflict.limit": 1}).classifier
    passages = [P(f"p{i}") for i in range(1, 8)]
    table = {
        "p1": S(ans=0.2),
        "p2": S(ans=0.9),
        "p3": S(ans=0.6),
        "p4": S(con=0.8),
        "p5": S(con=0.95),
        "p6": S(inj=0.99),
        "p7": None,
    }
    result = route_all(passages, [_outcome(pid, s) for pid, s in table.items()], c)
    assert [p.id for p in result.accepted] == ["p2", "p3"]  # by answers_query, capped at 2
    assert [p.id for p in result.conflicting] == ["p5"]  # by contradiction, capped at 1
    rec = {r["passage_id"]: r for r in result.records}
    assert [r["passage_id"] for r in result.records] == [f"p{i}" for i in range(1, 8)]
    assert rec["p1"]["route"] == "accept" and rec["p1"]["capped"] and not rec["p1"]["included"]
    assert rec["p2"]["position"] == 1 and rec["p3"]["position"] == 2
    assert rec["p6"]["reason"] == "injection" and not rec["p6"]["included"]
    assert rec["p7"] == {"passage_id": "p7", "route": "drop", "reason": "score_failed", "decided_by": None, "included": False, "sides": None}


def test_route_all_breaks_ties_by_rank_then_id(c):
    passages = [P("b"), P("a"), P("c")]
    outcomes = [_outcome(p.id, S(ans=0.5)) for p in passages]
    assert [p.id for p in route_all(passages, outcomes, c).accepted] == ["b", "a", "c"]  # input order
    assert [p.id for p in route_all(passages, outcomes, c, ranks=[3, 2, 1]).accepted] == ["c", "a", "b"]
    assert [p.id for p in route_all(passages, outcomes, c, ranks=[1, 1, 1]).accepted] == ["a", "b", "c"]


def test_route_all_retrieval_order_and_zero_limit():
    cfg = parse_config(
        "classifier:\n  accept: {order_by: null, limit: 2}\n  conflict: {order_by: contradicts_query_premise, limit: 0}\n"
    ).classifier
    passages = [P("x"), P("y"), P("z"), P("k")]
    table = {"x": S(ans=0.1), "y": S(ans=0.9), "z": S(ans=0.5), "k": S(con=0.9)}
    result = route_all(passages, [_outcome(p, s) for p, s in table.items()], cfg)
    assert [p.id for p in result.accepted] == ["x", "y"]
    assert result.conflicting == [] and result.records[3]["capped"] is True


def test_route_all_rejects_bad_input(c):
    with pytest.raises(ValueError, match="unique"):
        route_all([P("a"), P("a")], [], c)
    with pytest.raises(ValueError, match="ranks"):
        route_all([P("a")], [], c, ranks=[1, 2])
    incomplete = _outcome("a", {"is_relevant": 0.9})
    with pytest.raises(ValueError, match="lack"):
        route_all([P("a")], [incomplete], c)


# --- scoring retries --------------------------------------------------------------------------


class FlakyScorer:
    def __init__(self, fail_times: int, transient: bool = True):
        self.calls = 0
        self.fail_times = fail_times
        self.transient = transient

    def score(self, query, passage):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError("503") if self.transient else ValueError("bad request")
        return PassageScores(scores=S())


def test_transient_errors_are_retried_with_exponential_backoff():
    sleeps: list[float] = []
    ok = FlakyScorer(fail_times=2)
    [outcome] = score_all(ok, "q", [P("a")], sleep=sleeps.append)
    assert outcome.scores is not None and outcome.attempts == 3 and sleeps == [0.5, 1.0]

    bad = FlakyScorer(fail_times=5)
    [outcome] = score_all(bad, "q", [P("a")], sleep=lambda _: None)
    assert outcome.failed and bad.calls == 3 and "TransientError" in outcome.error


def test_attempts_and_backoff_are_configurable():
    sleeps: list[float] = []
    scorer = FlakyScorer(fail_times=9)
    outcome = score_one(scorer, "q", P("a"), attempts=4, sleep=sleeps.append, backoff_seconds=0.1)
    assert scorer.calls == 4 and sleeps == pytest.approx([0.1, 0.2, 0.4])
    assert outcome.attempts == 4


def test_permanent_errors_are_not_retried():
    scorer = FlakyScorer(fail_times=5, transient=False)
    [outcome] = score_all(scorer, "q", [P("a")], sleep=lambda _: None)
    assert outcome.failed and scorer.calls == 1 and "ValueError" in outcome.error


def test_custom_transient_predicate():
    scorer = FlakyScorer(fail_times=1, transient=False)
    outcome = score_one(scorer, "q", P("a"), sleep=lambda _: None, transient=lambda exc: isinstance(exc, ValueError))
    assert outcome.scores is not None and outcome.attempts == 2


def test_score_all_keeps_input_order():
    passages = [P(f"p{i}") for i in range(10)]
    assert [o.passage_id for o in score_all(FlakyScorer(0), "q", passages, max_workers=4)] == [p.id for p in passages]
    assert score_all(FlakyScorer(0), "q", []) == []


# --- TypeSafe scorer (HTTP mocked) ------------------------------------------------------------


def test_typesafe_scorer_sends_state_and_every_question(config):
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        answers = {key: 0.5 + i / 100 for i, key in enumerate(config.classifier.questions)}
        return httpx2.Response(200, json=noul_response(answers))

    scorer = TypeSafePassageScorer.from_config(config, "test-key", transport=typesafe_transport(handler))
    scores = scorer.score("What dose?", P("label-renal-a"))
    body = seen["body"]
    assert body["model"] == "jev-latest"
    assert body["state"] == {"query": "What dose?", "passage": {"id": "label-renal-a", "title": "T label-renal-a", "text": "text label-renal-a"}}
    assert list(body["questions"]) == list(config.classifier.questions)
    assert body["questions"]["is_relevant"] == {
        "type": "noul",
        "instructions": "Is the passage about the subject the query asks about?",
        "criteria": {"true": "It addresses the same subject", "false": "It only shares vocabulary with the query"},
    }
    assert scores["is_relevant"] == pytest.approx(0.50) and scores["answers_query"] == pytest.approx(0.54)
    assert (scores.input_tokens, scores.output_tokens) == (120, 5)
    assert scorer.model == "jev-latest"


def test_custom_questions_and_state_fields():
    cfg = parse_config(
        "typesafe:\n  model: jev-preview\n"
        "classifier:\n  state_fields: [id, text, source, effective_date, missing_key]\n"
        "  questions:\n    on_topic:\n      instructions: Is it on topic?\n      true_means: It is on topic\n"
        "    bare:\n      instructions: Anything else?\n"
        "  rules:\n    - {score: on_topic, when: above, threshold: 0.5, route: accept, reason: on_topic}\n"
        "  fallback: {route: drop, reason: off_topic, decided_by: on_topic}\n"
        "  accept: {order_by: on_topic, limit: 5}\n  conflict: {order_by: null, limit: 0}\n"
    )
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=noul_response({"on_topic": 0.8, "bare": 0.1}))

    passage = Passage(id="h1", text="Leave rules.", metadata={"source": "handbook", "effective_date": date(2025, 4, 1), "note": "x"})
    scores = TypeSafePassageScorer.from_config(cfg, "k", transport=typesafe_transport(handler)).score("q", passage)
    body = seen["body"]
    assert body["model"] == "jev-preview"
    assert body["state"]["passage"] == {"id": "h1", "text": "Leave rules.", "source": "handbook", "effective_date": "2025-04-01"}
    assert body["questions"]["on_topic"]["criteria"] == {"true": "It is on topic"}
    assert "criteria" not in body["questions"]["bare"]
    assert scores.scores == {"on_topic": 0.8, "bare": 0.1}


def test_passage_state_reads_subclass_fields():
    class LabelPassage(Passage):
        source_type: str
        effective_date: date | None = None

    p = LabelPassage(id="x", text="t", title="T", source_type="official_label")
    assert passage_state("q", p, ("id", "source_type", "effective_date")) == {
        "query": "q",
        "passage": {"id": "x", "source_type": "official_label"},
    }


@pytest.mark.parametrize("status, attempts", [(429, 3), (503, 3), (400, 1), (401, 1)])
def test_sdk_errors_are_classified_as_transient_or_permanent(config, status, attempts):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(status, json={"error": {"message": "nope"}})

    scorer = TypeSafePassageScorer.from_config(config, "k", transport=typesafe_transport(handler))
    outcome = score_one(scorer, "q", P("a"), sleep=lambda _: None)
    assert outcome.failed and outcome.attempts == attempts and len(calls) == attempts


def test_build_questions_omits_empty_criteria(config):
    questions = build_questions(config.classifier.questions)
    assert set(questions) == set(config.classifier.questions)
    assert questions["answers_query"].model_dump() == {
        "type": "noul",
        "instructions": "Does the passage supply the specific thing the query asks for?",
        "criteria": {"true": "It gives the direct answer", "false": "It is related but would need other passages to answer"},
    }
