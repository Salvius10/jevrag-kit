"""The three processes chained: classify -> evidence gate -> generate -> check -> (regenerate once) -> assemble.

Models produce scores and drafts; code makes every decision. Retrieval is the caller's job: pass
the candidate passages in retrieval order. Every score, route, prompt, draft, and verdict goes into
the trace, so thresholds can be re-tuned later with jevrag_kit.replay and no API calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jevrag_kit.checker.assemble import abstained, assemble_answer
from jevrag_kit.checker.policy import ClaimCheck, feedback_for, needs_regeneration, verify_draft
from jevrag_kit.checker.relation import ClaimVerifier
from jevrag_kit.classifier.router import RoutingResult, route_all
from jevrag_kit.classifier.scorer import PassageScorer, ScoreOutcome, score_all
from jevrag_kit.config import JevConfig, load_config
from jevrag_kit.llm.base import GenerationError, Generator
from jevrag_kit.llm.gate import evidence_gate
from jevrag_kit.llm.prompt import build_prompt
from jevrag_kit.types import Answer, Draft, Passage


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def model_name(component: object) -> str:
    """The model behind a component, or its class name for custom components and fakes."""
    return getattr(component, "model", None) or type(component).__name__


@dataclass
class Classification:
    outcomes: list[ScoreOutcome]
    routing: RoutingResult


@dataclass
class RunResult:
    answer: Answer
    trace: dict[str, Any]
    routing: RoutingResult | None = None
    outcomes: list[ScoreOutcome] = field(default_factory=list)
    checks: list[ClaimCheck] = field(default_factory=list)  # the final round's checks

    @property
    def review(self) -> list[ClaimCheck]:
        """Claims withheld for human review."""
        return [c for c in self.checks if c.action == "review"]


@dataclass
class _Round:
    round_no: int
    draft: Draft | None
    checks: list[ClaimCheck] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        scorer: PassageScorer,
        generator: Generator,
        verifier: ClaimVerifier,
        config: JevConfig | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.scorer = scorer
        self.generator = generator
        self.verifier = verifier
        self.config = config or JevConfig()
        self._sleep = sleep

    @classmethod
    def from_config(
        cls,
        config: JevConfig | str | Path | Mapping[str, Any] | None = None,
        *,
        typesafe_api_key: str | None = None,
        llm_api_key: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Engine:
        """Build the TypeSafe scorer and verifier and the configured LLM generator.

        Keys come from the arguments, else from the environment variables the config names.
        """
        from jevrag_kit.factory import build_generator, build_scorer, build_verifier

        cfg = load_config(config)
        return cls(
            build_scorer(cfg, api_key=typesafe_api_key, environ=environ),
            build_generator(cfg, api_key=llm_api_key, environ=environ),
            build_verifier(cfg, api_key=typesafe_api_key, environ=environ),
            cfg,
        )

    # --- the stages, usable on their own ---------------------------------------------------

    def classify(
        self,
        query: str,
        passages: Sequence[Passage],
        *,
        ranks: Sequence[int] | None = None,
        config: JevConfig | None = None,
    ) -> Classification:
        """Score and route passages: the passage classifier alone."""
        cfg = config or self.config
        c = cfg.classifier
        passages = list(passages)
        outcomes = score_all(
            self.scorer, query, passages, c.workers, self._sleep, attempts=c.attempts, backoff_seconds=c.backoff_seconds
        )
        return Classification(outcomes, route_all(passages, outcomes, c, ranks))

    def check(
        self, draft: Draft, supplied: Mapping[str, Passage], *, round_no: int = 1, config: JevConfig | None = None
    ) -> list[ClaimCheck]:
        """Verify a draft's claims against the passages it was given: the claims checker alone."""
        cfg = config or self.config
        return verify_draft(draft, supplied, self.verifier, cfg.checker, round_no)

    # --- the full run ------------------------------------------------------------------------

    def run(
        self,
        query: str,
        passages: Sequence[Passage],
        *,
        ranks: Sequence[int] | None = None,
        config: JevConfig | None = None,
        trace: dict[str, Any] | None = None,
    ) -> RunResult:
        """Answer `query` from `passages` (in retrieval order).

        `config` overrides routing, prompt, release, and answer settings for this run; the
        scorer, generator, and verifier (and their models) are fixed when the engine is built.
        Pass your own `trace` dict to have it filled in place, which keeps the partial trace
        when an exception propagates.
        """
        cfg = config or self.config
        passages = list(passages)
        ids = [p.id for p in passages]
        if len(set(ids)) != len(ids):
            raise ValueError("passage ids must be unique")
        if ranks is not None and len(ranks) != len(passages):
            raise ValueError(f"got {len(ranks)} ranks for {len(passages)} passages")
        rank_list = list(ranks) if ranks is not None else list(range(1, len(passages) + 1))

        trace = {} if trace is None else trace
        trace.setdefault("query", query)
        trace.setdefault("config_version", cfg.version)
        trace.setdefault("retrieved", [{"passage_id": p.id, "rank": r} for p, r in zip(passages, rank_list)])
        trace.update({"scores": {}, "routes": [], "prompt": [], "draft": [], "verdicts": []})
        trace.setdefault("usage", {})
        state: dict[str, Any] = {}
        try:
            answer = self._run(query, passages, rank_list, cfg, trace, state)
        except Exception as exc:
            trace["status"] = "error"
            trace["error"] = f"{type(exc).__name__}: {exc}"
            raise
        trace["status"] = answer.status
        trace["reason"] = answer.reason
        trace["answer"] = answer.model_dump(mode="json")
        return RunResult(answer, trace, state.get("routing"), state.get("outcomes", []), state.get("checks", []))

    def _run(
        self, query: str, passages: list[Passage], ranks: list[int], cfg: JevConfig, trace: dict, state: dict
    ) -> Answer:
        usage = trace["usage"]
        c = cfg.classifier

        # 1. passage scoring, one request per passage
        started = time.perf_counter()
        outcomes = score_all(
            self.scorer, query, passages, c.workers, self._sleep, attempts=c.attempts, backoff_seconds=c.backoff_seconds
        )
        state["outcomes"] = outcomes
        trace["scores"] = {
            o.passage_id: {
                **(dict(o.scores.scores) if o.scores else {}),
                "input_tokens": o.scores.input_tokens if o.scores else 0,
                "output_tokens": o.scores.output_tokens if o.scores else 0,
                "attempts": o.attempts,
                "latency_ms": o.latency_ms,
                "error": o.error,
            }
            for o in outcomes
        }
        usage["score"] = {
            "model": model_name(self.scorer),
            "latency_ms": _ms(started),
            "requests": sum(o.attempts for o in outcomes),
            "failed": sum(1 for o in outcomes if o.failed),
            "input_tokens": sum(o.scores.input_tokens for o in outcomes if o.scores),
            "output_tokens": sum(o.scores.output_tokens for o in outcomes if o.scores),
        }

        # 2. routing in code
        started = time.perf_counter()
        routing = route_all(passages, outcomes, c, ranks)
        state["routing"] = routing
        trace["routes"] = routing.records
        usage["route"] = {"latency_ms": _ms(started)}

        # 3. evidence gate
        usage["generate"] = {"model": model_name(self.generator), "calls": 0, "latency_ms": 0, "input_tokens": 0, "output_tokens": 0}
        usage["verify"] = {"model": model_name(self.verifier), "calls": 0, "latency_ms": 0, "input_tokens": 0, "output_tokens": 0}
        if evidence_gate(routing.accepted, routing.conflicting) == "abstain":
            return abstained("insufficient_evidence")

        supplied: dict[str, Passage] = {p.id: p for p in [*routing.accepted, *routing.conflicting]}

        # 4-5. generation and verification, with at most one regeneration
        first = self._round(1, query, routing, None, supplied, trace, cfg)
        if first.draft is None:
            return abstained("generation_failed")
        final = first
        if needs_regeneration(first.checks):
            second = self._round(2, query, routing, feedback_for(first.checks), supplied, trace, cfg)
            if second.draft is not None:
                final = second

        # 6. the answer, assembled in code from shipped claims
        state["checks"] = final.checks
        answer = assemble_answer(final.checks, final.draft, cfg.answer)
        for verdict in trace["verdicts"]:
            verdict["final"] = verdict["round"] == final.round_no
        return answer

    def _round(
        self,
        round_no: int,
        query: str,
        routing: RoutingResult,
        feedback: list[str] | None,
        supplied: dict[str, Passage],
        trace: dict,
        cfg: JevConfig,
    ) -> _Round:
        usage = trace["usage"]
        prompt = build_prompt(query, routing.accepted, routing.conflicting, feedback, cfg.llm.prompt)
        trace["prompt"].append({"round": round_no, **prompt.to_dict()})

        started = time.perf_counter()
        usage["generate"]["calls"] += 1
        try:
            draft = self.generator.generate(prompt)
        except GenerationError as exc:
            usage["generate"]["latency_ms"] += _ms(started)
            trace["draft"].append({"round": round_no, "error": str(exc), "raw_outputs": exc.raw_outputs})
            return _Round(round_no, None)
        usage["generate"]["latency_ms"] += _ms(started)
        usage["generate"]["input_tokens"] += draft.input_tokens
        usage["generate"]["output_tokens"] += draft.output_tokens
        trace["draft"].append(
            {
                "round": round_no,
                "insufficient": draft.insufficient,
                "missing": draft.missing,
                "claims": [c.model_dump() for c in draft.claims],
                "attempts": draft.attempts,
                "raw": draft.raw,
            }
        )

        started = time.perf_counter()
        checks = verify_draft(draft, supplied, self.verifier, cfg.checker, round_no)
        usage["verify"]["latency_ms"] += _ms(started)
        usage["verify"]["calls"] += sum(1 for c in checks if c.relation is not None or c.error)
        usage["verify"]["input_tokens"] += sum(c.input_tokens for c in checks)
        usage["verify"]["output_tokens"] += sum(c.output_tokens for c in checks)
        trace["verdicts"].extend(c.to_dict() for c in checks)
        return _Round(round_no, draft, checks)
