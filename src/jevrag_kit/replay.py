"""Recompute routing and release decisions from stored traces under a different configuration.

Pure functions over trace dicts as Engine.run writes them: no model, database, or network access.
Use them to tune thresholds on a project's own traffic before changing the live configuration.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import yaml

from jevrag_kit.checker.policy import release
from jevrag_kit.classifier.router import decide
from jevrag_kit.config import Block, CheckerConfig, ClassifierConfig, JevConfig
from jevrag_kit.errors import ConfigError
from jevrag_kit.text import normalize


@dataclass
class RouteReplay:
    routes: dict[str, str] = field(default_factory=dict)  # passage_id -> accept | conflict | drop
    reasons: dict[str, str] = field(default_factory=dict)
    included_accept: list[str] = field(default_factory=list)
    included_conflict: list[str] = field(default_factory=list)

    @property
    def supplied(self) -> set[str]:
        return set(self.included_accept) | set(self.included_conflict)

    @property
    def gate_abstains(self) -> bool:
        return not self.included_accept and not self.included_conflict


def _classifier(config: JevConfig | ClassifierConfig) -> ClassifierConfig:
    return config.classifier if isinstance(config, JevConfig) else config


def _checker(config: JevConfig | CheckerConfig) -> CheckerConfig:
    return config.checker if isinstance(config, JevConfig) else config


def _order(stored: Mapping[str, Any], block: Block) -> float:
    return float(stored[block.order_by]) if block.order_by else 0.0


def replay_routes(trace: Mapping[str, Any], config: JevConfig | ClassifierConfig) -> RouteReplay:
    """Route the trace's stored scores again. Passages whose stored scores lack a key the
    configuration needs (or that failed scoring) are dropped as score_failed."""
    c = _classifier(config)
    required = c.required_scores()
    out = RouteReplay()
    ranks = {r["passage_id"]: r["rank"] for r in trace.get("retrieved") or []}
    accepted: list[tuple[float, int, str]] = []
    conflicting: list[tuple[float, int, str]] = []
    for pid in ranks:
        stored = (trace.get("scores") or {}).get(pid) or {}
        if stored.get("error") or not all(k in stored for k in required):
            out.routes[pid], out.reasons[pid] = "drop", "score_failed"
            continue
        d = decide(stored, c)
        out.routes[pid], out.reasons[pid] = d.route, d.reason
        if d.route == "accept":
            accepted.append((-_order(stored, c.accept), ranks[pid], pid))
        elif d.route == "conflict":
            conflicting.append((-_order(stored, c.conflict), ranks[pid], pid))
    out.included_accept = [pid for *_, pid in sorted(accepted)[: c.accept.limit]]
    out.included_conflict = [pid for *_, pid in sorted(conflicting)[: c.conflict.limit]]
    return out


@dataclass
class ReleaseReplay:
    actions: Counter = field(default_factory=Counter)  # ship | review | drop | needs_model
    verdicts: Counter = field(default_factory=Counter)

    @property
    def status(self) -> str:
        if self.actions["ship"] and self.actions["review"]:
            return "partial"
        if self.actions["ship"]:
            return "answered"
        return "abstained"


def replay_release(trace: Mapping[str, Any], config: JevConfig | CheckerConfig) -> ReleaseReplay:
    """Re-apply the release policy to the final round's stored verdicts.

    A claim stored as too_short that a lower min_quote_chars would admit has no stored relation
    result; it is counted as `needs_model` rather than guessed.
    """
    policy = _checker(config)
    out = ReleaseReplay()
    for v in (v for v in trace.get("verdicts") or [] if v.get("final", True)):
        quote = normalize(v["claim"]["quote"])
        if len(quote) < policy.min_quote_chars:
            verdict, action = "fabricated", "drop"
        elif v["locate"] == "too_short":
            out.actions["needs_model"] += 1
            continue
        elif v["locate"] == "missing":
            verdict, action = "fabricated", "drop"
        elif v.get("error") or v.get("relation") is None:
            verdict, action = v["verdict"], "drop"
        else:
            verdict = v["verdict"]
            action, _ = release(verdict, v.get("confidence"), policy)
        out.verdicts[verdict] += 1
        out.actions[action] += 1
    return out


def evaluate(
    traces: Iterable[Mapping[str, Any]],
    config: JevConfig,
    gold_by_query: Mapping[str, set[str]] | None = None,
    planted: Sequence[str] = (),
) -> dict:
    """Aggregate replayed decisions over many traces.

    `prompt_changed` counts traces whose supplied passages would differ from the stored first
    prompt; their claims would need a new generation to evaluate fully. `planted` lists known
    injection passage ids; `gold_by_query` maps a query to the passage ids that should be supplied.
    """
    routes: Counter = Counter()
    actions: Counter = Counter()
    statuses: Counter = Counter()
    gate_abstain = prompt_changed = injection_in_prompt = 0
    gold_hit = gold_total = 0
    for trace in traces:
        r = replay_routes(trace, config)
        routes.update(r.routes.values())
        gate_abstain += r.gate_abstains
        first_prompt = (trace.get("prompt") or [{}])[0]
        stored = set(first_prompt.get("accepted_ids", [])) | set(first_prompt.get("conflicting_ids", []))
        prompt_changed += r.supplied != stored
        injection_in_prompt += any(p in r.supplied for p in planted)
        if gold_by_query and trace.get("query") in gold_by_query:
            gold = gold_by_query[trace["query"]]
            gold_total += len(gold)
            gold_hit += len(gold & r.supplied)
        if r.gate_abstains:
            statuses["abstained"] += 1
            continue
        rel = replay_release(trace, config)
        actions.update(rel.actions)
        statuses[rel.status] += 1
    return {
        "routes": dict(routes),
        "gate_abstain": gate_abstain,
        "prompt_changed": prompt_changed,
        "injection_in_prompt": injection_in_prompt,
        "gold_supplied_share": gold_hit / gold_total if gold_total else None,
        "claim_actions": dict(actions),
        "statuses": dict(statuses),
    }


def parse_grid(items: Iterable[str]) -> dict[str, list[Any]]:
    """`path=v1,v2` strings to {path: [v1, v2]}; values are read as YAML scalars (0.5, 8, review)."""
    grid: dict[str, list[Any]] = {}
    for item in items:
        path, sep, values = item.partition("=")
        if not sep or not path.strip():
            raise ConfigError(f"grid item '{item}' must look like path=v1,v2")
        grid[path.strip()] = [yaml.safe_load(v.strip()) for v in values.split(",") if v.strip()]
    return grid


def sweep(
    traces: Sequence[Mapping[str, Any]],
    config: JevConfig,
    grid: Mapping[str, Sequence[Any]],
    gold_by_query: Mapping[str, set[str]] | None = None,
    planted: Sequence[str] = (),
) -> list[dict]:
    """One row for `config`, then one per combination of grid values (dotted override paths)."""
    version = f"v{config.version}" if isinstance(config.version, int) else str(config.version)
    rows = [{"label": f"config {version}", "params": {}, **evaluate(traces, config, gold_by_query, planted)}]
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        variant = config.with_overrides(params)
        label = " ".join(f"{k}={v}" for k, v in params.items())
        rows.append({"label": label, "params": params, **evaluate(traces, variant, gold_by_query, planted)})
    return rows
