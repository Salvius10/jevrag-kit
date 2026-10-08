"""jevrag-kit command line.

    jevrag-kit init [jev.yaml]                       write the default configuration, with comments
    jevrag-kit check jev.yaml                        validate a configuration and summarise it
    jevrag-kit doctor --config jev.yaml              one live call to TypeSafe scoring, relation, and the LLM
    jevrag-kit try --config jev.yaml --passages p.jsonl --query "..."   answer live from your passages
    jevrag-kit sweep --config jev.yaml --traces t.jsonl --grid checker.auto_accept=0.8,0.9   replay, no API calls

`--env-file` loads KEY=VALUE lines into the environment (variables already set win). Key values
are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from jevrag_kit._version import __version__
from jevrag_kit.config import DEFAULT_CONFIG_FILE, JevConfig, load_config, template_fields
from jevrag_kit.envfile import load_env_file
from jevrag_kit.errors import ConfigError, JevragKitError, MissingCredentials, MissingDependency
from jevrag_kit.types import Passage

SAMPLE_PASSAGE = Passage(
    id="sample-leave-carry-over",
    title="Leave policy: Carry-over",
    text=(
        "Employees may carry over up to five days of unused annual leave into the next calendar year. "
        "Carried-over days expire on 31 March."
    ),
)
SAMPLE_QUERY = "How many days of unused annual leave can be carried over?"
SAMPLE_CLAIM = "Up to five days of unused annual leave can be carried over into the next year."


def sample_passage(cfg: JevConfig) -> Passage:
    """The doctor's sample passage, with a placeholder for every field the configuration's
    passage_template and state_fields read that the sample itself does not have."""
    needed = template_fields(cfg.llm.prompt.passage_template) | set(cfg.classifier.state_fields)
    missing = sorted(name for name in needed if not SAMPLE_PASSAGE.has(name))
    return SAMPLE_PASSAGE.model_copy(update={"metadata": {name: "sample" for name in missing}})


def read_passages(path: Path) -> list[Passage]:
    """Passages from JSON Lines (one object per line) or a JSON array. Each object needs `id` and
    `text`; `title` and `metadata` are optional, and any other keys are added to metadata."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the passages file ({exc.strerror or exc})") from exc
    stripped = text.lstrip()
    try:
        if stripped.startswith("["):
            items = [(i + 1, obj) for i, obj in enumerate(json.loads(text))]
        else:
            items = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), start=1) if line.strip()]
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: not valid JSON or JSON Lines: {exc}") from exc
    passages = []
    for n, obj in items:
        if not isinstance(obj, dict) or "id" not in obj or "text" not in obj:
            raise ConfigError(f"{path}: item {n} must be an object with at least `id` and `text`")
        known = {k: obj[k] for k in ("id", "text", "title") if obj.get(k) is not None}
        metadata = dict(obj.get("metadata") or {})
        metadata.update({k: v for k, v in obj.items() if k not in ("id", "text", "title", "metadata")})
        try:
            passages.append(Passage(**known, metadata=metadata))
        except ValueError as exc:
            raise ConfigError(f"{path}: item {n} is not a valid passage: {exc}") from exc
    return passages


def read_traces(paths: list[Path]) -> tuple[list[dict], dict[str, set[str]] | None]:
    """Traces from JSON Lines, a JSON array, or an object with `traces` (and optional `gold`,
    a list of {query, gold_passage_ids}), as Saandru's eval/run_eval.py --out writes."""
    traces: list[dict] = []
    gold: dict[str, set[str]] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ConfigError(f"{path}: cannot read the traces file ({exc.strerror or exc})") from exc
        try:
            document: Any = json.loads(text)  # one JSON document
        except json.JSONDecodeError:
            try:  # JSON Lines
                document = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{path}: not valid JSON or JSON Lines: {exc}") from exc
        if isinstance(document, list):
            payload: dict = {"traces": document}
        elif isinstance(document, dict):
            payload = document if "traces" in document else {"traces": [document]}
        else:
            raise ConfigError(f"{path}: expected trace objects, got {type(document).__name__}")
        traces.extend(payload["traces"])
        for item in payload.get("gold") or []:
            gold[item["query"]] = set(item.get("gold_passage_ids") or [])
    return traces, gold or None


def _key_status(name: str | None) -> str:
    if name is None:
        return "no key"
    return f"{name} ({'set' if os.environ.get(name, '').strip() else 'not set'})"


def describe(cfg: JevConfig) -> list[str]:
    c, llm, chk = cfg.classifier, cfg.llm, cfg.checker
    lines = [
        f"version     {cfg.version}",
        f"typesafe    model={cfg.typesafe.model} key={_key_status(cfg.typesafe.api_key_env)} "
        f"base_url={cfg.typesafe.base_url or 'default'}",
        f"classifier  model={cfg.classifier_model} state_fields=[{', '.join(c.state_fields)}] "
        f"questions=[{', '.join(c.questions)}]",
        "  rules, first match wins:",
    ]
    for i, rule in enumerate(c.rules, start=1):
        label = f" ({rule.name})" if rule.name else ""
        lines.append(f"    {i}. {rule.score} {rule.when} {rule.threshold:g} -> {rule.route} [{rule.reason}]{label}")
    lines.append(f"    else -> {c.fallback.route} [{c.fallback.reason}]")
    for name, block in (("accept", c.accept), ("conflict", c.conflict)):
        lines.append(f"  {name}: order by {block.order_by or 'retrieval rank'}, keep {block.limit}")
    lines += [
        f"llm         provider={llm.provider} model={llm.model} base_url={llm.base_url or 'default'} "
        f"key={_key_status(llm.api_key_env)} tool_choice={llm.tool_choice}",
        f"            request_params={json.dumps(llm.request_params)} max_tokens={llm.max_tokens}",
        f"checker     model={cfg.checker_model} min_quote_chars={chk.min_quote_chars} auto_accept={chk.auto_accept:g} "
        f"low_confidence->{chk.low_confidence_action} unsupported->{chk.unsupported_action}",
    ]
    return lines


# --- commands ------------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    if args.path.exists() and not args.force:
        print(f"error: {args.path} exists; use --force to overwrite", file=sys.stderr)
        return 1
    args.path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_CONFIG_FILE, args.path)
    print(f"wrote {args.path}; edit it, then run: jevrag-kit check {args.path}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    _load_env(args)
    cfg = load_config(args.config)
    print(f"ok: {args.config} is a valid configuration")
    print("\n".join(describe(cfg)))
    return 0


def _probe(name: str, fn: Callable[[], str]) -> bool:
    started = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure and keep going
        first = str(exc).splitlines()[0] if str(exc) else ""
        print(f"FAIL {name}: {type(exc).__name__}: {first[:300]}")
        return False
    print(f"ok   {name} ({int((time.perf_counter() - started) * 1000)} ms): {detail}")
    return True


def cmd_doctor(args: argparse.Namespace) -> int:
    from jevrag_kit.factory import build_generator, build_scorer, build_verifier
    from jevrag_kit.llm.prompt import build_prompt

    _load_env(args)
    cfg = load_config(args.config)
    print(f"jevrag-kit {__version__}")
    print("\n".join(describe(cfg)))
    print()

    passage = sample_passage(cfg)

    def scoring() -> str:
        scores = build_scorer(cfg).score(SAMPLE_QUERY, passage)
        return ", ".join(f"{k}={v:.2f}" for k, v in scores.scores.items())

    def relation() -> str:
        r = build_verifier(cfg).relation(SAMPLE_CLAIM, passage.text)
        return f"choice={r.choice} confidence={r.confidence:.2f}"

    def generation() -> str:
        prompt = build_prompt(SAMPLE_QUERY, [passage], [], None, cfg.llm.prompt)
        draft = build_generator(cfg).generate(prompt)
        return (
            f"{len(draft.claims)} claim(s), insufficient={draft.insufficient}, attempts={draft.attempts}, "
            f"tokens={draft.input_tokens}/{draft.output_tokens}"
        )

    results = []
    if "scorer" not in args.skip:
        results.append(_probe(f"typesafe scoring ({cfg.classifier_model})", scoring))
    if "verifier" not in args.skip:
        results.append(_probe(f"typesafe relation ({cfg.checker_model})", relation))
    if "llm" not in args.skip:
        results.append(_probe(f"llm ({cfg.llm.provider}:{cfg.llm.model})", generation))
    return 0 if all(results) else 1


def print_result(query: str, result: Any) -> None:
    trace = result.trace
    print(f"\nQuery: {query}")
    print("Routes:")
    for r in trace["routes"]:
        extra = f" #{r['position']}" if r.get("position") else (" (capped)" if r.get("capped") else "")
        print(f"  {r['route']:<9}{r['reason']:<18}{r['passage_id']}{extra}")
    for v in trace["verdicts"]:
        conf = f"{v['confidence']:.2f}" if v["confidence"] is not None else "  - "
        mark = "" if v.get("final", True) else " (superseded round)"
        print(
            f"  round {v['round']} {v['claim']['id']:<4}{v['action']:<7}{v['verdict']:<13}{conf}  "
            f"{v['locate']:<13}[{v['claim']['passage_id']}] {v['claim']['text']}{mark}"
        )
    answer = result.answer
    print(f"Answer ({answer.status}{', ' + answer.reason if answer.reason else ''}):")
    for line in (answer.text or "").splitlines() or ["(nothing released)"]:
        print(f"  {line}")
    if answer.missing:
        print(f"  missing: {answer.missing}")


def cmd_try(args: argparse.Namespace) -> int:
    from jevrag_kit.engine import Engine

    _load_env(args)
    cfg = load_config(args.config)
    passages = read_passages(args.passages)
    engine = Engine.from_config(cfg)
    print(f"{len(passages)} passages from {args.passages}; llm {cfg.llm.provider}:{cfg.llm.model}")
    failures = 0
    for query in args.query:
        trace: dict[str, Any] = {}
        try:
            result = engine.run(query, passages, trace=trace)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next query
            failures += 1
            print(f"\nQuery: {query}\nFAIL: {type(exc).__name__}: {exc}")
        else:
            print_result(query, result)
        if args.trace_out:
            args.trace_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.trace_out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(trace, default=str) + "\n")
    if args.trace_out:
        print(f"\nappended {len(args.query)} trace(s) to {args.trace_out}")
    return 1 if failures else 0


def short_path(path: str) -> str:
    """A compact display name: a rule's name for its threshold, else the path without its section."""
    parts = path.split(".")
    if len(parts) == 4 and parts[:2] == ["classifier", "rules"] and parts[3] == "threshold":
        return parts[2]
    if len(parts) > 1 and parts[0] in ("classifier", "checker", "llm", "answer", "typesafe"):
        return ".".join(parts[1:])
    return path


def print_rows(rows: list[dict]) -> None:
    labels = [
        " ".join(f"{short_path(k)}={v}" for k, v in r["params"].items()) if r["params"] else r["label"] for r in rows
    ]
    width = max(len("configuration"), *(len(label) for label in labels)) + 2
    print(
        f"{'configuration':<{width}}{'acc':>5}{'con':>5}{'drop':>6}{'gate':>6}{'chg':>5}{'inj':>5}{'gold':>6}"
        f"{'ship':>6}{'rev':>5}{'drop':>6}  statuses"
    )
    for label, r in zip(labels, rows):
        routes, actions = r["routes"], r["claim_actions"]
        gold = "n/a" if r["gold_supplied_share"] is None else f"{r['gold_supplied_share']:.0%}"
        print(
            f"{label:<{width}}{routes.get('accept', 0):>5}{routes.get('conflict', 0):>5}{routes.get('drop', 0):>6}"
            f"{r['gate_abstain']:>6}{r['prompt_changed']:>5}{r['injection_in_prompt']:>5}{gold:>6}"
            f"{actions.get('ship', 0):>6}{actions.get('review', 0):>5}{actions.get('drop', 0):>6}  {r['statuses']}"
        )


def cmd_sweep(args: argparse.Namespace) -> int:
    from jevrag_kit.replay import parse_grid, sweep

    cfg = load_config(args.config)
    traces, gold = read_traces(args.traces)
    print(f"{len(traces)} stored traces\n")
    rows = sweep(traces, cfg, parse_grid(args.grid), gold, args.planted)
    print_rows(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


def _load_env(args: argparse.Namespace) -> None:
    if getattr(args, "env_file", None):
        load_env_file(args.env_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevrag-kit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"jevrag-kit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write the default configuration, with comments")
    p.add_argument("path", nargs="?", default=Path("jev.yaml"), type=Path)
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("check", help="validate a configuration and summarise it")
    p.add_argument("config", type=Path)
    p.add_argument("--env-file", type=Path, help="report whether the keys it sets are present")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("doctor", help="one live call to each service (costs a few requests)")
    p.add_argument("--config", type=Path, help="default: the built-in defaults")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--skip", action="append", default=[], choices=["scorer", "verifier", "llm"])
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("try", help="answer queries live from passages in a JSON Lines file")
    p.add_argument("--config", type=Path, help="default: the built-in defaults")
    p.add_argument("--passages", type=Path, required=True, help="JSON Lines: {id, text, title?, metadata?}")
    p.add_argument("--query", action="append", required=True, help="repeat for several queries")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--trace-out", type=Path, help="append each trace as one JSON line (input for sweep)")
    p.set_defaults(func=cmd_try)

    p = sub.add_parser("sweep", help="replay stored traces under other settings (no API calls)")
    p.add_argument("--config", type=Path, help="default: the built-in defaults")
    p.add_argument("--traces", type=Path, nargs="+", required=True)
    p.add_argument("--grid", nargs="*", default=[], help="dotted.path=v1,v2 ... e.g. checker.auto_accept=0.8,0.9")
    p.add_argument("--planted", nargs="*", default=[], help="ids of known injection passages")
    p.add_argument("--out", type=Path, help="write the rows as JSON")
    p.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, MissingCredentials, MissingDependency) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except JevragKitError as exc:  # pragma: no cover - other library errors
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
