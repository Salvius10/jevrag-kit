"""The README's examples work as written.

Every Python block runs, in document order and in one namespace, the way a reader follows along.
Live services are replaced by deterministic stand-ins, so nothing touches the network. Every YAML
block must be a valid configuration, the JSON Lines example must read as passages, and every
documented command must parse.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

import pytest

import jevrag_kit
from jevrag_kit import Engine, load_config, parse_config
from jevrag_kit.cli import build_parser, read_passages
from jevrag_kit.llm import build_prompt
from jevrag_kit.testing import FakeVerifier
from jevrag_kit.types import Claim, Draft, PassageScores

README = Path(__file__).resolve().parent.parent / "README.md"
TEXT = README.read_text(encoding="utf-8")
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.M | re.S)


def blocks(lang: str) -> list[str]:
    return [m.group(2) for m in FENCE.finditer(TEXT) if m.group(1) == lang]


# --- stand-ins for the live services ----------------------------------------------------------


def _value(rule, match: bool) -> float:
    t = rule.threshold
    if rule.when == "above":
        return min(1.0, t + 0.01) if match else t
    if rule.when == "at_least":
        return t if match else max(0.0, t - 0.01)
    if rule.when == "below":
        return max(0.0, t - 0.01) if match else t
    return t if match else min(1.0, t + 0.01)  # at_most


class DocScorer:
    """Scores that route every passage through the configuration's first accept rule."""

    model = "doc-scorer"

    def __init__(self, config):
        self.config = config

    def score(self, query, passage):
        c = self.config.classifier
        values = {key: 0.5 for key in c.questions}
        for rule in c.rules:
            if rule.route == "accept":
                values[rule.score] = _value(rule, True)
                break
            values[rule.score] = _value(rule, False)
        return PassageScores(scores=values)


class DocGenerator:
    """One claim quoting the start of the first supplied passage."""

    model = "doc-generator"

    def generate(self, prompt):
        cited = (prompt.accepted or prompt.conflicting)[0]
        claim_type = "answer" if prompt.accepted else "premise_correction"
        claim = Claim(id="c1", type=claim_type, text="Doc claim.", passage_id=cited.id, quote=cited.text[:40])
        return Draft(insufficient=False, claims=[claim])


def call_my_model(text: str) -> str:
    """The README's placeholder for "your LLM": answers with JSON quoting the first passage."""
    match = re.search(r"^\[([^\]]+)\] [^\n]*\n(.+)$", text, re.M)
    claims = []
    if match:
        claims = [{"id": "c1", "type": "answer", "text": "Doc claim.", "passage_id": match.group(1), "quote": match.group(2)[:40]}]
    return json.dumps({"insufficient": not claims, "missing": None if claims else "nothing", "claims": claims})


class OtherAPI:
    """The README's placeholder for "the other provider's client" used by its adapter example."""

    GOOD = {"is_relevant": 0.9, "contains_answer_evidence": 0.9, "contradicts_query_premise": 0.05, "contains_prompt_injection": 0.02}

    def __init__(self):
        self.calls: list[str] = []

    def ask(self, query, text, questions):
        self.calls.append("ask")
        return {key: self.GOOD.get(key, 0.5) for key in questions}

    def judge(self, claim, section):
        self.calls.append("judge")
        return "supports", 0.95, {"supports": 0.95, "contradicts": 0.03, "says_nothing": 0.02}


# --- the tests ---------------------------------------------------------------------------------


def test_the_readme_has_examples():
    assert len(blocks("python")) >= 10 and len(blocks("yaml")) >= 8


@pytest.mark.parametrize("doc", ["README.md", "GETTING_STARTED.md", "DEVELOPMENT.md"])
def test_links_point_to_real_sections_and_files(doc):
    path = README.parent / doc
    text = FENCE.sub("", path.read_text(encoding="utf-8"))  # ignore code blocks
    anchors = {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in re.findall(r"^#+ (.+)$", text, re.M)}
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://")):
            continue
        if target.startswith("#"):
            assert target[1:] in anchors, f"{doc}: no section for {target}"
        else:
            assert (path.parent / target.split("#")[0]).exists(), f"{doc}: {target} does not exist"


@pytest.mark.parametrize("text", blocks("yaml"), ids=lambda t: t.strip().splitlines()[0][:40])
def test_yaml_examples_are_valid_configurations(text):
    parse_config(text)


def test_json_lines_example_reads_as_passages(tmp_path):
    [example] = blocks("json")
    path = tmp_path / "passages.jsonl"
    path.write_text(example, encoding="utf-8")
    passages = read_passages(path)
    assert len(passages) == 2 and all(p.text and p.metadata.get("department") for p in passages)


def _commands() -> list[str]:
    found = re.findall(r"`(jevrag-kit [^`]+)`", TEXT)
    for block in blocks("bash"):
        joined = block.replace("\\\n", " ")
        found += [line.strip() for line in joined.splitlines() if line.strip().startswith("jevrag-kit ")]
    return [c for c in found if len(shlex.split(c)) > 2]  # skip bare mentions such as `jevrag-kit try`


@pytest.mark.parametrize("command", _commands())
def test_documented_commands_parse(command):
    build_parser().parse_args(shlex.split(command)[1:])


def test_python_examples_run_in_order(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=doc-typesafe\nOPENAI_API_KEY=doc-openai\n", encoding="utf-8")
    quick_start = next(b for b in blocks("yaml") if b == "llm:\n  provider: openai\n  model: gpt-4.1-mini\n")
    (tmp_path / "jev.yaml").write_text(quick_start, encoding="utf-8")

    def doc_engine(cls, config=None, *, typesafe_api_key=None, llm_api_key=None, environ=None):
        cfg = load_config(config)
        return Engine(DocScorer(cfg), DocGenerator(), FakeVerifier(), cfg)

    monkeypatch.setattr(Engine, "from_config", classmethod(doc_engine))
    monkeypatch.setattr(jevrag_kit, "build_scorer", lambda config, **kw: DocScorer(load_config(config)))
    monkeypatch.setattr(jevrag_kit, "build_verifier", lambda config, **kw: FakeVerifier())
    monkeypatch.setattr(jevrag_kit, "build_generator", lambda config, **kw: DocGenerator())

    other_api = OtherAPI()
    namespace: dict = {"__name__": "readme", "call_my_model": call_my_model, "other_api": other_api}
    for number, code in enumerate(blocks("python"), start=1):
        before = set(namespace)
        exec(compile(code, f"README.md python block {number}", "exec"), namespace)
        for name in sorted(set(namespace) - before):
            if name.startswith("test_") and callable(namespace[name]):
                namespace[name]()  # the README's own test example must pass

    out = capsys.readouterr().out
    assert out.startswith("answered\nDoc claim. [leave-carry-over]\n")  # the quick start's two prints
    assert os.environ["TYPESAFE_API_KEY"] == "doc-typesafe"  # load_env_file(".env") ran
    assert namespace["result"].answer.status == "answered"
    assert (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip()
    draft = namespace["MyModel"]().generate(build_prompt("q", namespace["passages"], []))
    assert [c.passage_id for c in draft.claims] == [namespace["passages"][0].id]

    # The "another provider" adapters answer a question end to end, through both adapters.
    adapted = namespace["engine"].run("How many days of annual leave can I carry over?", namespace["passages"])
    assert adapted.answer.status == "answered"
    assert "ask" in other_api.calls and "judge" in other_api.calls
    assert adapted.trace["usage"]["score"]["model"] == adapted.trace["usage"]["verify"]["model"] == "other-model"
