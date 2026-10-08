import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import jevrag_kit.engine
import jevrag_kit.factory
from jevrag_kit import Engine, __version__
from jevrag_kit.cli import load_env_file, main, read_passages, read_traces
from jevrag_kit.config import DEFAULT_CONFIG_FILE
from jevrag_kit.errors import ConfigError
from jevrag_kit.testing import FakeGenerator, FakeScorer, FakeVerifier, grounded_scores, make_claim, make_draft


@pytest.fixture
def isolated_env(monkeypatch):
    """--env-file writes os.environ; give each test its own copy."""
    monkeypatch.setattr(os, "environ", dict(os.environ))


# --- init and check ----------------------------------------------------------------------------


def test_init_copies_the_annotated_defaults(tmp_path: Path, capsys):
    target = tmp_path / "cfg" / "jev.yaml"
    assert main(["init", str(target)]) == 0
    assert target.read_bytes() == DEFAULT_CONFIG_FILE.read_bytes()
    assert main(["init", str(target)]) == 1 and "exists" in capsys.readouterr().err
    assert main(["init", str(target), "--force"]) == 0


def test_check_summarises_a_valid_file(tmp_path: Path, capsys, isolated_env):
    cfg = tmp_path / "jev.yaml"
    cfg.write_text("llm:\n  provider: openai\n  model: kimi-k2\n  api_key_env: MOONSHOT_API_KEY\n", encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("TYPESAFE_API_KEY=abc\n", encoding="utf-8")
    assert main(["check", str(cfg), "--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    assert "is a valid configuration" in out and "rules, first match wins" in out
    assert "TYPESAFE_API_KEY (set)" in out and "MOONSHOT_API_KEY (not set)" in out
    assert "1. contains_prompt_injection above 0.7 -> drop [injection]" in out
    assert "abc" not in out  # key values are never printed


def test_check_reports_errors(tmp_path: Path, capsys):
    cfg = tmp_path / "jev.yaml"
    cfg.write_text("checker:\n  auto_accept: 2\n", encoding="utf-8")
    assert main(["check", str(cfg)]) == 2
    assert "checker.auto_accept" in capsys.readouterr().err


# --- env files, passages, traces --------------------------------------------------------------


def test_load_env_file(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "﻿# comment\n"
        "SQLite example, set DATABASE_URL below\n"
        "Use DATABASE_URL=sqlite\n"
        "export TYPESAFE_API_KEY=ts-key\n"
        'AIML_API_KEY="quoted value"\n'
        "OPENAI_API_KEY=plain # inline comment\n"
        "EMPTY_KEY=\n"
        "ALREADY_SET=from-file\n",
        encoding="utf-8",
    )
    env = {"ALREADY_SET": "from-shell"}
    loaded = load_env_file(path, env)
    assert loaded == ["TYPESAFE_API_KEY", "AIML_API_KEY", "OPENAI_API_KEY", "EMPTY_KEY"]
    assert env == {
        "ALREADY_SET": "from-shell",
        "TYPESAFE_API_KEY": "ts-key",
        "AIML_API_KEY": "quoted value",
        "OPENAI_API_KEY": "plain",
        "EMPTY_KEY": "",
    }
    with pytest.raises(ConfigError, match="cannot read"):
        load_env_file(tmp_path / "missing.env", {})


def test_read_passages(tmp_path: Path):
    jsonl = tmp_path / "p.jsonl"
    jsonl.write_text(
        '{"id": "a", "text": "Alpha.", "title": "A", "source": "handbook"}\n\n'
        '{"id": "b", "text": "Beta.", "metadata": {"year": 2025}}\n',
        encoding="utf-8",
    )
    a, b = read_passages(jsonl)
    assert (a.id, a.title, a.metadata) == ("a", "A", {"source": "handbook"})
    assert (b.title, b.metadata) == ("", {"year": 2025})
    array = tmp_path / "p.json"
    array.write_text('[{"id": "x", "text": "X.", "title": null, "metadata": null}]', encoding="utf-8")
    [x] = read_passages(array)
    assert (x.id, x.title, x.metadata) == ("x", "", {})
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="item 1"):
        read_passages(bad)
    bad.write_text('{"id": "", "text": "t"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="not a valid passage"):
        read_passages(bad)
    bad.write_text("{nope\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        read_passages(bad)


def test_read_traces_accepts_every_layout(tmp_path: Path):
    t1, t2 = {"query": "a"}, {"query": "b"}
    files = {
        "lines.jsonl": json.dumps(t1) + "\n" + json.dumps(t2) + "\n",
        "array.json": json.dumps([t1, t2], indent=2),
        "single.json": json.dumps(t1, indent=2),
        "eval.json": json.dumps({"traces": [t1, t2], "gold": [{"query": "a", "gold_passage_ids": ["p1"]}]}),
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert read_traces([tmp_path / "lines.jsonl"]) == ([t1, t2], None)
    assert read_traces([tmp_path / "array.json"]) == ([t1, t2], None)
    assert read_traces([tmp_path / "single.json"]) == ([t1], None)
    assert read_traces([tmp_path / "eval.json"]) == ([t1, t2], {"a": {"p1"}})
    assert len(read_traces([tmp_path / "lines.jsonl", tmp_path / "single.json"])[0]) == 3


# --- try, sweep, doctor (no network: components are fakes) ------------------------------------


def _fake_engine(config=None):
    claim = make_claim("c1", "Five days carry over.", "leave", "carry over up to five days")
    return Engine(
        FakeScorer({"leave": grounded_scores(), "noise": grounded_scores(rel=0.1)}),
        FakeGenerator([make_draft(claim)]),
        FakeVerifier(),
        config,
    )


def test_try_prints_the_decisions_and_appends_traces(tmp_path: Path, capsys, monkeypatch):
    passages = tmp_path / "p.jsonl"
    passages.write_text(
        '{"id": "leave", "title": "Leave", "text": "Employees may carry over up to five days of leave."}\n'
        '{"id": "noise", "text": "The office closes at six."}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(jevrag_kit.engine.Engine, "from_config", classmethod(lambda cls, cfg, **kw: _fake_engine(cfg)))
    traces = tmp_path / "out" / "traces.jsonl"
    assert main(["try", "--passages", str(passages), "--query", "How much leave carries over?", "--trace-out", str(traces)]) == 0
    out = capsys.readouterr().out
    assert "accept   evidence          leave #1" in out and "drop     not_relevant      noise" in out
    assert "Answer (answered):\n  Five days carry over. [leave]" in out
    [line] = traces.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["status"] == "answered"


def test_try_reports_a_failing_query(tmp_path: Path, capsys, monkeypatch):
    passages = tmp_path / "p.jsonl"
    passages.write_text('{"id": "leave", "text": "Employees may carry over up to five days of leave."}\n', encoding="utf-8")

    class Down:
        def generate(self, prompt):
            raise RuntimeError("401 invalid key")

    broken = Engine(FakeScorer({"leave": grounded_scores()}), Down(), FakeVerifier())
    monkeypatch.setattr(jevrag_kit.engine.Engine, "from_config", classmethod(lambda cls, cfg, **kw: broken))
    traces = tmp_path / "t.jsonl"
    assert main(["try", "--passages", str(passages), "--query", "q", "--trace-out", str(traces)]) == 1
    assert "FAIL: RuntimeError: 401 invalid key" in capsys.readouterr().out
    assert json.loads(traces.read_text(encoding="utf-8"))["status"] == "error"


def test_sweep_replays_stored_traces(tmp_path: Path, capsys):
    engine = _fake_engine()
    from jevrag_kit.testing import make_passage

    trace = engine.run("q", [make_passage("leave", "Employees may carry over up to five days of leave."), make_passage("noise", "x")]).trace
    traces = tmp_path / "t.jsonl"
    traces.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    out_file = tmp_path / "rows.json"
    grid = ["checker.auto_accept=0.8,0.99", "classifier.rules.contains_prompt_injection.threshold=0.7"]
    assert main(["sweep", "--traces", str(traces), "--grid", *grid, "--out", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "1 stored traces" in out and "config v1  " in out
    assert "auto_accept=0.99 contains_prompt_injection=0.7  " in out  # short, untruncated labels
    rows = json.loads(out_file.read_text(encoding="utf-8"))
    assert [r["label"] for r in rows] == [
        "config v1",
        "checker.auto_accept=0.8 classifier.rules.contains_prompt_injection.threshold=0.7",
        "checker.auto_accept=0.99 classifier.rules.contains_prompt_injection.threshold=0.7",
    ]
    assert rows[1]["claim_actions"] == {"ship": 1} and rows[2]["claim_actions"] == {"review": 1}
    assert main(["sweep", "--traces", str(traces), "--grid", "checker.nope=1"]) == 2


def test_doctor_probes_each_service(capsys, monkeypatch, isolated_env):
    claim = make_claim("c1", "Five days.", "sample-leave-carry-over", "carry over up to five days of unused annual leave")
    monkeypatch.setattr(jevrag_kit.factory, "build_scorer", lambda cfg: FakeScorer({}, default=grounded_scores()))
    monkeypatch.setattr(jevrag_kit.factory, "build_verifier", lambda cfg: FakeVerifier())
    monkeypatch.setattr(jevrag_kit.factory, "build_generator", lambda cfg: FakeGenerator([make_draft(claim)]))
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ok   typesafe scoring (jev-latest)" in out and "is_relevant=0.90" in out
    assert "ok   typesafe relation (jev-latest)" in out and "choice=supports" in out
    assert "ok   llm (anthropic:claude-sonnet-5)" in out and "1 claim(s)" in out

    def broken(cfg):
        raise RuntimeError("TYPESAFE_API_KEY is not set")

    monkeypatch.setattr(jevrag_kit.factory, "build_scorer", broken)
    assert main(["doctor", "--skip", "llm", "--skip", "verifier"]) == 1
    out = capsys.readouterr().out
    assert "FAIL typesafe scoring (jev-latest): RuntimeError: TYPESAFE_API_KEY is not set" in out and "llm" not in out.split("\n\n")[-1]


def test_doctor_sample_fits_any_passage_template(tmp_path: Path, capsys, monkeypatch, isolated_env):
    # A project whose template and state read metadata the built-in sample does not have.
    cfg = tmp_path / "jev.yaml"
    cfg.write_text(
        "classifier:\n  state_fields: [id, text, region]\n"
        'llm:\n  prompt:\n    passage_template: "[{id}] {title} ({effective})\\n{text}"\n',
        encoding="utf-8",
    )
    seen = {}

    class Recorder(FakeGenerator):
        def generate(self, prompt):
            seen["prompt"] = prompt
            return super().generate(prompt)

    claim = make_claim("c1", "Five days.", "sample-leave-carry-over", "carry over up to five days of unused annual leave")
    monkeypatch.setattr(jevrag_kit.factory, "build_scorer", lambda c: FakeScorer({}, default=grounded_scores()))
    monkeypatch.setattr(jevrag_kit.factory, "build_verifier", lambda c: FakeVerifier())
    monkeypatch.setattr(jevrag_kit.factory, "build_generator", lambda c: Recorder([make_draft(claim)]))
    assert main(["doctor", "--config", str(cfg)]) == 0
    passage = seen["prompt"].accepted[0]
    assert passage.metadata == {"effective": "sample", "region": "sample"}
    assert "(sample)\nEmployees may carry over" in seen["prompt"].text


def test_module_entry_point_prints_the_version():
    out = subprocess.run([sys.executable, "-m", "jevrag_kit", "--version"], capture_output=True, text=True, check=True).stdout
    assert out.strip() == f"jevrag-kit {__version__}"


def test_load_env_file_is_public_and_reads_dot_env_by_default(tmp_path: Path, monkeypatch, isolated_env):
    from jevrag_kit import load_env_file as public_load_env_file

    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert public_load_env_file() == ["TYPESAFE_API_KEY"] and os.environ["TYPESAFE_API_KEY"] == "from-file"
    assert public_load_env_file(str(tmp_path / ".env")) == []  # already set: the environment wins


def test_cli_explains_a_missing_sdk(tmp_path: Path, capsys, monkeypatch, isolated_env):
    cfg = tmp_path / "jev.yaml"
    cfg.write_text("llm:\n  provider: openai\n  model: gpt-small\n", encoding="utf-8")
    passages = tmp_path / "p.jsonl"
    passages.write_text('{"id": "a", "text": "Alpha."}\n', encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts")
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setitem(sys.modules, "openai", None)  # as if `pip install jevrag-kit` without [openai]
    assert main(["try", "--config", str(cfg), "--passages", str(passages), "--query", "q"]) == 2
    assert 'pip install "jevrag-kit[openai]"' in capsys.readouterr().err
