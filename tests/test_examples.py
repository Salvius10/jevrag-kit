import importlib.util
import json
import py_compile
from pathlib import Path

import pytest

from jevrag_kit import Engine, load_config
from jevrag_kit.cli import read_passages
from jevrag_kit.llm import GenerationError, build_prompt
from jevrag_kit.testing import FakeGenerator, FakeScorer, FakeVerifier, make_claim, make_draft

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_are_valid(path):
    load_config(path)


def test_saandru_example_only_adds_source_type():
    cfg = load_config(EXAMPLES / "saandru.yaml")
    defaults = load_config()
    assert cfg.classifier.state_fields == ("id", "title", "text", "source_type")
    assert cfg.classifier.model_copy(update={"state_fields": defaults.classifier.state_fields}) == defaults.classifier
    assert (cfg.checker, cfg.answer) == (defaults.checker, defaults.answer)
    assert cfg.llm.base_url == "https://api.aimlapi.com" and cfg.llm.api_key_env == "AIML_API_KEY"


def test_hr_example_runs_end_to_end_on_the_example_passages():
    cfg = load_config(EXAMPLES / "hr_policy.yaml")
    passages = read_passages(EXAMPLES / "passages.jsonl")
    assert len(passages) == 7 and passages[0].metadata == {"department": "HR", "effective": "2026-01-01"}

    off_topic = {"is_relevant": 0.1, "states_policy": 0.1, "contradicts_question": 0.05, "contains_prompt_injection": 0.02}
    table = {
        "leave-carry-over": {"is_relevant": 0.97, "states_policy": 0.95, "contradicts_question": 0.05, "contains_prompt_injection": 0.01},
        "leave-accrual": {"is_relevant": 0.8, "states_policy": 0.4, "contradicts_question": 0.05, "contains_prompt_injection": 0.01},
        "chat-export": {"is_relevant": 0.9, "states_policy": 0.9, "contradicts_question": 0.1, "contains_prompt_injection": 0.97},
    }
    claim = make_claim("c1", "Up to five days of unused annual leave carry over.", "leave-carry-over",
                       "carry over up to five days of unused annual leave")
    gen = FakeGenerator([make_draft(claim)])
    engine = Engine(FakeScorer(table, default=off_topic), gen, FakeVerifier(default=("supports", 0.9)), cfg)
    result = engine.run("How many days of annual leave can I carry over?", passages)

    assert result.answer.status == "answered"
    assert result.answer.text == "Up to five days of unused annual leave carry over. [leave-carry-over]"
    routes = {r["passage_id"]: r["reason"] for r in result.trace["routes"]}
    assert routes["chat-export"] == "injection" and routes["leave-accrual"] == "no_policy_rule"
    prompt = gen.prompts[0].text
    assert "Employee question:\nHow many days" in prompt
    assert "[leave-carry-over] Annual leave: Carry-over (effective 2026-01-01)\nEmployees may carry over" in prompt
    assert "unlimited" not in prompt


def test_custom_generator_example():
    spec = importlib.util.spec_from_file_location("custom_generator", EXAMPLES / "custom_generator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    good = {"insufficient": False, "missing": None, "claims": [
        {"id": "c1", "type": "answer", "text": "Five days.", "passage_id": "leave", "quote": "carry over up to five days"}]}
    replies = ["Sure! Here it is.", "```json\n" + json.dumps(good) + "\n```"]
    requests: list[str] = []

    def complete(text):
        requests.append(text)
        return replies.pop(0)

    from jevrag_kit.testing import make_passage

    prompt = build_prompt("q", [make_passage("leave", "Employees may carry over up to five days of leave.")], [])
    draft = mod.JsonModeGenerator(complete).generate(prompt)
    assert draft.attempts == 2 and draft.claims[0].passage_id == "leave"
    assert '"required": ["insufficient", "missing", "claims"]' in requests[0] and "was rejected" in requests[1]

    with pytest.raises(GenerationError):
        mod.JsonModeGenerator(lambda text: "no json here").generate(prompt)

    engine = Engine(FakeScorer({}, default={"is_relevant": 0.9, "contains_answer_evidence": 0.9, "contradicts_query_premise": 0.0,
                                            "contains_prompt_injection": 0.0, "answers_query": 0.9}),
                    mod.JsonModeGenerator(lambda text: json.dumps(good)), FakeVerifier())
    result = engine.run("q", [make_passage("leave", "Employees may carry over up to five days of leave.")])
    assert result.answer.status == "answered" and result.trace["usage"]["generate"]["model"] == "my-model"


def test_quickstart_compiles():
    py_compile.compile(str(EXAMPLES / "quickstart.py"), doraise=True)
