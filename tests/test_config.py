from pathlib import Path

import pytest
from pydantic import ValidationError

from jevrag_kit.config import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_INSTRUCTIONS,
    JevConfig,
    LLMConfig,
    load_config,
    parse_config,
)
from jevrag_kit.errors import ConfigError

# --- defaults and files ---------------------------------------------------------------------


def test_annotated_default_file_equals_the_built_in_defaults():
    assert load_config(DEFAULT_CONFIG_FILE) == JevConfig()


def test_defaults_are_the_proven_saandru_settings(config):
    c = config.classifier
    assert config.typesafe.model == "jev-latest" and config.classifier_model == "jev-latest"
    assert list(c.questions) == [
        "is_relevant",
        "contains_answer_evidence",
        "contradicts_query_premise",
        "contains_prompt_injection",
        "answers_query",
    ]
    assert [(r.score, r.when, r.threshold, r.route, r.reason) for r in c.rules] == [
        ("contains_prompt_injection", "above", 0.70, "drop", "injection"),
        ("contradicts_query_premise", "above", 0.70, "conflict", "premise_conflict"),
        ("is_relevant", "below", 0.45, "drop", "not_relevant"),
        ("contains_answer_evidence", "above", 0.55, "accept", "evidence"),
    ]
    assert (c.fallback.route, c.fallback.reason, c.fallback.decided_by) == ("drop", "no_evidence", "contains_answer_evidence")
    assert (c.accept.order_by, c.accept.limit, c.conflict.order_by, c.conflict.limit) == (
        "answers_query", 8, "contradicts_query_premise", 4,
    )
    assert (config.checker.min_quote_chars, config.checker.auto_accept) == (20, 0.90)
    assert config.llm.provider == "anthropic" and config.llm.request_params == {"thinking": {"type": "disabled"}}
    assert config.llm.prompt.instructions == DEFAULT_INSTRUCTIONS


@pytest.mark.parametrize(
    "cfg",
    [
        JevConfig(),
        parse_config(
            "llm:\n  provider: openai\n  model: kimi-k2\n  base_url: https://api.moonshot.ai/v1\n"
            "  request_params: {temperature: 0.3}\n  prompt:\n    instructions: |-\n      Line one.\n\n      Line three.\n"
            "classifier:\n  state_fields: [id, text, source]\n"
        ),
    ],
)
def test_to_yaml_round_trips(cfg):
    assert parse_config(cfg.to_yaml()) == cfg


def test_load_config_sources(tmp_path: Path):
    assert load_config(None) == JevConfig()
    cfg = JevConfig(version=7)
    assert load_config(cfg) is cfg
    assert load_config({"version": 3}).version == 3
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty) == JevConfig()
    bom = tmp_path / "bom.yaml"
    bom.write_bytes("﻿version: 5\n".encode("utf-8"))  # Windows editors may add a BOM
    assert load_config(bom).version == 5
    assert load_config(str(bom)).version == 5


def test_load_config_errors_are_config_errors(tmp_path: Path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("classifier: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        parse_config("- a\n- b\n")


def test_partial_file_keeps_every_other_default():
    cfg = parse_config("classifier:\n  workers: 8\nchecker:\n  auto_accept: 0.8\n")
    assert cfg.classifier.workers == 8 and cfg.checker.auto_accept == 0.8
    defaults = JevConfig()
    assert cfg.classifier.questions == defaults.classifier.questions
    assert cfg.classifier.rules == defaults.classifier.rules
    assert cfg.llm == defaults.llm


# --- validation ----------------------------------------------------------------------------


def test_unknown_keys_are_rejected_with_their_location():
    with pytest.raises(ConfigError, match="classifer"):
        parse_config("classifer:\n  workers: 2\n")
    with pytest.raises(ConfigError, match=r"classifier\.wokers"):
        parse_config("classifier:\n  wokers: 2\n")


def test_error_message_names_the_exact_setting():
    with pytest.raises(ConfigError) as err:
        parse_config(
            "classifier:\n  rules:\n    - {score: is_relevant, when: above, threshold: 1.5, route: accept, reason: r}\n"
        )
    assert "classifier.rules[0].threshold" in str(err.value)


def test_rules_must_use_defined_questions():
    yaml_text = "classifier:\n  questions:\n    on_topic:\n      instructions: Is it on topic?\n"
    with pytest.raises(ConfigError, match=r"rules\[0\]\.score uses 'contains_prompt_injection'"):
        parse_config(yaml_text)


def test_fallback_and_order_by_must_use_defined_questions():
    base = (
        "classifier:\n  questions:\n    on_topic:\n      instructions: Is it on topic?\n"
        "  rules:\n    - {score: on_topic, when: above, threshold: 0.5, route: accept, reason: on_topic}\n"
    )
    with pytest.raises(ConfigError, match="fallback.decided_by uses 'contains_answer_evidence'"):
        parse_config(base)
    with pytest.raises(ConfigError, match="accept.order_by uses 'answers_query'"):
        parse_config(base + "  fallback: {route: drop, reason: off_topic, decided_by: on_topic}\n")
    ok = parse_config(
        base
        + "  fallback: {route: drop, reason: off_topic, decided_by: on_topic}\n"
        + "  accept: {order_by: on_topic, limit: 5}\n  conflict: {order_by: null, limit: 0}\n"
    )
    assert ok.classifier.required_scores() == {"on_topic"}


def test_block_needs_both_settings():
    with pytest.raises(ConfigError, match=r"classifier\.accept\.order_by"):
        parse_config("classifier:\n  accept: {limit: 3}\n")


def test_duplicate_rule_names_need_a_name():
    rules = (
        "classifier:\n  rules:\n"
        "    - {score: is_relevant, when: below, threshold: 0.2, route: drop, reason: off_topic}\n"
        "    - {score: is_relevant, when: above, threshold: 0.9, route: accept, reason: on_topic}\n"
    )
    with pytest.raises(ConfigError, match="used twice"):
        parse_config(rules)
    named = rules.replace("reason: on_topic}", "reason: on_topic, name: strongly_relevant}")
    cfg = parse_config(named)
    assert [r.key for r in cfg.classifier.rules] == ["is_relevant", "strongly_relevant"]


def test_question_keys_must_be_simple_names():
    with pytest.raises(ConfigError, match="question key 'is relevant'"):
        parse_config("classifier:\n  questions:\n    is relevant:\n      instructions: x\n")


def test_unquoted_yaml_true_false_keys_are_caught():
    # YAML reads unquoted true/false keys as booleans, which is why the fields are true_means/false_means.
    with pytest.raises(ConfigError):
        parse_config("classifier:\n  questions:\n    is_relevant:\n      instructions: x\n      true: yes it is\n")


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        ("checker:\n  auto_accept: 1.2\n", "checker.auto_accept"),
        ("checker:\n  low_confidence_action: ship\n", "checker.low_confidence_action"),
        ("llm:\n  provider: gemini\n", "llm.provider"),
        ("llm:\n  tool_choice: forced\n", "llm.tool_choice"),
        ("llm:\n  tool_name: submit answer\n", "llm.tool_name"),
        ("llm:\n  max_attempts: 0\n", "llm.max_attempts"),
        ("classifier:\n  state_fields: []\n", "classifier.state_fields"),
        ("classifier:\n  state_fields: [id, id]\n", "classifier.state_fields"),
        ("classifier:\n  attempts: 0\n", "classifier.attempts"),
        ("llm:\n  api_key_env: '  '\n", "llm.api_key_env"),
    ],
)
def test_out_of_range_and_unknown_values(yaml_text, message):
    with pytest.raises(ConfigError, match=message.replace(".", r"\.")):
        parse_config(yaml_text)


def test_provider_dependent_defaults():
    assert LLMConfig().api_key_env == "ANTHROPIC_API_KEY"
    openai_cfg = LLMConfig(provider="openai")
    assert openai_cfg.api_key_env == "OPENAI_API_KEY" and openai_cfg.request_params == {}
    explicit = parse_config("llm:\n  provider: openai\n  api_key_env: MOONSHOT_API_KEY\n").llm
    assert explicit.api_key_env == "MOONSHOT_API_KEY"
    local = parse_config("llm:\n  provider: openai\n  base_url: http://localhost:11434/v1\n  api_key_env: null\n").llm
    assert local.api_key_env is None
    no_thinking = parse_config("llm:\n  request_params: {}\n").llm
    assert no_thinking.provider == "anthropic" and no_thinking.request_params == {}


def test_request_params_and_client_options_guard_reserved_keys():
    assert LLMConfig(request_params={"temperature": 0}).request_params == {"temperature": 0}
    for key in ("model", "messages", "tools", "tool_choice", "max_tokens", "max_completion_tokens", "stream"):
        with pytest.raises(ValidationError, match="is set by jevrag-kit"):
            LLMConfig(request_params={key: 1})
    assert LLMConfig(client_options={"max_retries": 4}).client_options == {"max_retries": 4}
    for key in ("api_key", "auth_token", "base_url", "timeout"):
        with pytest.raises(ValidationError, match="does not belong here"):
            LLMConfig(client_options={key: "x"})


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        ("llm:\n  prompt:\n    passage_template: '{title}: {text}'\n", "must include {id}"),
        ("llm:\n  prompt:\n    passage_template: '[{id] {text}'\n", "passage_template"),
        ("llm:\n  prompt:\n    passage_template: '[{}] {text}'\n", "named fields"),
        ("llm:\n  prompt:\n    feedback_template: Try again.\n", "must include {lines}"),
        ("answer:\n  citation_template: '{text} ({source})'\n", "unknown fields"),
        ("answer:\n  withheld_many: '{count} withheld'\n", "only {n}"),
    ],
)
def test_templates_are_checked_on_load(yaml_text, message):
    with pytest.raises(ConfigError, match=message.replace("{", r"\{").replace("}", r"\}").replace("(", r"\(")):
        parse_config(yaml_text)


def test_models_inherit_from_typesafe_unless_set():
    cfg = parse_config("typesafe:\n  model: jev-preview\nchecker:\n  model: jev-latest\n")
    assert cfg.classifier_model == "jev-preview" and cfg.checker_model == "jev-latest"


def test_config_is_immutable(config):
    with pytest.raises(ValidationError):
        config.version = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.checker.auto_accept = 0.5  # type: ignore[misc]


# --- overrides -----------------------------------------------------------------------------


def test_with_overrides_by_dotted_path(config):
    tuned = config.with_overrides(
        {
            "classifier.rules.contains_prompt_injection.threshold": 0.8,
            "classifier.accept.limit": 3,
            "checker.auto_accept": 0.85,
            "version": 2,
        }
    )
    assert tuned.classifier.rules[0].threshold == 0.8 and tuned.classifier.rules[1].threshold == 0.70
    assert (tuned.classifier.accept.limit, tuned.checker.auto_accept, tuned.version) == (3, 0.85, 2)
    assert config.classifier.rules[0].threshold == 0.70  # the original is untouched


def test_with_overrides_finds_named_rules():
    cfg = parse_config(
        "classifier:\n  rules:\n"
        "    - {score: is_relevant, when: below, threshold: 0.2, route: drop, reason: off_topic, name: floor}\n"
        "    - {score: contains_answer_evidence, when: above, threshold: 0.5, route: accept, reason: evidence}\n"
    )
    assert cfg.with_overrides({"classifier.rules.floor.threshold": 0.3}).classifier.rules[0].threshold == 0.3


@pytest.mark.parametrize(
    "path, value, message",
    [
        ("checker.auto_acept", 0.8, "unknown setting 'auto_acept'"),
        ("classifier.rules.no_such_rule.threshold", 0.8, "no rule named 'no_such_rule'"),
        ("classifier.rules.is_relevant", 0.8, "name a setting of the rule"),
        ("checker.auto_accept", 1.5, "checker.auto_accept"),
    ],
)
def test_with_overrides_errors(config, path, value, message):
    with pytest.raises(ConfigError, match=message.replace(".", r"\.")):
        config.with_overrides({path: value})
