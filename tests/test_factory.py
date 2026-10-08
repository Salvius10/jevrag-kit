import re
import sys

import pytest

from jevrag_kit import parse_config
from jevrag_kit.errors import MissingCredentials, MissingDependency
from jevrag_kit.factory import build_generator, build_scorer, build_verifier, resolve_key
from jevrag_kit.llm import AnthropicGenerator, OpenAIGenerator

ENV = {"TYPESAFE_API_KEY": "ts-key", "ANTHROPIC_API_KEY": "an-key", "AIML_API_KEY": "aiml-key"}


def test_resolve_key_order():
    assert resolve_key("X", "explicit", {"X": "env"}) == "explicit"
    assert resolve_key("X", None, {"X": " env "}) == "env"
    assert resolve_key(None, None, {}) is None
    for env in ({}, {"X": "   "}):
        with pytest.raises(MissingCredentials, match="X is not set"):
            resolve_key("X", None, env)


def test_typesafe_components_follow_the_config():
    cfg = parse_config("typesafe:\n  model: jev-preview\n  api_key_env: TS_KEY\nclassifier:\n  state_fields: [id, text]\n")
    scorer = build_scorer(cfg, environ={"TS_KEY": "k"})
    assert scorer.model == "jev-preview" and scorer.state_fields == ("id", "text")
    assert set(scorer._questions) == set(cfg.classifier.questions)
    assert build_verifier(cfg, environ={"TS_KEY": "k"}).model == "jev-preview"
    with pytest.raises(MissingCredentials, match="TS_KEY"):
        build_scorer(cfg, environ={})


def test_anthropic_generator_auth():
    official = build_generator(None, environ=ENV)
    assert isinstance(official, AnthropicGenerator)
    assert official._client.api_key == "an-key" and official._client.auth_token is None
    assert str(official._client.base_url).startswith("https://api.anthropic.com")

    cfg = parse_config("llm:\n  base_url: https://api.aimlapi.com\n  api_key_env: AIML_API_KEY\n  model: claude-haiku-4-5\n")
    compatible = build_generator(cfg, environ=ENV)
    assert compatible._client.api_key == "aiml-key" and compatible._client.auth_token == "aiml-key"
    assert str(compatible._client.base_url).startswith("https://api.aimlapi.com")
    assert compatible.model == "claude-haiku-4-5"


def test_openai_generator_and_client_options():
    cfg = parse_config(
        "llm:\n  provider: openai\n  model: gpt-small\n  base_url: https://api.aimlapi.com/v1\n  api_key_env: AIML_API_KEY\n"
        "  client_options: {max_retries: 5}\n  timeout: 30\n"
    )
    gen = build_generator(cfg, environ=ENV)
    assert isinstance(gen, OpenAIGenerator)
    assert gen._client.api_key == "aiml-key" and str(gen._client.base_url).startswith("https://api.aimlapi.com/v1")
    assert gen._client.max_retries == 5 and gen._client.timeout == 30


def test_endpoints_without_a_key():
    cfg = parse_config("llm:\n  provider: openai\n  model: llama3.1\n  base_url: http://localhost:11434/v1\n  api_key_env: null\n")
    gen = build_generator(cfg, environ={})
    assert gen._client.api_key == "unused"


def test_missing_llm_key_is_named():
    cfg = parse_config("llm:\n  provider: openai\n  api_key_env: MOONSHOT_API_KEY\n")
    with pytest.raises(MissingCredentials, match="MOONSHOT_API_KEY"):
        build_generator(cfg, environ={})
    assert build_generator(cfg, api_key="explicit", environ={})._client.api_key == "explicit"


@pytest.mark.parametrize("provider, key_env", [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_a_missing_sdk_names_the_install_command(monkeypatch, provider, key_env):
    monkeypatch.setitem(sys.modules, provider, None)  # makes `import <provider>` fail, as if not installed
    cfg = parse_config(f"llm:\n  provider: {provider}\n  model: m\n")
    with pytest.raises(MissingDependency, match=re.escape(f'pip install "jevrag-kit[{provider}]"')) as err:
        build_generator(cfg, environ={key_env: "k"})
    assert isinstance(err.value, ImportError)  # still catchable as an ImportError
