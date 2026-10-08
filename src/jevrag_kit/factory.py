"""Build the live components a configuration describes. Keys come from arguments or the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from jevrag_kit.checker.relation import TypeSafeClaimVerifier
from jevrag_kit.classifier.scorer import TypeSafePassageScorer
from jevrag_kit.config import JevConfig, load_config
from jevrag_kit.errors import MissingCredentials
from jevrag_kit.llm.base import StructuredGenerator

ConfigSource = JevConfig | str | Path | Mapping[str, Any] | None


def resolve_key(env_name: str | None, explicit: str | None = None, environ: Mapping[str, str] | None = None) -> str | None:
    """`explicit`, else the named environment variable. None only when `env_name` is None (no key needed)."""
    if explicit:
        return explicit
    if env_name is None:
        return None
    value = (os.environ if environ is None else environ).get(env_name, "").strip()
    if not value:
        raise MissingCredentials(f"{env_name} is not set; export it, or pass the key explicitly")
    return value


def typesafe_key(config: JevConfig, api_key: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    key = resolve_key(config.typesafe.api_key_env, api_key, environ)
    assert key is not None  # typesafe.api_key_env is never None
    return key


def build_scorer(
    config: ConfigSource = None, *, api_key: str | None = None, environ: Mapping[str, str] | None = None, transport: Any = None
) -> TypeSafePassageScorer:
    cfg = load_config(config)
    return TypeSafePassageScorer.from_config(cfg, typesafe_key(cfg, api_key, environ), transport=transport)


def build_verifier(
    config: ConfigSource = None, *, api_key: str | None = None, environ: Mapping[str, str] | None = None, transport: Any = None
) -> TypeSafeClaimVerifier:
    cfg = load_config(config)
    return TypeSafeClaimVerifier.from_config(cfg, typesafe_key(cfg, api_key, environ), transport=transport)


def build_generator(
    config: ConfigSource = None, *, api_key: str | None = None, environ: Mapping[str, str] | None = None, client: Any = None
) -> StructuredGenerator:
    """The generator for llm.provider: `anthropic` (Messages API) or `openai` (Chat Completions API)."""
    cfg = load_config(config)
    key = resolve_key(cfg.llm.api_key_env, api_key, environ)
    if cfg.llm.provider == "anthropic":
        from jevrag_kit.llm.anthropic_generator import AnthropicGenerator

        return AnthropicGenerator.from_config(cfg, key, client=client)
    from jevrag_kit.llm.openai_generator import OpenAIGenerator

    return OpenAIGenerator.from_config(cfg, key, client=client)


def build_engine(
    config: ConfigSource = None,
    *,
    typesafe_api_key: str | None = None,
    llm_api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
):
    """An Engine with the TypeSafe scorer and verifier and the configured LLM generator."""
    from jevrag_kit.engine import Engine

    return Engine.from_config(config, typesafe_api_key=typesafe_api_key, llm_api_key=llm_api_key, environ=environ)
