from __future__ import annotations

import os

import pytest

# Tests never make network calls or read real keys. SDKs also read these variables, so clear them.
for _key in (
    "TYPESAFE_API_KEY",
    "TYPESAFE_BASE_URL",
    "TYPESAFE_LOG_LEVEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "AIML_API_KEY",
    "MOONSHOT_API_KEY",
):
    os.environ.pop(_key, None)

from jevrag_kit.config import JevConfig  # noqa: E402


@pytest.fixture
def config() -> JevConfig:
    return JevConfig()
