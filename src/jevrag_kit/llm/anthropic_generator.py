"""Anthropic Messages API: Anthropic itself, or any compatible endpoint given by base_url
(AIML API, Kimi's Anthropic-compatible endpoint, LiteLLM, ...)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from jevrag_kit.errors import MissingDependency
from jevrag_kit.llm.base import Reply, StructuredGenerator
from jevrag_kit.llm.schema import TOOL_DESCRIPTION, TOOL_NAME, anthropic_tool

if TYPE_CHECKING:
    from jevrag_kit.config import JevConfig, LLMConfig

# Forced tool choice cannot run with extended thinking, and some Claude models think by default.
DEFAULT_REQUEST_PARAMS: dict[str, Any] = {"thinking": {"type": "disabled"}}
NO_KEY = "unused"  # placeholder for endpoints that need no key (llm.api_key_env: null)


def _block_dict(block: Any) -> Any:
    return block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block


class AnthropicGenerator(StructuredGenerator):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        auth_token: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        client_options: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_tokens: int = 4096,
        max_attempts: int = 2,
        tool_name: str = TOOL_NAME,
        tool_description: str = TOOL_DESCRIPTION,
        tool_choice: str = "named",
        request_params: dict[str, Any] | None = None,
    ):
        super().__init__(
            model,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_choice=tool_choice,
            request_params=copy.deepcopy(DEFAULT_REQUEST_PARAMS if request_params is None else request_params),
        )
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise MissingDependency(
                    "llm.provider 'anthropic' needs the Anthropic SDK, which is not installed. "
                    'Install it with: pip install "jevrag-kit[anthropic]"'
                ) from exc

            options = dict(client_options or {})
            if timeout is not None:
                options["timeout"] = timeout
            client = anthropic.Anthropic(api_key=api_key, auth_token=auth_token, base_url=base_url, **options)
        self._client = client
        self._tool = anthropic_tool(tool_name, tool_description)
        choices = {"named": {"type": "tool", "name": tool_name}, "required": {"type": "any"}, "auto": {"type": "auto"}}
        if tool_choice not in choices:
            raise ValueError(f"tool_choice must be one of {sorted(choices)}")
        self._tool_choice = choices[tool_choice]

    @classmethod
    def from_config(cls, config: JevConfig | LLMConfig, api_key: str | None, *, client: Any = None) -> AnthropicGenerator:
        llm = getattr(config, "llm", config)
        key = api_key or NO_KEY
        return cls(
            llm.model,
            api_key=key,
            # Compatible endpoints (AIML API and others) document Bearer auth; Anthropic uses x-api-key.
            # Sending the key in both also stops the SDK from falling back to ANTHROPIC_API_KEY.
            auth_token=key if llm.base_url and api_key else None,
            base_url=llm.base_url,
            client=client,
            client_options=dict(llm.client_options),
            timeout=llm.timeout,
            max_tokens=llm.max_tokens,
            max_attempts=llm.max_attempts,
            tool_name=llm.tool_name,
            tool_description=llm.tool_description,
            tool_choice=llm.tool_choice,
            request_params=dict(llm.request_params),
        )

    def _start(self, text: str) -> list[dict]:
        return [{"role": "user", "content": text}]

    def _send(self, conversation: list[dict]) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "tools": [self._tool],
            "tool_choice": self._tool_choice,
            "messages": conversation,
        }
        if self.request_params:
            kwargs["extra_body"] = copy.deepcopy(self.request_params)
        message = self._client.messages.create(**kwargs)
        content = list(getattr(message, "content", None) or [])
        block = next((b for b in content if getattr(b, "type", None) == "tool_use"), None)
        has_text = any(getattr(b, "type", None) == "text" and (getattr(b, "text", "") or "").strip() for b in content)
        usage = getattr(message, "usage", None)
        return Reply(
            called=block is not None,
            raw=getattr(block, "input", None),
            parse_error=None,
            has_text=has_text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            native=(content, block),
        )

    def _reject(self, conversation: list[dict], reply: Reply, message: str) -> list[dict]:
        content, block = reply.native
        return [
            *conversation,
            {"role": "assistant", "content": [_block_dict(b) for b in content]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": block.id, "is_error": True, "content": message}],
            },
        ]

    def _nudge(self, conversation: list[dict], reply: Reply) -> list[dict]:
        if not reply.has_text:
            return conversation  # nothing to echo back: ask again
        content, _ = reply.native
        return [
            *conversation,
            {"role": "assistant", "content": [_block_dict(b) for b in content]},
            {"role": "user", "content": self._reminder()},
        ]
