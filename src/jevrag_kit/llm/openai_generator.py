"""OpenAI Chat Completions API: OpenAI itself, or any compatible endpoint given by base_url
(Azure OpenAI v1, Kimi/Moonshot, DeepSeek, Groq, Together, Gemini's OpenAI endpoint, AIML API,
Ollama, vLLM, LM Studio, ...)."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

from jevrag_kit.errors import MissingDependency
from jevrag_kit.llm.base import Reply, StructuredGenerator
from jevrag_kit.llm.schema import TOOL_DESCRIPTION, TOOL_NAME, openai_tool

if TYPE_CHECKING:
    from jevrag_kit.config import JevConfig, LLMConfig

NO_KEY = "unused"  # placeholder for endpoints that need no key (llm.api_key_env: null)


class OpenAIGenerator(StructuredGenerator):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        client_options: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_tokens: int = 4096,
        max_tokens_param: str = "max_tokens",
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
            request_params=copy.deepcopy(request_params or {}),
        )
        if max_tokens_param not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("max_tokens_param must be 'max_tokens' or 'max_completion_tokens'")
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise MissingDependency(
                    "llm.provider 'openai' needs the OpenAI SDK, which is not installed. "
                    'Install it with: pip install "jevrag-kit[openai]"'
                ) from exc

            options = dict(client_options or {})
            if timeout is not None:
                options["timeout"] = timeout
            client = openai.OpenAI(api_key=api_key, base_url=base_url, **options)
        self._client = client
        self.max_tokens_param = max_tokens_param
        self._tool = openai_tool(tool_name, tool_description)
        choices: dict[str, Any] = {
            "named": {"type": "function", "function": {"name": tool_name}},
            "required": "required",
            "auto": "auto",
        }
        if tool_choice not in choices:
            raise ValueError(f"tool_choice must be one of {sorted(choices)}")
        self._tool_choice = choices[tool_choice]

    @classmethod
    def from_config(cls, config: JevConfig | LLMConfig, api_key: str | None, *, client: Any = None) -> OpenAIGenerator:
        llm = getattr(config, "llm", config)
        return cls(
            llm.model,
            api_key=api_key or NO_KEY,
            base_url=llm.base_url,
            client=client,
            client_options=dict(llm.client_options),
            timeout=llm.timeout,
            max_tokens=llm.max_tokens,
            max_tokens_param=llm.max_tokens_param,
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
            "messages": conversation,
            "tools": [self._tool],
            "tool_choice": self._tool_choice,
            self.max_tokens_param: self.max_tokens,
        }
        if self.request_params:
            kwargs["extra_body"] = copy.deepcopy(self.request_params)
        response = self._client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        message = choices[0].message if choices else None
        calls = list(getattr(message, "tool_calls", None) or [])
        call = next((c for c in calls if getattr(getattr(c, "function", None), "name", None) == self.tool_name), None)
        raw: object = None
        parse_error = None
        if call is not None:
            arguments = call.function.arguments
            if isinstance(arguments, str):
                try:
                    raw = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raw, parse_error = arguments, f"the tool arguments are not valid JSON ({exc})"
            else:
                raw = arguments
        text = getattr(message, "content", None)
        usage = getattr(response, "usage", None)
        return Reply(
            called=call is not None,
            raw=raw,
            parse_error=parse_error,
            has_text=isinstance(text, str) and bool(text.strip()),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            native=(message, call),
        )

    def _reject(self, conversation: list[dict], reply: Reply, message: str) -> list[dict]:
        assistant, call = reply.native
        if not getattr(call, "id", None):  # some local servers omit call ids; answer in a user turn
            return [*conversation, {"role": "user", "content": message}]
        arguments = call.function.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        return [
            *conversation,
            {
                "role": "assistant",
                "content": getattr(assistant, "content", None),
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": arguments}}
                ],
            },
            {"role": "tool", "tool_call_id": call.id, "content": message},
        ]

    def _nudge(self, conversation: list[dict], reply: Reply) -> list[dict]:
        if not reply.has_text:
            return conversation  # nothing to echo back: ask again
        assistant, _ = reply.native
        return [
            *conversation,
            {"role": "assistant", "content": assistant.content},
            {"role": "user", "content": self._reminder()},
        ]
