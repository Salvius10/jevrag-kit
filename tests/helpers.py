"""HTTP-level mocks for the TypeSafe, Anthropic, and OpenAI SDKs (no network).

Anthropic >= 1 and OpenAI >= 3 use httpx2; older versions use httpx. http_lib() returns whichever
module the installed SDK actually uses, so these tests run against old and new SDKs alike.
"""

from __future__ import annotations

import importlib
import json
from types import ModuleType
from typing import Any, Callable

import httpx2

from jevrag_kit.types import Passage


def http_lib(sdk_name: str) -> ModuleType:
    base = importlib.import_module(f"{sdk_name}._base_client")
    return getattr(base, "httpx2", None) or base.httpx


def mock_http_client(sdk_name: str, handler: Callable[[Any], Any]) -> Any:
    http = http_lib(sdk_name)
    return http.Client(transport=http.MockTransport(handler))


def typesafe_transport(handler: Callable[[httpx2.Request], httpx2.Response]) -> httpx2.MockTransport:
    return httpx2.MockTransport(handler)


def noul_response(answers: dict[str, float], input_tokens: int = 120, output_tokens: int = 5) -> dict:
    return {
        "model": "jev-latest",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "answers": {key: {"type": "noul", "noul": value} for key, value in answers.items()},
    }


def choice_response(choice: str, confidence: float, probabilities: dict[str, float]) -> dict:
    return {
        "model": "jev-latest",
        "usage": {"input_tokens": 50, "output_tokens": 3},
        "answers": {
            "relation": {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}
        },
    }


def anthropic_message(tool_input: Any, model: str = "claude-test", tool_name: str = "submit_answer") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "tool_use", "id": "tu_1", "name": tool_name, "input": tool_input}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


def openai_completion(arguments: Any, model: str = "gpt-test", tool_name: str = "submit_answer") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "logprobs": None,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 60, "completion_tokens": 12, "total_tokens": 72},
    }


def raw_draft(**over: Any) -> dict:
    claim = {"id": "c1", "type": "answer", "text": "Reduce to 25 mg.", "passage_id": "a", "quote": "Reduce the dose to 25 mg"}
    return {"insufficient": False, "missing": None, "claims": [claim], **over}


def P(pid: str, text: str = "Reduce the dose to 25 mg once daily.", title: str | None = None, **metadata: Any) -> Passage:
    return Passage(id=pid, text=text, title=f"Label: {pid}" if title is None else title, metadata=metadata)
