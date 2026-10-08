"""The generator interface and the provider-independent structured-output loop.

A generator turns a Prompt into a Draft. The built-in generators force a single tool call so the
LLM returns claims as data; its free prose is never used. Output that fails validation is sent
back once with the reason (llm.max_attempts counts the first try).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from jevrag_kit.errors import JevragKitError
from jevrag_kit.llm.prompt import Prompt
from jevrag_kit.llm.schema import DraftValidationError, validate_draft
from jevrag_kit.types import Draft


class Generator(Protocol):
    """Anything that can draft claims from a prompt. Implement this to plug in any LLM."""

    def generate(self, prompt: Prompt) -> Draft: ...


class GenerationError(JevragKitError, RuntimeError):
    """Every attempt returned output that failed validation."""

    def __init__(self, errors: list[str], raw_outputs: list[object]):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.raw_outputs = raw_outputs


@dataclass
class Reply:
    """One LLM response, reduced to what the loop needs."""

    called: bool  # the answer tool was called
    raw: object  # the tool input (parsed arguments, or the unparsable string); None without a call
    parse_error: str | None
    has_text: bool  # the reply carried text, so it can be echoed back with a reminder
    input_tokens: int
    output_tokens: int
    native: Any  # provider objects needed to continue the conversation


class StructuredGenerator(ABC):
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int,
        max_attempts: int,
        tool_name: str,
        tool_description: str,
        tool_choice: str,
        request_params: dict[str, Any],
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_choice = tool_choice
        self.request_params = dict(request_params)

    def generate(self, prompt: Prompt) -> Draft:
        supplied = prompt.supplied_ids
        conversation = self._start(prompt.text)
        errors: list[str] = []
        raw_outputs: list[object] = []
        input_tokens = output_tokens = 0

        for attempt in range(1, self.max_attempts + 1):
            reply = self._send(conversation)
            input_tokens += reply.input_tokens
            output_tokens += reply.output_tokens
            raw_outputs.append(reply.raw)
            try:
                if not reply.called:
                    raise DraftValidationError(f"the response has no {self.tool_name} tool call")
                if reply.parse_error:
                    raise DraftValidationError(reply.parse_error)
                draft = validate_draft(reply.raw, supplied)
            except DraftValidationError as exc:
                errors.append(f"attempt {attempt}: {exc}")
                if attempt < self.max_attempts:
                    if reply.called:
                        message = f"Rejected: {exc}. Call {self.tool_name} again with corrected output."
                        conversation = self._reject(conversation, reply, message)
                    else:
                        conversation = self._nudge(conversation, reply)
                continue
            return draft.model_copy(
                update={"attempts": attempt, "input_tokens": input_tokens, "output_tokens": output_tokens}
            )
        raise GenerationError(errors, raw_outputs)

    def _reminder(self) -> str:
        return f"Call the {self.tool_name} tool to submit your answer."

    @abstractmethod
    def _start(self, text: str) -> list[dict]:
        """The opening conversation for the prompt text."""

    @abstractmethod
    def _send(self, conversation: list[dict]) -> Reply:
        """One API call."""

    @abstractmethod
    def _reject(self, conversation: list[dict], reply: Reply, message: str) -> list[dict]:
        """The conversation continued with the rejected tool call and the reason."""

    @abstractmethod
    def _nudge(self, conversation: list[dict], reply: Reply) -> list[dict]:
        """The conversation continued after a reply without a tool call."""
