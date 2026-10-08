"""Plug any LLM into jevrag-kit by implementing one method: generate(prompt) -> Draft.

The built-in generators cover the Anthropic Messages API and the OpenAI Chat Completions API
(and every service compatible with either). This pattern is for anything else, for example a
model without tool calling: ask for JSON that matches the answer schema, then validate it.
The classifier, the claims checker, and the release policy work exactly as with the built-ins.
"""

from __future__ import annotations

import json
from typing import Callable

from jevrag_kit.llm import DraftValidationError, GenerationError, Prompt, draft_schema, validate_draft
from jevrag_kit.types import Draft


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


class JsonModeGenerator:
    """`complete` is your LLM call: prompt text in, reply text out."""

    def __init__(self, complete: Callable[[str], str], model: str = "my-model", max_attempts: int = 2):
        self.complete = complete
        self.model = model  # recorded in traces as usage.generate.model
        self.max_attempts = max_attempts

    def generate(self, prompt: Prompt) -> Draft:
        request = (
            prompt.text
            + "\n\nReply with only a JSON object that matches this JSON schema:\n"
            + json.dumps(draft_schema())
        )
        errors: list[str] = []
        raw_outputs: list[object] = []
        for attempt in range(1, self.max_attempts + 1):
            reply = self.complete(request)
            raw_outputs.append(reply)
            try:
                draft = validate_draft(json.loads(_strip_fences(reply)), prompt.supplied_ids)
            except (json.JSONDecodeError, DraftValidationError) as exc:
                errors.append(f"attempt {attempt}: {exc}")
                request += f"\n\nYour previous reply was rejected: {exc}. Reply again with corrected JSON only."
                continue
            return draft.model_copy(update={"attempts": attempt})
        raise GenerationError(errors, raw_outputs)


if __name__ == "__main__":  # pragma: no cover - needs a TypeSafe key and your LLM
    from jevrag_kit import Engine, load_config
    from jevrag_kit.factory import build_scorer, build_verifier

    def complete(text: str) -> str:
        raise NotImplementedError("call your LLM here and return its reply text")

    config = load_config()
    engine = Engine(build_scorer(config), JsonModeGenerator(complete), build_verifier(config), config)
