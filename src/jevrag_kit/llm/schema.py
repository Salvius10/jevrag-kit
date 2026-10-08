"""The answer tool's JSON schema, its two wire formats, and validation of what the LLM returns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from jevrag_kit.types import CLAIM_TYPES, Claim, Draft

TOOL_NAME = "submit_answer"
TOOL_DESCRIPTION = "Submit the answer as a list of claims, each citing one supplied passage with a verbatim quote."


def draft_schema() -> dict:
    """A fresh copy of the JSON schema for the answer tool's input."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["insufficient", "missing", "claims"],
        "properties": {
            "insufficient": {
                "type": "boolean",
                "description": "True if the passages do not contain the answer.",
            },
            "missing": {
                "type": ["string", "null"],
                "description": "What information is missing, when insufficient is true.",
            },
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "type", "text", "passage_id", "quote"],
                    "properties": {
                        "id": {"type": "string", "description": "c1, c2, ..."},
                        "type": {"type": "string", "enum": list(CLAIM_TYPES)},
                        "text": {"type": "string", "description": "one self-contained sentence"},
                        "passage_id": {"type": "string", "description": "id of a supplied passage"},
                        "quote": {"type": "string", "description": "verbatim span from that passage"},
                    },
                },
            },
        },
    }


def anthropic_tool(name: str = TOOL_NAME, description: str = TOOL_DESCRIPTION) -> dict:
    """The tool definition for the Anthropic Messages API."""
    return {"name": name, "description": description, "input_schema": draft_schema()}


def openai_tool(name: str = TOOL_NAME, description: str = TOOL_DESCRIPTION) -> dict:
    """The tool definition for the OpenAI Chat Completions API."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": draft_schema()}}


class _ClaimOut(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    type: Literal["answer", "premise_correction"]
    text: str
    passage_id: str
    quote: str


class _DraftOut(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    insufficient: bool
    missing: str | None = None
    claims: list[_ClaimOut]


class DraftValidationError(ValueError):
    pass


def validate_draft(raw: object, supplied_ids: set[str]) -> Draft:
    """Schema check plus: every cited passage_id was supplied, claim ids are unique, no empty claims."""
    try:
        parsed = _DraftOut.model_validate(raw)
    except ValidationError as exc:
        raise DraftValidationError(f"output does not match the schema: {exc.errors(include_url=False)}") from exc
    unknown = sorted({c.passage_id for c in parsed.claims} - supplied_ids)
    if unknown:
        raise DraftValidationError(f"claims cite passage ids that were not supplied: {unknown}")
    ids = [c.id for c in parsed.claims]
    if len(ids) != len(set(ids)):
        raise DraftValidationError("claim ids must be unique")
    if any(not c.text.strip() or not c.quote.strip() for c in parsed.claims):
        raise DraftValidationError("claims need non-empty text and quote")
    return Draft(
        insufficient=parsed.insufficient,
        missing=parsed.missing,
        claims=[Claim(**c.model_dump()) for c in parsed.claims],
        raw=raw if isinstance(raw, dict) else None,
    )
