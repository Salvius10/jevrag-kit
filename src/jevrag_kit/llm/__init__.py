"""LLM layer: evidence gate, prompt, and structured claims from any LLM.

Built in: AnthropicGenerator (Anthropic Messages API and compatible endpoints) and
OpenAIGenerator (OpenAI Chat Completions API and compatible endpoints). Anything with
`generate(prompt) -> Draft` also works.
"""

from jevrag_kit.llm.anthropic_generator import AnthropicGenerator
from jevrag_kit.llm.base import GenerationError, Generator, Reply, StructuredGenerator
from jevrag_kit.llm.gate import GateAction, evidence_gate
from jevrag_kit.llm.openai_generator import OpenAIGenerator
from jevrag_kit.llm.prompt import Prompt, build_prompt, format_block, format_feedback, format_passage
from jevrag_kit.llm.schema import (
    TOOL_DESCRIPTION,
    TOOL_NAME,
    DraftValidationError,
    anthropic_tool,
    draft_schema,
    openai_tool,
    validate_draft,
)

__all__ = [
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "AnthropicGenerator",
    "DraftValidationError",
    "GateAction",
    "GenerationError",
    "Generator",
    "OpenAIGenerator",
    "Prompt",
    "Reply",
    "StructuredGenerator",
    "anthropic_tool",
    "build_prompt",
    "draft_schema",
    "evidence_gate",
    "format_block",
    "format_feedback",
    "format_passage",
    "openai_tool",
    "validate_draft",
]
