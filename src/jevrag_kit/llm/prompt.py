"""The prompt: instructions, the query, and two labeled blocks of evidence, built from PromptConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from jevrag_kit.config import PromptConfig
from jevrag_kit.types import Passage

_DEFAULT = PromptConfig()


@dataclass
class Prompt:
    text: str
    accepted_block: str
    conflicting_block: str
    accepted_ids: list[str]
    conflicting_ids: list[str]
    feedback: list[str] | None
    # The inputs, for generators that build their own request; not part of to_dict().
    query: str = ""
    accepted: list[Passage] = field(default_factory=list, repr=False)
    conflicting: list[Passage] = field(default_factory=list, repr=False)

    @property
    def supplied_ids(self) -> set[str]:
        """Every passage id the LLM may cite."""
        return set(self.accepted_ids) | set(self.conflicting_ids)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "accepted_block": self.accepted_block,
            "conflicting_block": self.conflicting_block,
            "accepted_ids": list(self.accepted_ids),
            "conflicting_ids": list(self.conflicting_ids),
            "feedback": list(self.feedback) if self.feedback is not None else None,
        }


class _PassageFields:
    """str.format_map source: declared passage fields and metadata keys; None renders as ''."""

    def __init__(self, passage: Passage):
        self._passage = passage

    def __getitem__(self, key: str) -> Any:
        if not self._passage.has(key):
            raise ValueError(
                f"passage_template uses {{{key}}}, but passage '{self._passage.id}' has no field or metadata key '{key}'"
            )
        value = self._passage.get(key)
        return "" if value is None else value


def format_passage(passage: Passage, config: PromptConfig = _DEFAULT) -> str:
    return config.passage_template.format_map(_PassageFields(passage))


def format_block(passages: Sequence[Passage], config: PromptConfig = _DEFAULT) -> str:
    if not passages:
        return config.empty_block
    return config.passage_separator.join(format_passage(p, config) for p in passages)


def format_feedback(feedback: Sequence[str], config: PromptConfig = _DEFAULT) -> str:
    lines = "\n".join(f"- {item}" for item in feedback)
    return config.feedback_template.replace("{lines}", lines)


def build_prompt(
    query: str,
    accepted: Sequence[Passage],
    conflicting: Sequence[Passage],
    feedback: list[str] | None = None,
    config: PromptConfig | None = None,
) -> Prompt:
    cfg = config or _DEFAULT
    accepted_block = format_block(accepted, cfg)
    conflicting_block = format_block(conflicting, cfg)
    parts = [
        cfg.instructions,
        f"{cfg.query_heading}\n{query}",
        f"{cfg.accepted_heading}\n{accepted_block}",
        f"{cfg.conflicting_heading}\n{conflicting_block}",
    ]
    if feedback:
        parts.append(format_feedback(feedback, cfg))
    return Prompt(
        text=cfg.section_separator.join(parts),
        accepted_block=accepted_block,
        conflicting_block=conflicting_block,
        accepted_ids=[p.id for p in accepted],
        conflicting_ids=[p.id for p in conflicting],
        feedback=feedback,
        query=query,
        accepted=list(accepted),
        conflicting=list(conflicting),
    )
