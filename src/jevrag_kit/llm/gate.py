from __future__ import annotations

from typing import Literal, Sequence

from jevrag_kit.types import Passage

GateAction = Literal["abstain", "correct_premise", "generate"]


def evidence_gate(accepted: Sequence[Passage], conflicting: Sequence[Passage]) -> GateAction:
    """Abstain without calling the LLM when there is nothing to answer from."""
    if not accepted and not conflicting:
        return "abstain"
    if not accepted:
        return "correct_premise"
    return "generate"
