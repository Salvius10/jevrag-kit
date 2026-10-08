"""Answer a question from your own passages with jevrag-kit.

    pip install "jevrag-kit[anthropic]"              # or "jevrag-kit[openai]" for OpenAI-compatible APIs
    set TYPESAFE_API_KEY and the LLM key your configuration names (ANTHROPIC_API_KEY by default)
    python examples/quickstart.py

Retrieval stays in your project: pass the passages your search returned, best first.
"""

from __future__ import annotations

import json
from pathlib import Path

from jevrag_kit import Engine, Passage, load_config

HERE = Path(__file__).parent

# 1. Configuration: the defaults, or your project's YAML file (`jevrag-kit init` writes a starting point).
config = load_config()  # e.g. load_config(HERE / "hr_policy.yaml")

# 2. The engine: TypeSafe scorer and verifier plus the configured LLM. Keys come from the environment.
engine = Engine.from_config(config)

# 3. Your passages, converted from whatever your search returns. Extra values go in metadata.
rows = [json.loads(line) for line in (HERE / "passages.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
passages = [
    Passage(id=r["id"], title=r["title"], text=r["text"], metadata={"department": r["department"], "effective": r["effective"]})
    for r in rows
]

# 4. Run: classify -> gate -> generate -> check -> (regenerate once) -> assemble.
result = engine.run("How many days of unused annual leave can I carry over?", passages)

print(f"{result.answer.status}: {result.answer.text or result.answer.reason}")
for claim in result.answer.claims:
    print(f"  {claim.id} [{claim.passage_id}] confidence {claim.confidence:.2f}: \"{claim.quote}\"")
for item in result.review:  # withheld claims, for a human review queue
    print(f"  withheld {item.claim.id} ({item.review_reason}): {item.claim.text}")

# 5. Keep the trace: every score, route, prompt, draft, and verdict, for audit and threshold tuning
#    (`jevrag-kit sweep --traces traces.jsonl ...` replays them under other settings with no API calls).
with open(HERE / "traces.jsonl", "a", encoding="utf-8") as fh:
    fh.write(json.dumps(result.trace, default=str) + "\n")
