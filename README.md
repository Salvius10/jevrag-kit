# jevrag-kit

> **New here? Start with [GETTING_STARTED.md](GETTING_STARTED.md)**: install, keys, choosing a model, a first test, and using it from your code, step by step. This README is the reference.

Three reusable processes for answers that are grounded in your documents and checked before release:

- **JEV passage classifier** (`jevrag_kit.classifier`): TypeSafe's JEV model answers yes/no questions about every (query, passage) pair, and ordered rules in code route each passage to *accept*, *conflict*, or *drop*.
- **LLM layer** (`jevrag_kit.llm`): any LLM drafts the answer as structured claims, each citing one passage with a verbatim quote. Anthropic Messages API and OpenAI Chat Completions API are built in, which covers Anthropic, OpenAI, Azure OpenAI, Kimi/Moonshot, DeepSeek, Groq, Gemini, AIML API, Ollama, vLLM, and others. Any other model plugs in through one method.
- **JEV claims checker** (`jevrag_kit.checker`): each quote must be found word for word in the passage it cites, the JEV model judges whether that passage supports the claim, and a release policy in code ships, withholds, or drops it. The answer is assembled from shipped claims only; the LLM's own prose is never shown.

`jevrag_kit.engine` chains them (classify, gate, generate, check, regenerate at most once, assemble) and records a full audit trace. `jevrag_kit.replay` re-runs routing and release decisions on stored traces under new thresholds without any API calls.

Models only score and draft. Every decision is made in code from settings in one YAML file, so a new project configures the package instead of rewriting it. The defaults are Saandru's settings.

```
your retrieval -> passages (best first)
  -> classifier: TypeSafe scores (one request per passage) -> rules -> accept / conflict / drop
  -> evidence gate: nothing accepted or conflicting -> abstain without calling the LLM
  -> LLM: claims with passage ids and verbatim quotes (validated; one corrective retry)
  -> checker: exact quote match -> TypeSafe relation -> ship / review / drop (one regeneration if nothing ships)
  -> answer assembled in code, every sentence cited; trace with every score, route, prompt, draft, verdict
```

## Verified status

Checked on 2026-10-08.

| Check | Result |
|---|---|
| `uv run pytest` | 189 passed on Python 3.10, 3.11, 3.13, and 3.14 (no keys, no network) |
| Lowest supported dependencies | 189 passed with pydantic 2.12.0, PyYAML 6.0.1, typesafe-sdk 0.7.2, anthropic 1.0.0, openai 1.55.3 and 2.0.0 |
| `tools/parity_saandru.py` | 57,186 comparisons against Saandru's `rag/` code, 0 differences: routing, ordering and caps, quote location, release policy, prompts, HTTP request bodies, end-to-end traces under five threshold sets, edge-case pipelines, replay and sweeps. Injecting two small defects into jevrag-kit makes it report 2,469 differences. |
| Fresh project | the built wheel installed into an empty project outside this repo and ran with a different domain's configuration |
| `jevrag-kit doctor` (live) | TypeSafe `jev-latest` scoring and relation; `openai/gpt-4.1-nano` (OpenAI protocol) and `anthropic/claude-haiku-4.5` (Anthropic protocol), both through AIML API |
| `jevrag-kit try` (live, `examples/hr_policy.yaml`) | answered and cited a leave question with the injection passage dropped; abstained on an off-topic question without an LLM call; on a false-premise question the classifier routed the correcting passage to *conflict*, `gpt-4.1-nano` declined to write the correction (so jevrag-kit abstained), and `claude-haiku-4.5` wrote it and it shipped |

## Install

jevrag-kit needs Python 3.10 or later. It is installed from the team's private GitHub repository, not from PyPI. Choose the extra for the API style your LLM speaks:

```bash
pip install "jevrag-kit[openai] @ git+https://github.com/Salvius10/jevrag-kit.git@v0.1.0"      # OpenAI-style APIs
pip install "jevrag-kit[anthropic] @ git+https://github.com/Salvius10/jevrag-kit.git@v0.1.0"   # Anthropic-style APIs
pip install "jevrag-kit[all] @ git+https://github.com/Salvius10/jevrag-kit.git@v0.1.0"         # both
```

You need read access to the repository and git installed. A wheel file (`pip install "jevrag_kit-0.1.0-py3-none-any.whl[openai]"`) or a copy of this folder (`pip install "path/to/jevrag-kit[openai]"`) works too. The core needs only pydantic, PyYAML, and the TypeSafe SDK.

jevrag-kit is not related to the PyPI packages named `jevkit`, `jevguard`, or `jevrag`. A plain `pip install jevrag-kit` finds nothing, because it is not published on PyPI; always install with one of the commands above.

## Fit it to a new project

1. **Write a configuration.** `jevrag-kit init jev.yaml` copies [`src/jevrag_kit/default_config.yaml`](src/jevrag_kit/default_config.yaml), which lists every setting with comments. Keep only the keys you change.
2. **Map your passages.** Convert whatever your search returns into `Passage(id=..., text=..., title=..., metadata={...})`, best first. Values in `metadata` can be sent to TypeSafe (`classifier.state_fields`) and shown to the LLM (`llm.prompt.passage_template`). You can also subclass `Passage` to add typed fields.
3. **Pick the LLM.** Set `llm.provider`, `llm.model`, `llm.base_url`, and `llm.api_key_env` (see [LLM providers](#llm-providers)).
4. **Check it.** `jevrag-kit check jev.yaml` validates every setting and prints the routing rules in order. `jevrag-kit doctor --config jev.yaml --env-file .env` makes one live call to TypeSafe scoring, the relation check, and the LLM.
5. **Try it on your data.** Put a few passages in a JSON Lines file (`{"id", "text", "title"?, "metadata"?}`; other keys become metadata) and run `jevrag-kit try --config jev.yaml --passages passages.jsonl --query "..." --trace-out traces.jsonl`. It prints each passage's route, each claim's verdict, and the released answer.
6. **Tune the thresholds.** `jevrag-kit sweep --config jev.yaml --traces traces.jsonl --grid classifier.rules.is_relevant.threshold=0.35,0.45,0.55 checker.auto_accept=0.8,0.9` replays the stored traces under each combination, with no API calls.
7. **Wire it in.** Call `Engine.run`, store `result.trace`, and send `result.review` (withheld claims) to a review queue if you have one.

[`examples/hr_policy.yaml`](examples/hr_policy.yaml) is a complete configuration for a different domain (HR policy questions, its own questions and rules, Kimi as the LLM), with [`examples/passages.jsonl`](examples/passages.jsonl) to try it on. [`examples/saandru.yaml`](examples/saandru.yaml) is Saandru's configuration.

## Python API

```python
from jevrag_kit import Engine, Passage, load_config, load_env_file

load_env_file(".env")                         # optional: KEY=VALUE lines; real environment variables win
config = load_config("jev.yaml")              # or load_config() for the defaults
engine = Engine.from_config(config)           # keys from the variables the config names
# or pass keys directly: Engine.from_config(config, typesafe_api_key=..., llm_api_key=...)

passages = [Passage(id=h.id, title=h.title, text=h.body, metadata={"source": h.source}) for h in search(query)]
result = engine.run(query, passages)          # optional: ranks=[...], trace={...}, config=<per-run override>

result.answer.status                          # "answered" | "partial" | "abstained"
result.answer.text                            # one cited line per shipped claim, or None
result.answer.reason                          # why it abstained: insufficient_evidence, pending_review, ...
result.answer.claims                          # id, type, text, passage_id, quote, confidence
result.review                                 # withheld claims (ClaimCheck) for human review
result.trace                                  # JSON-serialisable record of every step
```

Each stage also works on its own:

```python
classification = engine.classify(query, passages)          # the classifier as a filter or re-ranker
classification.routing.accepted, classification.routing.records

checks = engine.check(draft, {p.id: p for p in passages})  # check claims drafted elsewhere
```

The functions behind them are public too: `jevrag_kit.classifier.score_all` and `route_all`, `jevrag_kit.llm.build_prompt` and `validate_draft`, `jevrag_kit.checker.verify_draft` and `assemble_answer`.

**Any other LLM.** Write a class with `generate(prompt) -> Draft`. The `prompt` carries the finished text plus the query and passages. [`examples/custom_generator.py`](examples/custom_generator.py) shows one for models without tool calling: it asks for JSON matching `draft_schema()` and validates it with `validate_draft`. Then build the engine with it: `Engine(build_scorer(config), MyGenerator(), build_verifier(config), config)`.

**Tests in your project.** `jevrag_kit.testing` has deterministic fakes (`FakeScorer`, `FakeGenerator`, `FakeVerifier`) and helpers (`make_passage`, `make_claim`, `make_draft`, `grounded_scores`), so your tests need no keys or network.

## Configuration

All settings live in one YAML file, validated strictly: an unknown key, an out-of-range number, or a rule that names a question that doesn't exist fails on load with the exact setting named. A key you set replaces that key; keys you leave out keep their defaults. Mappings (`questions`, `request_params`) and lists (`rules`, `state_fields`) are replaced as a whole. API keys never go in the file. It names environment variables instead.

| Section | What it controls |
|---|---|
| `version` | copied into every trace as `config_version`; bump it on every change |
| `typesafe` | JEV model (`jev-latest`), key variable, endpoint, timeout |
| `classifier.questions` | the yes/no questions; each has `instructions`, optional `true_means` and `false_means` |
| `classifier.rules` | ordered rules: `score`, `when` (`above`, `at_least`, `below`, `at_most`), `threshold`, `route`, `reason`, optional `name`. The first match wins |
| `classifier.fallback` | the route when no rule matches |
| `classifier.accept`, `classifier.conflict` | sort key (`order_by`, highest first, then retrieval rank) and `limit` |
| `classifier.state_fields` | passage fields sent to TypeSafe |
| `classifier.attempts`, `backoff_seconds`, `workers` | retries on transient errors and concurrency |
| `llm` | provider, model, endpoint, key variable, tokens, attempts, tool choice, extra request fields, SDK client options |
| `llm.prompt` | instructions, headings, passage template, feedback template for the one regeneration |
| `checker` | `min_quote_chars`, `auto_accept`, what happens to low-confidence and unsupported claims (`review` or `drop`), relation question wording |
| `answer` | citation template and the withheld-claims sentence |

The keys are `true_means` and `false_means` rather than `true` and `false` because YAML reads unquoted `true:`, `false:`, `yes:` and `no:` keys as booleans. For the same reason, quote string values such as `"Yes"` or `"no"`.

In Python, `config.with_overrides({"classifier.rules.is_relevant.threshold": 0.5, "checker.auto_accept": 0.85})` returns a re-validated copy. Rules are addressed by `name`, or by their score key when they have no name.

### LLM providers

| Service | `provider` | `base_url` | `api_key_env` | Notes |
|---|---|---|---|---|
| Anthropic | `anthropic` | (default) | `ANTHROPIC_API_KEY` | the default |
| AIML API, Claude | `anthropic` | `https://api.aimlapi.com` | `AIML_API_KEY` | key sent as Bearer and x-api-key |
| OpenAI | `openai` | (default) | `OPENAI_API_KEY` | newer reasoning models need `max_tokens_param: max_completion_tokens` |
| AIML API, any chat model | `openai` | `https://api.aimlapi.com/v1` | `AIML_API_KEY` | e.g. `openai/gpt-4.1-nano` |
| Kimi (Moonshot) | `openai` | `https://api.moonshot.ai/v1` | `MOONSHOT_API_KEY` | see `examples/hr_policy.yaml` |
| Azure OpenAI | `openai` | `https://<resource>.openai.azure.com/openai/v1/` | your variable | the v1 API; `model` is the deployment name |
| DeepSeek, Groq, Together, Gemini | `openai` | their OpenAI-compatible URL | your variable | |
| Ollama, vLLM, LM Studio | `openai` | e.g. `http://localhost:11434/v1` | `null` | no key needed |

Live-tested: AIML API with both protocols. The requests sent to Anthropic, AIML API, and an OpenAI-compatible endpoint (Kimi's URL) are also checked against mocked HTTP. The other rows use the same two code paths with a different `base_url`.

- `tool_choice`: `named` (the default) forces the answer tool. Use `required` or `auto` for providers or models that reject a named tool choice. With `auto`, a reply without the tool call is answered with a reminder on the next attempt.
- `request_params` are passed through as extra request-body fields (`temperature`, reasoning settings, and so on). For `anthropic`, the default is `{thinking: {type: disabled}}`, because forced tool use cannot run with extended thinking. Set `request_params: {}` for an endpoint that rejects it.
- `client_options` go to the SDK client, for example `{max_retries: 4, default_headers: {...}}`.

## The trace

`result.trace` uses the layout of Saandru's traces for these fields, so `jevrag-kit sweep` can also replay traces Saandru stored.

| Field | Contents |
|---|---|
| `query`, `config_version`, `retrieved` | the input, the config version, and the passage order (`passage_id`, `rank`); fields your code set beforehand are kept |
| `scores` | per passage: every question's probability, tokens, attempts, and any error |
| `routes` | per passage: route, reason, the deciding score, which side of each rule's threshold it fell on, included, position or capped |
| `prompt`, `draft` | per round: the exact prompt text and blocks, the parsed claims or the generation error with the raw outputs |
| `verdicts` | per claim: quote location, relation, probabilities, confidence, verdict, action, review reason, and `final` for the round that was released |
| `usage` | per stage (`score`, `route`, `generate`, `verify`): model, latency, calls, tokens |
| `status`, `reason`, `answer` | the outcome; on an exception, `status` is `error` and `error` holds the message |

Pass your own dict as `trace=` to add fields (a trace id, user id) and to keep the partial trace when an exception propagates.

## Fixed contracts

Wording, numbers, and order are configurable. These names are not, because the three stages depend on them:

- routes `accept`, `conflict`, `drop`
- claim types `answer`, `premise_correction` (premise corrections are listed first)
- relation labels `supports`, `contradicts`, `says_nothing`
- verdicts `verified`, `unsupported`, `contradicted`, `fabricated`
- actions `ship`, `review`, `drop`

A fabricated claim (its quote isn't in any supplied passage) and a contradicted claim are always dropped. A claim whose relation check errors is dropped: an unverifiable claim never ships.

## Development

```powershell
$env:UV_LINK_MODE = "copy"                               # only in OneDrive folders: they don't support uv's hardlinks
uv sync
uv run pytest                                            # 189 tests, no keys, no network
$env:UV_PROJECT_ENVIRONMENT = ".venv-3.10"; uv run --python 3.10 pytest   # another interpreter, separate environment
uv build                                                 # dist/jevrag_kit-<version>-py3-none-any.whl and .tar.gz
```

To release a version: bump `src/jevrag_kit/_version.py`, add a CHANGELOG entry, commit, then `git tag v<version>` and `git push --tags`. Projects upgrade by changing the `@v<version>` in their install command.

Parity with Saandru, using Saandru's interpreter and environment:

```powershell
$env:PYTHONPATH = "$PWD\src"
..\Saandru\.venv\Scripts\python.exe tools\parity_saandru.py --saandru ..\Saandru
```

It ends with `N comparisons, 0 differences` and exits 0. Run it after any change to the core logic.

## Known limits

- `typesafe-sdk` is pinned to `>=0.7.2,<0.8` because it is pre-1.0. Run the tests and the parity check before raising the bound.
- `jev-latest` is an alias that TypeSafe can move to a new model, which shifts score distributions. Traces record the model per stage; re-run `jevrag-kit sweep` on recent traces after a model change.
- The default thresholds are Saandru's starting values (from the TypeSafe cookbook corpus), not tuned on a labeled set. Tune them on your own traffic with `jevrag-kit sweep`.
- Very small LLMs follow the claim instructions less reliably. In the live check, `gpt-4.1-nano` did not write a premise correction that `claude-haiku-4.5` wrote; jevrag-kit abstained instead of releasing anything unverified.
- Calls are synchronous, with thread pools for scoring and relation checks. There is no async API and no streaming.
- Retrieval, storage, and the review UI are not included; they stay in each project.
- Queries and passages are sent to TypeSafe and to the LLM provider. Confirm their data-handling terms before using confidential material.
