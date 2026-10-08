# Getting started with jevrag-kit

This guide takes you from a laptop that has never seen jevrag-kit to grounded, quote-checked answers in your own project. Steps 1 to 5 take about 15 minutes. The [README](README.md) is the full reference.

## What jevrag-kit does, and what you bring

jevrag-kit takes a question and the passages your search found for it, and returns an answer in which every sentence is backed by an exact quote from one of those passages. When it can't back a sentence, it leaves the sentence out, or abstains.

```
your search (any kind) ─► passages ─► jevrag-kit ─► answer with citations, or an explained abstention
```

You bring four things:

| You need | Where it comes from |
|---|---|
| Python 3.10 or later | [python.org](https://www.python.org/downloads/). Check with `python --version` |
| A TypeSafe API key | your TypeSafe account at [typesafe.ai](https://typesafe.ai). The JEV models score passages and check claims |
| An LLM | an API key for the provider you choose (Anthropic, OpenAI, Kimi, AIML API, Azure, ...), or a local model with no key |
| Your passages | **your own search.** jevrag-kit does not index or search documents. Your project finds the candidate passages (a search engine, a vector database, SQL, or a plain list) and hands them over |

## 1. Install

Create a virtual environment for your project first.

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Then install jevrag-kit from the team's GitHub repository, with the extra for the API style your LLM speaks:

```bash
pip install "jevrag-kit[openai] @ git+https://github.com/Salvius10/jevrag-kit.git@v0.1.0"
```

With uv, use `uv add "jevrag-kit[openai] @ git+https://github.com/Salvius10/jevrag-kit.git@v0.1.0"` instead.

| Extra | Use it for |
|---|---|
| `[openai]` | the OpenAI Chat Completions API: OpenAI, Azure OpenAI, Kimi/Moonshot, DeepSeek, Groq, Together, Gemini's OpenAI endpoint, AIML API, Ollama, vLLM, LM Studio |
| `[anthropic]` | the Anthropic Messages API: Anthropic, AIML API's Claude endpoint |
| `[all]` | both |

The repository is private. You need git installed, read access to the repository (ask its owner), and to sign in to GitHub the first time git asks.

- On Windows, Git Credential Manager opens a browser window to sign in.
- On macOS and Linux, use a [personal access token](https://github.com/settings/tokens) as the password, or install with SSH: `pip install "jevrag-kit[openai] @ git+ssh://git@github.com/Salvius10/jevrag-kit.git@v0.1.0"`.

Other ways to install, if you were given the files instead:

```bash
pip install "jevrag_kit-0.1.0-py3-none-any.whl[openai]"   # a wheel file someone sent you
pip install "/path/to/jevrag-kit[openai]"                   # a copy of the project folder
```

`jevrag-kit` is not on PyPI, so a plain `pip install jevrag-kit` finds nothing. Always use one of the commands above.

Check the install:

```bash
jevrag-kit --version
```

If your shell says the command is not found, the virtual environment isn't active. `python -m jevrag_kit --version` always works.

## 2. Create your configuration

```bash
jevrag-kit init jev.yaml
```

This writes a file with every setting and a comment on each. Every key is optional: delete what you don't change, and the default applies. The smallest working configuration is an empty file. That gives you the defaults, with Anthropic's `claude-sonnet-5` as the LLM.

## 3. Choose your LLM

Put an `llm:` section in `jev.yaml`. Pick one of these and change the model name if you like. The model must support tool or function calling.

**OpenAI**
```yaml
llm:
  provider: openai
  model: gpt-4.1-mini
```

**Anthropic (Claude)**
```yaml
llm:
  provider: anthropic
  model: claude-sonnet-5
```

**Kimi (Moonshot)**
```yaml
llm:
  provider: openai
  model: kimi-k2-turbo-preview
  base_url: https://api.moonshot.ai/v1
  api_key_env: MOONSHOT_API_KEY
```

**AIML API: any of its chat models, through its OpenAI-style endpoint**
```yaml
llm:
  provider: openai
  model: openai/gpt-4.1-nano
  base_url: https://api.aimlapi.com/v1
  api_key_env: AIML_API_KEY
```

**AIML API: Claude models, through its Anthropic-style endpoint**
```yaml
llm:
  provider: anthropic
  model: anthropic/claude-haiku-4.5
  base_url: https://api.aimlapi.com
  api_key_env: AIML_API_KEY
```

**Azure OpenAI (the v1 API)**
```yaml
llm:
  provider: openai
  model: my-gpt-deployment              # your deployment name
  base_url: https://YOUR-RESOURCE.openai.azure.com/openai/v1/
  api_key_env: AZURE_OPENAI_API_KEY
```

**A local model with Ollama, vLLM, or LM Studio (no key)**
```yaml
llm:
  provider: openai
  model: llama3.1
  base_url: http://localhost:11434/v1
  api_key_env: null
```

AIML API was tested live with both styles. The other entries use the same two code paths with a different `base_url`, and were tested against simulated responses.

Any other OpenAI-compatible service works the same way: set `provider: openai`, its `base_url`, its model name, and the environment variable that holds its key. For a model with no compatible API at all, see [Any other LLM](#any-other-llm) below.

The TypeSafe model is set separately. The default is `jev-latest`; change it with:

```yaml
typesafe:
  model: jev-latest
```

## 4. Plug in your keys

Keys never go in `jev.yaml`. The file only *names* the environment variables that hold them:

| Setting | Default variable |
|---|---|
| `typesafe.api_key_env` | `TYPESAFE_API_KEY` |
| `llm.api_key_env` | `ANTHROPIC_API_KEY` for `provider: anthropic`, `OPENAI_API_KEY` for `provider: openai`. Set your own name, as in the Kimi and AIML examples, or `null` for no key |

Give the keys to jevrag-kit in any one of three ways.

**A. A `.env` file (simplest).** Create a file named `.env` next to your code:

```
TYPESAFE_API_KEY=ts-...
OPENAI_API_KEY=sk-...
```

The command-line tool reads it with `--env-file .env`. In Python, call `load_env_file()` before building the engine (see step 7). Add `.env` to your `.gitignore` so keys never reach git.

**B. Environment variables.**

```powershell
# Windows, this terminal only
$env:TYPESAFE_API_KEY = "ts-..."
$env:OPENAI_API_KEY = "sk-..."
# Windows, permanently (applies to terminals opened afterwards)
setx TYPESAFE_API_KEY "ts-..."
```

```bash
# macOS / Linux (add the lines to ~/.zshrc or ~/.bashrc to keep them)
export TYPESAFE_API_KEY="ts-..."
export OPENAI_API_KEY="sk-..."
```

A real environment variable always wins over a value in `.env`.

**C. In code**, for example from your own secrets manager:

```python
engine = Engine.from_config(config, typesafe_api_key=my_secrets["typesafe"], llm_api_key=my_secrets["openai"])
```

Check what jevrag-kit sees. It prints whether each key is set, never the key itself:

```bash
jevrag-kit check jev.yaml --env-file .env
```

## 5. Test the connection

```bash
jevrag-kit doctor --config jev.yaml --env-file .env
```

This makes one small live call to each service: TypeSafe scoring, the TypeSafe relation check, and your LLM. Expect three `ok` lines:

```
ok   typesafe scoring (jev-latest) (1098 ms): is_relevant=0.98, ...
ok   typesafe relation (jev-latest) (409 ms): choice=supports confidence=1.00
ok   llm (openai:openai/gpt-4.1-nano) (3405 ms): 1 claim(s), insufficient=False, attempts=1, tokens=365/75
```

A `FAIL` line names the problem; see [Troubleshooting](#troubleshooting).

## 6. Try it on your own passages

Write a few of your passages to a JSON Lines file, one per line. Only `id` and `text` are required. Any other key (`title`, `source`, `date`, ...) is kept as metadata.

```json
{"id": "leave-carry-over", "title": "Annual leave: Carry-over", "text": "Employees may carry over up to five days of unused annual leave into the next calendar year.", "department": "HR"}
{"id": "sick-leave", "title": "Sick leave", "text": "Sick leave is separate from annual leave and does not carry over.", "department": "HR"}
```

Then ask questions:

```bash
jevrag-kit try --config jev.yaml --env-file .env --passages passages.jsonl --query "How many days of leave can I carry over?" --trace-out traces.jsonl
```

For each passage, it prints whether it was accepted, used as a correction, or dropped, and why. Then it prints each claim's check and the final answer.

## 7. Use it from your code

```python
from jevrag_kit import Engine, Passage, load_config, load_env_file

load_env_file(".env")                      # or set the variables another way (step 4)
config = load_config("jev.yaml")
engine = Engine.from_config(config)        # build once, reuse for every question


def answer(question: str) -> dict:
    hits = my_search(question)             # YOUR search: best match first
    passages = [
        Passage(id=h.id, title=h.title, text=h.text, metadata={"source": h.source})
        for h in hits
    ]
    result = engine.run(question, passages)

    save_trace(result.trace)               # JSON-serialisable: keep it for audit and tuning
    for item in result.review:             # claims that were withheld for a human to check
        send_to_review_queue(item.claim.text, item.claim.passage_id, item.review_reason)

    return {
        "status": result.answer.status,    # answered | partial | abstained
        "text": result.answer.text,        # one cited line per verified claim, or None
        "reason": result.answer.reason,    # why it abstained, e.g. insufficient_evidence
        "sources": result.answer.source_ids,
    }
```

`my_search`, `save_trace`, and `send_to_review_queue` stand for your own code. Passage ids must be unique within one call.

Each stage also works alone. `engine.classify(question, passages)` scores and routes passages, so you can use it as a filter in front of an existing RAG system. `engine.check(draft, passages_by_id)` checks claims your own LLM wrote.

### Any other LLM

If your model speaks neither API style, write a class with one method, `generate(prompt) -> Draft`, and pass it in place of the built-in generator. [`examples/custom_generator.py`](examples/custom_generator.py) is a complete example for models without tool calling.

```python
from jevrag_kit import Engine, build_scorer, build_verifier

engine = Engine(build_scorer(config), MyGenerator(), build_verifier(config), config)
```

## 8. Adapt it to your domain

The defaults were built for questions over official documents. To fit another domain, edit these parts of `jev.yaml`; the comments in the file explain each one:

- **`classifier.questions`**: the yes/no questions TypeSafe answers about each passage. Write them in your domain's words.
- **`classifier.rules`**: the ordered rules that decide each passage's route from those answers. The first match wins.
- **`llm.prompt`**: the instructions, headings, and how each passage is shown to the LLM. `passage_template` can show metadata, for example `"[{id}] {title} ({department})\n{text}"`.
- **`checker`**: how confident a claim must be to ship (`auto_accept`), and whether weak claims are withheld for review or dropped.

[`examples/hr_policy.yaml`](examples/hr_policy.yaml) is a complete example for HR policy questions. Run `jevrag-kit check jev.yaml` after every edit: it reports any mistake and the exact setting it is in.

One YAML trap: unquoted `yes`, `no`, `true`, `false`, `on`, and `off` are read as true/false. Quote them when you mean the words.

## 9. Tune the thresholds on your own data

Collect traces with `--trace-out` (step 6) or `save_trace` (step 7). Then compare settings without any API calls:

```bash
jevrag-kit sweep --config jev.yaml --traces traces.jsonl --grid checker.auto_accept=0.8,0.9 classifier.rules.is_relevant.threshold=0.35,0.45
```

Each row shows how many passages each setting would accept or drop, and how many claims would ship. When you change `jev.yaml`, bump its `version`, so traces record which settings produced them.

## Troubleshooting

| You see | Do this |
|---|---|
| `requires a different Python` during `pip install` | install Python 3.10 or later and recreate the virtual environment |
| `Repository not found` or `Authentication failed` during `pip install` | ask the repository owner for access, then sign in when git asks (step 1) |
| `jevrag-kit` is not recognized | activate the virtual environment, or use `python -m jevrag_kit` |
| `MissingDependency: ... pip install "jevrag-kit[openai]"` | install the extra named in the message |
| `MissingCredentials: X is not set` | set variable `X` (step 4). Check that `api_key_env` names the variable you actually set |
| `ConfigError: ...` | the message names the setting and the problem; fix it and run `jevrag-kit check jev.yaml` |
| `doctor`: `FAIL llm ... 401` | the key is wrong, or belongs to a different provider than `base_url` |
| `doctor`: `FAIL llm ... 404` or "model not found" | that provider doesn't serve this `model` name |
| An error mentioning `tool_choice` | set `llm.tool_choice: auto` |
| An error mentioning `thinking` (Anthropic-style endpoints) | set `llm.request_params: {}` |
| An error saying to use `max_completion_tokens` (newer OpenAI models) | set `llm.max_tokens_param: max_completion_tokens` |
| Answers abstain more than expected | run `jevrag-kit try` and read the routes: passages dropped as `not_relevant` or `no_evidence` point to the questions or thresholds (step 9). Very small models also write claims less reliably |

Still stuck? Run `jevrag-kit doctor` and share its output without your `.env`. It never prints key values.
