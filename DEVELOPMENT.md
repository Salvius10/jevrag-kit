# Developing jevrag-kit

Notes for people who change jevrag-kit itself. Users of the package only need the [README](README.md).

## Setup

```powershell
git clone https://github.com/Salvius10/jevrag-kit.git
cd jevrag-kit
uv sync
```

If the folder is inside OneDrive, set `$env:UV_LINK_MODE = "copy"` first: OneDrive folders don't support the hardlinks uv uses by default.

## Layout

| Path | Contents |
|---|---|
| `src/jevrag_kit/classifier/` | passage scoring (TypeSafe) and routing rules |
| `src/jevrag_kit/llm/` | prompt, answer-tool schema, Anthropic and OpenAI generators |
| `src/jevrag_kit/checker/` | quote location, relation check (TypeSafe), release policy, answer assembly |
| `src/jevrag_kit/engine.py` | the pipeline that chains the three stages and writes the trace |
| `src/jevrag_kit/config.py` | every setting, its default, and its validation |
| `src/jevrag_kit/default_config.yaml` | the same defaults with comments; `jevrag-kit init` copies it, and a test keeps it equal to `config.py` |
| `src/jevrag_kit/replay.py` | threshold replay and sweeps over stored traces |
| `src/jevrag_kit/cli.py` | the `jevrag-kit` command |
| `tests/` | unit tests; `test_readme.py` runs every example in the README |
| `tools/parity_saandru.py` | compares decisions with Saandru's original code |

## Tests

```powershell
uv run pytest                                                              # no keys, no network
$env:UV_PROJECT_ENVIRONMENT = ".venv-3.10"; uv run --python 3.10 pytest    # another Python, separate environment
```

`tests/test_readme.py` runs every Python example in the README, validates every YAML example, and parses every documented command. When you change the README, run the tests.

To check the lowest supported dependency versions:

```powershell
uv venv --python 3.11 .venv-lowest
uv pip install --python .venv-lowest --resolution lowest-direct -e ".[all]" "pytest>=8.0"
.venv-lowest\Scripts\python.exe -m pytest
```

Delete the extra `.venv-*` folders afterwards. They are ignored by git, but OneDrive syncs them.

## Parity with Saandru

jevrag-kit was extracted from Saandru's `rag/score`, `rag/generate`, and `rag/verify`. [`examples/saandru.yaml`](examples/saandru.yaml) is Saandru's configuration in jevrag-kit terms. `tools/parity_saandru.py` runs both implementations on the same inputs and compares routing, ordering, quote location, release decisions, prompts, HTTP request bodies, end-to-end traces, and threshold replay. Run it with Saandru's interpreter:

```powershell
$env:PYTHONPATH = "$PWD\src"
..\Saandru\.venv\Scripts\python.exe tools\parity_saandru.py --saandru ..\Saandru
```

It must end with `N comparisons, 0 differences`. Run it after any change to the classifier, the LLM layer, the checker, or the engine.

## Releasing

1. Update `src/jevrag_kit/_version.py` and add an entry to [CHANGELOG.md](CHANGELOG.md).
2. Run the tests (and the parity check if core logic changed).
3. Commit, then `git tag v<version>` and `git push origin main --tags`.

Projects upgrade by changing `@v<version>` in their install command. Never move a tag that has been pushed: projects pinned to it would silently get different code.

## Verification record for 0.1.0

Checked on 2026-10-08:

| Check | Result |
|---|---|
| Unit tests | passed on Python 3.10, 3.11, 3.13, and 3.14, and with the lowest supported dependencies (pydantic 2.12.0, PyYAML 6.0.1, typesafe-sdk 0.7.2, anthropic 1.0.0, openai 1.55.3 and 2.0.0) |
| Parity with Saandru | 57,186 comparisons, 0 differences. With two small defects injected into jevrag-kit, it reported 2,469 differences |
| Install from GitHub | the pip and uv install commands from the README work in clean environments, and a fresh clone passes the tests |
| Live check | TypeSafe `jev-latest` scoring and relation; `openai/gpt-4.1-nano` (OpenAI style) and `anthropic/claude-haiku-4.5` (Anthropic style), both through AIML API |
