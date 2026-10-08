# Changelog

## 0.1.0 (2026-10-08)

First release, extracted from Saandru's `rag/score`, `rag/generate`, and `rag/verify`.

- JEV passage classifier with configurable questions, ordered routing rules (`above`, `at_least`, `below`, `at_most`), fallback, ordering, and caps.
- LLM layer with two built-in protocols: Anthropic Messages API and OpenAI Chat Completions API, each with any `base_url`. Configurable tool choice, extra request fields, SDK client options, and prompt templates. Pluggable `generate(prompt) -> Draft` interface.
- JEV claims checker with configurable relation wording, `auto_accept`, `min_quote_chars`, and the actions for low-confidence and unsupported claims.
- `Engine` chaining the three stages, with Saandru's trace format.
- `jevrag_kit.replay` and `jevrag-kit sweep` for threshold tuning without API calls.
- CLI: `init`, `check`, `doctor`, `try`, `sweep`.
- `jevrag_kit.testing` fakes for projects' own tests.
- `load_env_file` for `.env` files, and `MissingDependency` errors that name the install command for a missing SDK.
- With the defaults plus `source_type` in `classifier.state_fields`, decisions match Saandru exactly (`tools/parity_saandru.py`).
- Named `jevrag-kit` (import `jevrag_kit`). It was developed as `jevkit`, but that name and `jevguard` and `jevrag` belong to unrelated PyPI packages.
