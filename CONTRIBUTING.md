# Contributing to AI Competitor Intelligence & Market Analyst Team

Thanks for your interest in contributing! This document covers the basics
for getting the project running locally and submitting changes.

## Development setup

```powershell
# Clone and enter the project
cd "AI Competitor Intelligence & Market Analyst Team"

# Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dev dependencies (includes ruff + pytest)
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

You'll need API keys for real research runs. Copy `.env.example` to `.env`
and fill in your [Gemini](https://aistudio.google.com/app/apikey) and
[Tavily](https://tavily.com/) keys. To work without keys, enable **Demo
Mode** in the sidebar — the mock analysis path is fully runnable offline.

## Running tests

```powershell
pytest tests/ -v
```

The suite has 80+ tests, all offline — LLM and Tavily calls are mocked.
Coverage highlights:

- `test_agents.py` — `CostGuard` budget enforcement, `RateLimiter`
  sliding window, `retry` decorator, `get_market_scores` range &
  determinism, `run_mock_analysis` callbacks, model selection / fallback
  chain, `_sanitize_topic` (known company expansion, whitespace /
  punctuation stripping, length cap), `_topic_search_queries`.
- `test_pdf_generator.py` — empty input, ragged tables, fenced code
  blocks, nested lists, horizontal rules, inline code / bold, Unicode
  content, unbalanced markdown, multi-page output, auto-created output
  directories.
- `test_history_store.py` — atomic JSON persistence, concurrent-append
  safety, schema-corruption recovery, env-var path overrides.

## Linting

```powershell
ruff check .
ruff format --check .
```

Ruff is configured in `pyproject.toml`. The project targets Python 3.10+
and a 100-character line length.

## Code style

- **No comments unless asked.** Functions use docstrings only when the
  intent isn't obvious from the signature.
- Match the existing file conventions — `agents.py` uses a structured
  section header pattern (see lines 12, 121, 178, etc.) for new sections.
- Prefer the smallest fix that solves the problem. Don't refactor
  unrelated code in a feature commit.

## Pull request process

1. Fork the repo and create a feature branch (`git checkout -b feat/...`).
2. Add tests for any new behavior — the CI workflow runs on Python 3.11
   and 3.12.
3. Make sure `pytest` and `ruff check` both pass locally.
4. Update the README if you change user-facing behavior (env vars, model
   list, project structure, etc.).
5. Open a PR against `main` with a clear description of the change and
   the problem it solves.

## Reporting issues

Use GitHub Issues. Include:

- What you expected to happen
- What actually happened
- A minimal repro (topic, depth, model — omit any API keys)
- The full traceback if it's a crash

## License

By contributing, you agree that your contributions will be licensed under
the MIT License (see `LICENSE`).
