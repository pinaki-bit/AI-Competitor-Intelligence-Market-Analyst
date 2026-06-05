# AI Competitor Intelligence & Market Analyst Team 📊

An interactive dashboard application where a user inputs a company name or product niche, and a team of specialized AI agents crawls the web, performs a SWOT analysis, compiles competitor data, and generates a polished, downloadable PDF report.

The system features a **dual-mode architecture** built for resilience. If `crewai` is available and compatible, it runs the workflow as a CrewAI multi-agent sequence. If there are package version conflicts (often seen with python 3.13 on Windows), it seamlessly falls back to a custom LangChain orchestrator with identical agent goals and behaviors.

---

## 🛠️ Tech Stack

*   **Frontend UI**: [Streamlit](https://streamlit.io/) with custom CSS for modern glassmorphism and dark mode aesthetics.
*   **Orchestration**: [CrewAI](https://www.crewai.com/) / [LangChain](https://www.langchain.com/) (Fallback).
*   **LLM API**: Google Gemini (`gemini-2.0-flash` by default, `gemini-2.5-pro` for Deep Dive) with a fallback chain that auto-switches to `gemini-2.5-flash` / legacy 1.5 models if the primary returns 404.
*   **Search Engine**: [Tavily API](https://tavily.com/) for fast, AI-optimized web searches.
*   **PDF Generation**: [FPDF2](https://pyfpdf.github.io/fpdf2/) for clean, corporate-styled reports.
*   **Reliability**: Token-aware `CostGuard` (per-run USD cap, default $0.50, applies to BOTH the CrewAI and the custom LangChain paths), `RateLimiter` (12 RPM sliding window), and exponential-backoff `retry` decorator on all LLM/search calls.
*   **Topic handling**: Built-in `_sanitize_topic` expands bare product names like `Figma` / `Vercel` / `Supabase` into disambiguated strings so the LLM doesn't misread them. Override defaults with `GEMINI_MODEL_FAST` / `GEMINI_MODEL_PRO`.
*   **Scoring**: LLM-as-judge market scores (6 dimensions, 0–100) with a deterministic heuristic fallback so the dashboard always has values. Demo Mode always uses the heuristic.
*   **Persistence**: Analysis history is saved to `~/.market_analyst/history.json` (override with `MARKET_ANALYST_HISTORY_PATH`) so a browser refresh doesn't lose your work.

---

## 🤖 Meet the Agent Team

1.  **Researcher Agent 🔍**: Crawls the web via Tavily to retrieve detailed information, features, tech stack details, and recent news about the target company or product niche.
2.  **Competitor Analyst Agent ⚖️**: Synthesizes research findings, identifies the top 3 direct competitors, builds a comparison benchmark table, and conducts a complete SWOT analysis.
3.  **Report Writer Agent ✍️**: Structure expert that gathers outputs from the Researcher and Analyst to compile a beautifully formatted Executive Markdown report.

---

## 📂 Project Structure

```text
market-analyst-agent/
│
├── app.py                # Streamlit Frontend Dashboard UI
├── agents.py             # Agent Configurations (CrewAI / Custom fallback)
├── pdf_generator.py      # Markdown to PDF renderer (fpdf2)
├── history_store.py      # Atomic JSON persistence for analysis history
├── styles.css            # Glassmorphism / dark-mode styling
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Ruff + pytest configuration
├── .env.example          # Sample environment configuration
├── .gitignore            # Standard Python + Streamlit exclusions
├── LICENSE               # MIT License
├── CONTRIBUTING.md       # Contribution guidelines
├── .github/workflows/    # GitHub Actions CI (ruff + pytest on 3.11/3.12)
│   └── ci.yml
├── tests/                # Pytest suite — 105 tests, no network required
│   ├── test_agents.py
│   ├── test_pdf_generator.py
│   └── test_history_store.py
└── README.md             # Project documentation
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
Ensure you are in the project folder:
```powershell
cd "AI Competitor Intelligence & Market Analyst Team"
```

### 2. Configure Environment Variables
Create a `.env` file in the root folder with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```
*Note: You can also enter these keys directly in the sidebar of the Streamlit interface.*

### 3. Run the App
Activate the virtual environment and launch Streamlit:
```powershell
# Activate venv (Windows)
.\venv\Scripts\Activate.ps1

# Run the app
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📄 PDF Generation Details
The PDF converter parses Markdown syntax line-by-side and is hardened against malformed LLM output:
*   **Cover Page**: A professional dark navy header banner, clear metadata block, and color-coded separator accents.
*   **Header/Footer**: Page headers tracking the company name and page footers dynamically rendering page numbers.
*   **Benchmarking Tables**: Renders comparison grids with alternating row backgrounds and high-contrast header columns. Ragged column counts are auto-padded.
*   **Rich Text**: Converts inline bolding (`**text**`), inline `code`, fenced code blocks, horizontal rules, and nested lists into formatted structures.
*   **Unicode**: Transliterates non-Latin-1 glyphs to safe ASCII so the PDF never crashes on em-dashes, smart quotes, CJK, or emoji.

## 🧪 Tests

The project ships with a 116-test `pytest` suite covering the PDF renderer, the agent helpers, model selection / fallback, topic sanitization, market-score parsing, and history persistence. Nothing in the suite hits the network — all LLM and Tavily calls are mocked.

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/ -v
```

Coverage highlights:
* `test_pdf_generator.py` — empty input, ragged tables, fenced code blocks, nested lists, horizontal rules, inline code/bold, Unicode content, unbalanced markdown, multi-page output, auto-created output directories.
* `test_agents.py` — `CostGuard` budget enforcement, `RateLimiter` sliding window, `retry` decorator (success/retry/exhaustion), `get_market_scores` dispatcher (LLM path + heuristic fallback), `run_mock_analysis` callbacks, missing `TAVILY_API_KEY` fallback, model selection & fallback chain, model-not-found detection, `_sanitize_topic` (known company expansion, whitespace/punctuation stripping, length cap), `_topic_search_queries` (multi-angle coverage), score JSON parsing (clean / fenced / prose-wrapped / out-of-range / missing-keys).
* `test_history_store.py` — atomic JSON persistence, schema-corruption recovery, env-var path overrides, concurrent-append safety (10 threads × 10 writes), clear-and-resume.

## 💸 Cost & Rate Safety

Every LLM call in both orchestrators (CrewAI and the custom LangChain fallback) is tracked by a `CostGuard` (default $0.50/run, override with `AGENT_BUDGET_USD` env var) and a module-level `RateLimiter` (12 RPM, well under Gemini's free-tier 15 RPM). The `retry` decorator short-circuits on `BudgetExceededError` so a runaway run fails fast instead of burning more spend. The CrewAI path uses a `step_callback` to estimate spend per agent step and abort the crew if the budget cap is hit.

## 🔁 History & Persistence

Every completed analysis is appended to a JSON file on disk (default `~/.market_analyst/history.json`, override with `MARKET_ANALYST_HISTORY_PATH`). The file is written atomically via temp-file + `os.replace`, and the read-modify-write cycle is serialized on a module-level lock so concurrent Streamlit threads can't lose updates. The list is capped at 20 entries to keep the file small. Use the **🗑️ Clear history** button in the sidebar to wipe the store.

## 🤖 Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and PR against `main` / `master`:

- Lint with `ruff check` + `ruff format --check`
- Run the full 105-test pytest suite on Python 3.11 and 3.12
- All steps run without network access (no API keys required)

## 📜 License

MIT — see [`LICENSE`](./LICENSE). Contributions are welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md).
