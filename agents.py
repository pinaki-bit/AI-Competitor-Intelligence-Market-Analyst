import json
import logging
import os
import re
import threading
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_orchestrator")


# ──────────────────────────────────────────────────────────────
#  MODEL CONFIG — names, fallbacks, and pricing
# ──────────────────────────────────────────────────────────────
# Tier → ordered list of model names to try. The first one whose API call
# succeeds is used; the rest are fallbacks for older keys / region blocks.
# Override the primary model per tier with:
#   GEMINI_MODEL_FAST  (default tier="Quick")
#   GEMINI_MODEL_PRO   (default tier="Deep Dive")
#
# Conservative per-1K-token USD estimates. Used only for the CostGuard —
# not for billing.
_GEMINI_COST_PER_1K = {
    # Current generation
    "gemini-2.5-pro":     {"input": 0.00125,  "output": 0.010},
    "gemini-2.5-flash":   {"input": 0.0003,   "output": 0.0025},
    "gemini-2.0-flash":   {"input": 0.0001,   "output": 0.0004},
    "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
    # Legacy — still listed so older keys keep working as fallbacks
    "gemini-1.5-pro":     {"input": 0.00125,  "output": 0.005},
    "gemini-1.5-flash":   {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-flash-8b": {"input": 0.0000375, "output": 0.00015},
}
_DEFAULT_PRICING_KEY = "gemini-2.0-flash"

# Tier → fallback chain. First entry is the default; the rest are tried in
# order if the previous model returns 404 / not found.
_MODEL_TIERS = {
    "Quick": [
        os.environ.get("GEMINI_MODEL_FAST", "gemini-2.0-flash"),
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ],
    "Deep Dive": [
        os.environ.get("GEMINI_MODEL_PRO", "gemini-2.5-pro"),
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
}

_DEFAULT_BUDGET_USD = float(os.environ.get("AGENT_BUDGET_USD", "0.50"))


# Gemini error messages worth retrying with a different model. v1beta and
# v1 both return NOT_FOUND for deprecated model IDs.
_MODEL_NOT_FOUND_HINTS = (
    "is not found",
    "NOT_FOUND",
    "is not supported for generateContent",
    "model not found",
)


def _is_model_not_found(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(h.lower() in msg for h in _MODEL_NOT_FOUND_HINTS)


class BudgetExceeded(RuntimeError):
    """Raised when the per-run USD budget is exceeded."""


class CostGuard:
    """Thread-safe accumulator for estimated USD spend during a single run."""

    def __init__(self, budget_usd: float = _DEFAULT_BUDGET_USD):
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self._spent = 0.0
        self.calls = 0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        pricing = _GEMINI_COST_PER_1K.get(model, _GEMINI_COST_PER_1K[_DEFAULT_PRICING_KEY])
        cost = (input_tokens / 1000.0) * pricing["input"] + (output_tokens / 1000.0) * pricing["output"]
        with self._lock:
            self._spent += cost
            self.calls += 1
            if self._spent > self.budget_usd:
                raise BudgetExceeded(
                    f"Run budget exceeded: ${self._spent:.4f} > ${self.budget_usd:.4f} "
                    f"after {self.calls} LLM calls. Set AGENT_BUDGET_USD to raise the cap."
                )

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent


def _models_for_tier(depth: str) -> list[str]:
    """Return the ordered fallback list of model names for a depth tier."""
    chain = _MODEL_TIERS.get(depth) or _MODEL_TIERS["Quick"]
    # De-dup while preserving order
    seen, out = set(), []
    for m in chain:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars / token). Replace with a real tokenizer if needed."""
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────────────────────
#  TOPIC SANITIZATION — make prompts robust to any input
# ──────────────────────────────────────────────────────────────
# Common reasons prompts fail on company names like "Figma" or "Vercel":
#   1. Trailing punctuation or extra whitespace from the UI
#   2. LLM doesn't recognize the bare ticker / lowercase form
#   3. The search query is too narrow to find useful results
#   4. The model gets a one-word topic and hallucinates context
#
# _sanitize_topic normalizes the string and adds a hint when the topic
# looks like a bare product/company name, so the LLM has a clear anchor.
_KNOWN_COMPANIES = {
    "figma": "Figma (the collaborative interface design tool)",
    "vercel": "Vercel (the cloud platform for frontend frameworks)",
    "supabase": "Supabase (the open-source Firebase alternative)",
    "notion": "Notion (the connected workspace tool)",
    "linear": "Linear (the issue tracking tool for product teams)",
    "loom": "Loom (the async video messaging tool)",
    "stripe": "Stripe (the online payments platform)",
    "retool": "Retool (the low-code internal tools builder)",
    "slack": "Slack (the team messaging platform)",
    "airtable": "Airtable (the spreadsheet-database hybrid)",
    "zapier": "Zapier (the workflow automation platform)",
    "openai": "OpenAI (the AI research and products company)",
    "anthropic": "Anthropic (the AI safety company behind Claude)",
    "github": "GitHub (the code hosting and collaboration platform)",
    "gitlab": "GitLab (the DevOps platform)",
    "cloudflare": "Cloudflare (the web infrastructure and security company)",
    "datadog": "Datadog (the cloud monitoring and analytics platform)",
    "snowflake": "Snowflake (the cloud data platform)",
    "mongodb": "MongoDB (the document database)",
    "postgresql": "PostgreSQL (the open-source relational database)",
    "redis": "Redis (the in-memory data store)",
    "docker": "Docker (the container platform)",
    "kubernetes": "Kubernetes (the container orchestration system)",
    "shopify": "Shopify (the e-commerce platform)",
    "hubspot": "HubSpot (the CRM and marketing platform)",
    "salesforce": "Salesforce (the CRM platform)",
    "atlassian": "Atlassian (the team collaboration and DevOps tools maker)",
    "jira": "Jira (the project and issue tracking tool)",
    "confluence": "Confluence (the team wiki and documentation tool)",
    "trello": "Trello (the visual project management tool)",
    "asana": "Asana (the work management platform)",
    "monday": "monday.com (the work operating system)",
    "zoom": "Zoom (the video conferencing platform)",
    "dropbox": "Dropbox (the cloud file storage service)",
    "box": "Box (the enterprise content management platform)",
    "twilio": "Twilio (the customer engagement platform)",
    "sendgrid": "SendGrid (the email delivery service)",
    "mailchimp": "Mailchimp (the email marketing platform)",
    "klaviyo": "Klaviyo (the ecommerce marketing automation platform)",
    "intercom": "Intercom (the customer messaging platform)",
    "zendesk": "Zendesk (the customer service platform)",
    "freshworks": "Freshworks (the SaaS business software suite)",
}


def _sanitize_topic(topic: str) -> str:
    """
    Normalize a user-supplied topic so the LLM and Tavily handle it well.

    - Strips whitespace, collapses internal runs of whitespace.
    - Removes leading/trailing punctuation that often slips in from UIs.
    - Caps the length defensively (the UI also caps at 100 chars).
    - For a known company/product, returns a short expansion so the model
      has unambiguous context — fixes the "Figma" / "Vercel" being
      misread as a vague one-word niche.
    """
    if not topic:
        return ""
    t = topic.strip()
    # Collapse internal whitespace
    t = re.sub(r"\s+", " ", t)
    # Strip leading/trailing punctuation (keep hyphens inside words)
    t = t.strip(" .,;:!?\"'`~()[]{}<>*_/\\")
    # Cap length defensively
    t = t[:100].rstrip()
    if not t:
        return ""
    # If it's a known product / company, return the disambiguated form
    key = t.lower().strip()
    if key in _KNOWN_COMPANIES:
        return _KNOWN_COMPANIES[key]
    return t


def _topic_search_queries(topic: str) -> list[str]:
    """
    Build a small set of diverse search queries for a topic.

    The original code used a single narrow query; the LLM then had nothing
    useful to work with if Tavily returned thin results. A few varied
    queries dramatically improve coverage for bare product/company names.
    """
    t = topic.strip()
    return [
        f"{t} company overview products features pricing",
        f"{t} competitors alternatives market share",
        f"{t} funding revenue business model",
    ]


# ──────────────────────────────────────────────────────────────
#  RATE LIMITER — simple sliding-window per-key throttle
# ──────────────────────────────────────────────────────────────
class RateLimiter:
    """Minimal sliding-window limiter. Thread-safe."""

    def __init__(self, max_calls: int = 10, per_seconds: float = 60.0):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._lock = threading.Lock()
        self._calls: list[float] = []

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self.per_seconds]
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.per_seconds - (now - self._calls[0])
            time.sleep(max(0.05, wait))


# Module-level shared limiter (Gemini free tier is ~15 RPM)
_llm_limiter = RateLimiter(max_calls=12, per_seconds=60.0)


def retry(max_attempts: int = 3, initial_delay: float = 1.0, backoff: float = 2.0,
          retry_on: tuple = (Exception,)):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except BudgetExceeded:
                    # Never retry a budget error — it will just keep failing
                    raise
                except retry_on as e:
                    last_exc = e
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                                   f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff
            raise last_exc
        return wrapper
    return decorator


@retry(max_attempts=3, initial_delay=1.5, backoff=2.0)
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for competitor information and market news using Tavily.
    Retries up to 3 times on transient failures (network, rate limits, etc.).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set."

    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    logger.info(f"Executing Tavily search for query: {query}")
    response = client.search(query=query, max_results=max_results)
    results = []
    for item in response.get("results", []):
        results.append(
            f"Title: {item.get('title')}\n"
            f"URL: {item.get('url')}\n"
            f"Content: {item.get('content')}\n"
            f"---"
        )
    return "\n".join(results)


# Define fallback agent execution using standard LangChain calls
def run_custom_agent_analysis(topic: str, depth: str, status_callback=None) -> str:
    """
    Fallback agent workflow using direct LangChain + Tavily search calls.
    Provides progress updates via status_callback.

    Tries the primary model for the tier, then walks the fallback chain if
    the API rejects the model name (404 / NOT_FOUND — common when keys were
    issued before a model was released or after one was retired).
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    google_key = os.environ.get("GEMINI_API_KEY")
    candidates = _models_for_tier(depth)

    # Sanitize the topic up-front so prompts and search queries are clean.
    # This fixes the "Figma", "Vercel" → rejected / misread issues.
    topic = _sanitize_topic(topic)
    if not topic:
        raise ValueError("Topic is empty after sanitization — please enter a company name or product niche.")

    # Build the LLM for the first model that we can construct without error.
    # Construction rarely fails — the 404 surfaces on the first .invoke().
    # We pick the first candidate; fallback kicks in inside _invoke().
    model_name = candidates[0]
    if status_callback:
        status_callback("Researcher", f"Using Gemini model: {model_name} (fallbacks: {', '.join(candidates[1:]) or 'none'})")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_key,
        temperature=0.2,
    )

    guard = CostGuard()

    def _build_llm(name):
        return ChatGoogleGenerativeAI(
            model=name,
            google_api_key=google_key,
            temperature=0.2,
        )

    def _invoke(prompt, **vars):
        """Rate-limited, budget-tracked LLM invocation with model fallback."""
        nonlocal model_name, llm
        _llm_limiter.acquire()
        chain = prompt | llm
        try:
            msg = chain.invoke(vars)
        except Exception as e:
            # If this looks like a model-not-found error AND we have a fallback,
            # swap the model and try once. Otherwise re-raise the original.
            if not _is_model_not_found(e) or model_name == candidates[-1]:
                raise
            new_model = candidates[candidates.index(model_name) + 1]
            logger.warning(
                f"Model '{model_name}' unavailable ({type(e).__name__}: {e}). "
                f"Falling back to '{new_model}'."
            )
            if status_callback:
                status_callback("Researcher", f"⚠ {model_name} unavailable — switching to {new_model}")
            model_name = new_model
            llm = _build_llm(new_model)
            chain = prompt | llm
            msg = chain.invoke(vars)

        content = msg.content if isinstance(msg.content, str) else "".join(
            getattr(c, "text", str(c)) for c in msg.content
        )
        usage = getattr(msg, "usage_metadata", None) or {}
        in_tok = usage.get("input_tokens") or estimate_tokens(str(vars))
        out_tok = usage.get("output_tokens") or estimate_tokens(content)
        guard.record(model_name, in_tok, out_tok)
        return content

    # ----------------------------------------------------
    # Step 1: Researcher Agent
    # ----------------------------------------------------
    if status_callback:
        status_callback("Researcher", f"Searching the web for details on '{topic}'...")

    # Run a few diverse search queries — fixes the "single query returned
    # thin results" failure mode that made the LLM produce empty output.
    search_queries = _topic_search_queries(topic)
    all_results: list[str] = []
    for q in search_queries[:2]:  # cap to keep Tavily usage reasonable
        all_results.append(tavily_search(q, max_results=6 if depth == "Deep Dive" else 3))
    search_results = "\n\n".join(all_results) or "No web results were returned for this topic."

    if status_callback:
        status_callback("Researcher", "Analyzing search results & compiling profile...")

    researcher_prompt = ChatPromptTemplate.from_template(
        "You are an expert market research analyst. Your job is to gather and synthesize detailed information about the target company or product niche: {topic}.\n\n"
        "Here are search results from the web (multiple queries combined):\n"
        "{search_results}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. {topic} is a real, well-known company or product. Treat it as a specific named entity, not a vague category.\n"
        "2. If the search results are thin, use your own training knowledge of {topic} to fill in the profile — be specific.\n"
        "3. Include key features, recent news (2024-2026), tech stack details, target audience, and business model.\n"
        "4. Keep the summary factual and professional. If a data point is uncertain, say so.\n\n"
        "Output a structured markdown profile of {topic}."
    )

    research_report = _invoke(researcher_prompt,
                              topic=topic, search_results=search_results)

    # ----------------------------------------------------
    # Step 2: Competitor Analyst Agent
    # ----------------------------------------------------
    if status_callback:
        status_callback("Analyst", f"Researching key competitors of '{topic}'...")

    comp_queries = [
        f"{topic} main competitors alternatives",
        f"{topic} vs market share comparison",
    ]
    comp_results: list[str] = []
    for q in comp_queries[:2]:
        comp_results.append(tavily_search(q, max_results=6 if depth == "Deep Dive" else 3))
    comp_search_results = "\n\n".join(comp_results) or "No competitor data was returned for this topic."

    if status_callback:
        status_callback("Analyst", "Conducting SWOT analysis and competitor benchmarking...")

    analyst_prompt = ChatPromptTemplate.from_template(
        "You are a veteran competitive intelligence specialist. Your job is to identify the top 3 competitors and perform a detailed SWOT analysis for {topic}.\n\n"
        "Here is the Research Profile of the target:\n"
        "{research_report}\n\n"
        "Here are web search results regarding competitors:\n"
        "{comp_search_results}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. {topic} is a real, named company or product — identify its REAL direct competitors from the same market category.\n"
        "2. If web results are thin, fall back on your training-data knowledge of {topic} and its market.\n"
        "3. The comparison table MUST compare {topic} against the 3 competitors on the same dimensions (features, pricing, target audience, market position).\n"
        "4. The SWOT must be specific to {topic} — not generic boilerplate.\n\n"
        "Please provide:\n"
        "1. An identification of the top 3 direct competitors of {topic}.\n"
        "2. A structured comparison table benchmarking {topic} against these 3 competitors.\n"
        "3. A complete SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for {topic}."
    )

    analysis_report = _invoke(analyst_prompt,
                              topic=topic, research_report=research_report,
                              comp_search_results=comp_search_results)

    # ----------------------------------------------------
    # Step 3: Report Writer Agent
    # ----------------------------------------------------
    if status_callback:
        status_callback("Writer", "Formatting and polishing final intelligence report...")

    writer_prompt = ChatPromptTemplate.from_template(
        "You are a professional business writer. You specialize in taking raw competitor data and SWOT analysis and writing clear, engaging executive-level reports formatted in clean Markdown.\n\n"
        "Here is the Research Profile:\n"
        "{research_report}\n\n"
        "Here is the Competitor & SWOT Analysis:\n"
        "{analysis_report}\n\n"
        "Compile and format these into a final, cohesive Market Intelligence Report for {topic}. "
        "Use clean Markdown formatting with clear H1, H2, H3 headers, bullet points, and tables. "
        "Make sure the report is comprehensive, professional, and well-structured. "
        "Do not include any greeting, intro conversational text, or wrapper markdown blocks (like ```markdown ... ```). "
        "Start directly with the title '# Market Intelligence Report: {topic}'."
    )

    final_report = _invoke(writer_prompt,
                           topic=topic, research_report=research_report,
                           analysis_report=analysis_report)

    if status_callback:
        status_callback("Writer", f"Analysis complete! (≈${guard.spent:.4f} estimated)")

    return final_report


def run_crew_analysis(topic: str, depth: str, status_callback=None) -> str:
    """
    Orchestration using the CrewAI framework.

    Wraps the run with a CostGuard via a per-step callback so a runaway
    CrewAI iteration can't burn through API budget. CrewAI's internal LLM
    calls aren't directly hookable, so we estimate spend per step and
    abort with BudgetExceeded when the cap is hit.
    """
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI

    google_key = os.environ.get("GEMINI_API_KEY")
    candidates = _models_for_tier(depth)
    model_name = candidates[0]

    # Sanitize topic the same way the custom path does, so behaviour is
    # consistent regardless of which orchestrator is selected.
    topic = _sanitize_topic(topic)
    if not topic:
        raise ValueError("Topic is empty after sanitization — please enter a company name or product niche.")

    if status_callback:
        status_callback("Researcher", f"Using Gemini model: {model_name}")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_key,
        temperature=0.2,
    )

    # Per-run budget tracker. We can't intercept CrewAI's internal LLM
    # calls cleanly, so we estimate spend after each step (a step is
    # roughly one LLM round-trip per agent) and bail out via the step
    # callback if we exceed the cap.
    guard = CostGuard()
    _aborted = {"flag": False}

    def _step_callback(step_output):
        """Called by CrewAI after each agent step."""
        try:
            # step_output is a CrewAgentAction / AgentAction-like object;
            # .text and .tool_calls give us a rough prompt/response size.
            text = getattr(step_output, "text", "") or str(step_output)
            in_tok = estimate_tokens(text)
            out_tok = max(1, len(text) // 8)  # output is typically smaller
            guard.record(model_name, in_tok, out_tok)
        except BudgetExceeded as be:
            logger.warning(f"CrewAI step hit budget cap: {be}")
            _aborted["flag"] = True
            raise
        except Exception as e:
            # Never let a logging hiccup break the run
            logger.debug(f"step_callback swallowed: {e}")

    @tool("WebSearch")
    def web_search_tool(query: str) -> str:
        """Search the web for competitor details, tech stacks, and company news."""
        return tavily_search(query, max_results=6 if depth == "Deep Dive" else 3)

    # Researcher
    if status_callback:
        status_callback("Researcher", "Initializing web research agent...")

    researcher = Agent(
        role="Market Researcher",
        goal=f"Gather detailed profile, features, tech stack, and news about {topic}",
        backstory=f"You are an expert market research analyst. Your job is to crawl the web, find tech stacks, key features, pricing, and recent news about {topic}, a real, named company or product. If web results are thin, fall back on your training-data knowledge.",
        tools=[web_search_tool],
        llm=llm,
        verbose=True
    )

    research_task = Task(
        description=f"Conduct deep research on {topic}. {topic} is a real, well-known company or product — treat it as a specific named entity. Gather features, recent news (2024-2026), target audience, and tech stack details. If search results are thin, rely on your training knowledge of {topic}.",
        expected_output="A structured markdown summary of the company/product details, tech stack, features, and recent news.",
        agent=researcher
    )

    # Analyst
    if status_callback:
        status_callback("Analyst", "Initializing competitor analysis agent...")

    analyst = Agent(
        role="Competitor & SWOT Analyst",
        goal=f"Analyze competitors and perform a detailed SWOT analysis for {topic}",
        backstory=f"You are a veteran competitive intelligence specialist. You identify the top 3 direct competitors of {topic}, analyze their strengths and weaknesses, and formulate a clear SWOT matrix. Use your training knowledge of {topic}'s market category to identify real competitors if web results are thin.",
        tools=[web_search_tool],
        llm=llm,
        verbose=True
    )

    analyst_task = Task(
        description=f"Identify the top 3 REAL direct competitors of {topic}. {topic} is a specific company or product — name actual competitors from the same market category. Perform a SWOT analysis comparing {topic} against these competitors.",
        expected_output="A comprehensive SWOT analysis layout and a comparison table of the top 3 competitors with key metrics (features, pricing, market share).",
        agent=analyst
    )

    # Writer
    if status_callback:
        status_callback("Writer", "Initializing executive report writer agent...")

    writer = Agent(
        role="Executive Report Writer",
        goal="Compile and write a polished, professional market intelligence report",
        backstory="You are a professional business writer. You specialize in taking raw competitor data and SWOT analysis and writing clear, engaging executive-level reports formatted in Markdown.",
        llm=llm,
        verbose=True
    )

    writer_task = Task(
        description=f"Synthesize the research and SWOT analysis into a final, polished executive report in Markdown. Ensure the report has a clean structure with clear headers (H1, H2, H3), bullet points, and tables. Do not include any markdown outer wrappers like ```markdown. Start directly with '# Market Intelligence Report: {topic}'.",
        expected_output="The final market intelligence report in clean Markdown formatting, ready to be converted into a PDF. Starting directly with '# Market Intelligence Report: {topic}'.",
        agent=writer
    )

    if status_callback:
        status_callback("Analyst", "Running AI Agent Crew workflow...")

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analyst_task, writer_task],
        process=Process.sequential,
        verbose=True,
        step_callback=_step_callback,
        max_rpm=12,  # align with module-level RateLimiter
    )

    try:
        result = crew.kickoff(inputs={"topic": topic})
    except BudgetExceeded:
        raise
    except Exception:
        # If the step callback aborted us, surface that as BudgetExceeded
        if _aborted["flag"]:
            raise BudgetExceeded(
                f"CrewAI run aborted: ${guard.spent:.4f} exceeded the per-run budget."
            ) from None
        raise

    if status_callback:
        status_callback("Writer", f"Analysis complete! (≈${guard.spent:.4f} estimated)")

    return str(result)


def run_analysis(topic: str, depth: str, status_callback=None) -> tuple[str, str]:
    """
    Main entrypoint to run analysis. Attempts CrewAI first and falls back to custom langchain runner.
    Returns: (report_markdown, execution_mode)
    """
    # Attempt CrewAI
    try:
        logger.info("Attempting to run analysis with CrewAI...")
        # Check if crewai can be imported
        import crewai
        # Verify crewai has expected attributes to make sure it's valid
        _ = crewai.Agent

        if status_callback:
            status_callback("Researcher", "CrewAI active. Running agent crew...")
        result = run_crew_analysis(topic, depth, status_callback)
        return result, "CrewAI Orchestrator"
    except Exception as e:
        logger.warning(f"CrewAI fallback: {str(e)}")
        if status_callback:
            status_callback("Researcher", "Falling back to Custom LangChain Orchestrator...")

        result = run_custom_agent_analysis(topic, depth, status_callback)
        return result, "Custom LangChain Orchestrator"


def run_mock_analysis(topic: str, status_callback=None) -> str:
    """
    Simulates a full multi-agent analysis with rich structured output.

    NOTE: The report body below is a generic template — the data points
    (ARR figures, founded year, competitors, tech stack) are illustrative
    placeholders, not real research. The UI labels the result as Demo Mode.
    """
    import time

    steps = [
        ("Researcher", f"Spawning web scraper session for '{topic}'..."),
        ("Researcher", f"Crawling Tavily index — query: '{topic} company overview features'..."),
        ("Researcher", "Extracted 12 search results. Parsing tech stack & feature lists..."),
        ("Researcher", "Compiling product profile, target audience & recent news..."),
        ("Analyst",   "Identifying top 3 direct competitors in the market..."),
        ("Analyst",   "Benchmarking pricing models, feature depth & market position..."),
        ("Analyst",   "Running SWOT matrix — cross-referencing strengths vs competitor gaps..."),
        ("Analyst",   "Scoring market opportunity, competitive pressure & growth trajectory..."),
        ("Writer",    "Drafting Executive Summary & Market Overview sections..."),
        ("Writer",    "Formatting Competitor Benchmark Table & SWOT quadrants..."),
        ("Writer",    "Adding Strategic Recommendations & finalizing report structure..."),
    ]

    for agent, msg in steps:
        if status_callback:
            status_callback(agent, msg)
        time.sleep(0.6)

    return f"""# Market Intelligence Report: {topic}

## Executive Summary

**{topic}** operates in a rapidly evolving technology landscape characterized by intense competition, accelerating AI adoption, and shifting customer expectations. This report presents a synthesized analysis of the company's market positioning, competitive dynamics, and strategic outlook based on automated web research and AI-driven analysis.

Key findings indicate that **{topic}** holds a differentiated position in the market through its modern technical architecture and user-centric design philosophy. However, scaling challenges and increasing competitive pressure from well-funded incumbents represent meaningful risks to sustained growth.

---

## Company Profile

- **Category**: B2B SaaS
- **Founded**: ~2020 (illustrative)
- **Headquarters**: San Francisco, CA (illustrative)
- **Business Model**: Freemium SaaS with usage-based paid tiers
- **Target Audience**: Developers, startups, and mid-market engineering teams
- **Core Value Proposition**: Modern, scalable platform with strong developer experience
- **Tech Stack**: Cloud-native stack (TypeScript / Python / Go — illustrative)
- **Integrations**: GitHub, Vercel, Stripe, Auth providers (illustrative)

---

## Market Overview

The global B2B SaaS market is projected to grow at a **CAGR of ~15%** through 2030, reaching an estimated **$300B+** by 2030. Key macro-drivers include:

- Accelerating AI-native application development
- Enterprise shift toward cloud-native architectures
- Growing preference for open-source vendor flexibility
- Increased demand for real-time data and edge computing capabilities

**{topic}** is strategically positioned at the intersection of these trends, with developer adoption serving as a powerful growth flywheel.

---

## Competitive Landscape

### Top 3 Direct Competitors

| Metric | **{topic}** | Competitor A (Established Leader) | Competitor B (Scale Player) | Competitor C (Niche Innovator) |
| --- | --- | --- | --- | --- |
| Market Position | Rising Challenger | Dominant Leader | Strong Contender | Niche Innovator |
| Pricing Model | Freemium + Usage | Enterprise + Tiered | Freemium + Scale | Freemium + Scale |
| Open Source | Yes | No | Partial | Yes |
| Real-time | Yes | Yes | Limited | Limited |
| Self-Hostable | Yes | No | No | Yes |
| Vendor Lock-in | Low | High | Medium | Low |
| Est. ARR (illustrative) | $30M+ | $500M+ | $50M+ | $10M+ |

---

## SWOT Analysis

### Strengths
- **Developer-First Approach**: Strong emphasis on DX drives community-led adoption
- **Modern Architecture**: Cloud-native foundation enables rapid scaling
- **Flexible Deployment**: Self-hostable for compliance-sensitive customers
- **Open Ecosystem**: Open-source components foster trust and extensibility
- **Active Community**: Engaged developer base and contributor community
- **Venture Backing**: Well-capitalized for sustained product investment

### Weaknesses
- **Enterprise Gaps**: Compliance certifications may lag established incumbents
- **Operational Complexity**: Advanced deployment modes require DevOps expertise
- **Brand Awareness**: Lower unaided recall vs. category leaders among non-developers
- **Platform Coverage Gaps**: Secondary platform support may trail category leaders

### Opportunities
- **AI-Native Workloads**: Position {topic} as the foundation for AI applications
- **Enterprise Tier Expansion**: Move upmarket with compliance packages and SLAs
- **Edge Computing**: Capture demand for sub-50ms global response times
- **Geographic Expansion**: Underpenetrated APAC and EMEA developer markets

### Threats
- **Hyperscaler Bundling**: Major cloud vendors bundle competing offerings with cloud credits
- **Open-Source Fragmentation**: Adjacent OSS projects could erode differentiation
- **Talent War**: Competition for senior distributed-systems engineers is intense
- **Macro Slowdown**: Reduced IT budgets in a downturn delay purchase decisions

---

## Strategic Recommendations

1. **Lead with AI Positioning**: Promote AI-native use cases aggressively — fastest-growing segment
2. **Build Enterprise Pipeline**: Dedicated sales motion with compliance certifications
3. **Deepen Platform Coverage**: Close feature gaps with category leaders to unlock new segments
4. **Double Down on Community**: Open-source community programs remain the most capital-efficient growth channel
5. **Edge-First Messaging**: Position for sub-50ms global API response demand

---

## Conclusion

**{topic}** has established itself as a credible challenger in its category, driven by strong developer experience and a modern product foundation. To reach the next growth stage, the company must balance its community-first ethos with enterprise-grade reliability and compliance. The AI infrastructure wave represents a significant near-term opportunity that {topic} is well-positioned to capture.

*Report generated by AI Competitor Intelligence & Market Analyst Team — Demo Mode (template data)*
"""


# ──────────────────────────────────────────────────────────────
#  MARKET SCORING — LLM-as-judge (preferred) + heuristic fallback
# ──────────────────────────────────────────────────────────────
_SCORE_KEYS = (
    "market_opportunity",
    "competitive_pressure",
    "growth_trajectory",
    "innovation_score",
    "risk_level",
    "market_maturity",
)

_SCORING_PROMPT = """You are a senior market intelligence analyst. Read the report below about **{topic}** and score it on 6 dimensions, each on a 0–100 integer scale.

Definitions:
- market_opportunity: how large and accessible is the addressable market
- competitive_pressure: how intense the existing competition is
- growth_trajectory: the momentum / expansion rate implied by the report
- innovation_score: degree of differentiation and technical novelty
- risk_level: overall strategic and market risk exposure
- market_maturity: how saturated / established the category is

Output a JSON object with EXACTLY these six keys (lowercase, snake_case). Each value must be an integer between 0 and 100. No commentary, no markdown fences, no other text.

Example: {{"market_opportunity": 78, "competitive_pressure": 64, ...}}

Report:
{report}
"""


def _clamp_score(value, lo=0, hi=100):
    """Clamp a score to a 0–100 integer."""
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))


def _parse_score_json(text: str) -> dict | None:
    """
    Robustly extract a {key: int, ...} score dict from LLM output.

    Handles:
    - bare JSON
    - ```json ... ``` fenced JSON
    - JSON with surrounding prose
    - Missing keys (filled with None)
    - Out-of-range values (clamped)
    """
    if not text:
        return None
    s = text.strip()
    # Strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # Find the first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start:end + 1]
    try:
        data = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    result = {}
    for k in _SCORE_KEYS:
        if k in data:
            v = _clamp_score(data[k])
            if v is not None:
                result[k] = v
    # All six required for a successful parse
    if len(result) != len(_SCORE_KEYS):
        return None
    return result


def score_market_with_llm(
    report_markdown: str,
    topic: str,
    model_name: str | None = None,
) -> dict | None:
    """
    Use an LLM to score a market intelligence report.

    Returns a dict of 6 scores (0–100 ints) on success, or None on any
    failure (missing key, parse error, etc.). Callers should fall back to
    the heuristic scorer when this returns None.
    """
    if not report_markdown or not topic:
        return None
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        name = model_name or _models_for_tier("Quick")[0]
        llm = ChatGoogleGenerativeAI(
            model=name,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.0,  # scoring should be deterministic
        )
        # Truncate the report to keep the scoring call cheap (~20k tokens max)
        truncated = (report_markdown or "")[:20_000]
        prompt = ChatPromptTemplate.from_template(_SCORING_PROMPT)
        chain = prompt | llm
        msg = chain.invoke({"topic": topic, "report": truncated})
        content = msg.content if isinstance(msg.content, str) else "".join(
            getattr(c, "text", str(c)) for c in msg.content
        )
        return _parse_score_json(content)
    except Exception as e:
        logger.warning(f"LLM scoring failed, will fall back to heuristic: {e}")
        return None


def _heuristic_market_scores(report_markdown: str, topic: str) -> dict:
    """
    Derive illustrative market intelligence scores for dashboard display.

    The scores combine a deterministic topic seed with lightweight keyword
    signals pulled from the generated report. They are intended as a quick
    at-a-glance summary, NOT as authoritative metrics. For real scoring,
    use `score_market_with_llm` which calls a Gemini pass to emit
    structured numeric scores.

    Returns a dict with score keys on a 0–100 scale.
    """
    import hashlib

    text = (report_markdown or "").lower()
    seed = int(hashlib.md5((topic or "").lower().encode()).hexdigest()[:4], 16)

    def base(value, variance=15):
        return value + (seed % variance) - variance // 2

    def signal(keywords, cap=6):
        """Cap keyword-match contribution so a single word can't dominate."""
        return min(cap, sum(text.count(k) for k in keywords))

    def clamp(value, lo=10, hi=100):
        return max(lo, min(hi, value))

    return {
        "market_opportunity":  clamp(base(70, 20) + signal(
            ["opportunit", "growth", "expansion", "emerging", "untapped"])),
        "competitive_pressure":clamp(base(65, 25) + signal(
            ["competit", "rival", "incumbent", "market share", "wars"])),
        "growth_trajectory":   clamp(base(75, 18) + signal(
            ["growth", "scaling", "traction", "adoption", "momentum"])),
        "innovation_score":    clamp(base(68, 22) + signal(
            ["innovat", "ai-native", "differenti", "novel", "patent"])),
        "risk_level":          clamp(base(45, 30) + signal(
            ["risk", "threat", "challenge", "vulnerab", "concern", "headwind"])),
        "market_maturity":     clamp(base(55, 20) + signal(
            ["mature", "saturat", "established", "consolidat"])),
    }


def get_market_scores(report_markdown: str, topic: str,
                      prefer_llm: bool = True) -> dict:
    """
    Return market intelligence scores for the report.

    If `prefer_llm` is True AND a Gemini API key is configured, this calls
    `score_market_with_llm` for real LLM-as-judge scoring. On any LLM
    failure (missing key, parse error, network) it falls back to
    `_heuristic_market_scores` so the dashboard always has values to show.

    The `prefer_llm=False` path is used by Demo Mode so the report works
    without an API key.
    """
    if prefer_llm:
        llm_scores = score_market_with_llm(report_markdown, topic)
        if llm_scores is not None:
            return llm_scores
    return _heuristic_market_scores(report_markdown, topic)
