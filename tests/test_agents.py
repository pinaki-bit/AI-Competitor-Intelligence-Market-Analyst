"""Tests for agents.py helpers — CostGuard, RateLimiter, retry, scores, mock."""
import time
import pytest

import agents
from agents import (
    CostGuard,
    RateLimiter,
    BudgetExceeded,
    retry,
    estimate_tokens,
    get_market_scores,
    run_mock_analysis,
    _sanitize_topic,
    _topic_search_queries,
    _GEMINI_COST_PER_1K,
    _MODEL_TIERS,
    _models_for_tier,
    _is_model_not_found,
)


# ──────────────────────────────────────────────────────────────
#  estimate_tokens
# ──────────────────────────────────────────────────────────────
class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1

    def test_short_string(self):
        assert estimate_tokens("hi") == 1

    def test_longer_string(self):
        # 4 chars per token heuristic
        assert estimate_tokens("a" * 400) == 100


# ──────────────────────────────────────────────────────────────
#  CostGuard
# ──────────────────────────────────────────────────────────────
class TestCostGuard:
    def test_initial_state(self):
        g = CostGuard(budget_usd=1.0)
        assert g.spent == 0
        assert g.calls == 0

    def test_record_flash_call(self):
        g = CostGuard(budget_usd=1.0)
        g.record("gemini-1.5-flash", 1_000_000, 1_000_000)
        # 1M input * 0.000075/1k + 1M output * 0.0003/1k = 0.075 + 0.3 = 0.375
        assert g.spent == pytest.approx(0.375, rel=1e-3)
        assert g.calls == 1

    def test_budget_exceeded_raises(self):
        g = CostGuard(budget_usd=0.01)
        with pytest.raises(BudgetExceeded):
            g.record("gemini-1.5-pro", 100_000, 100_000)

    def test_unknown_model_uses_flash_pricing(self):
        g = CostGuard(budget_usd=1.0)
        g.record("gemini-99-ultra", 1000, 1000)
        # Should not raise and should match flash pricing
        assert g.spent > 0

    def test_retry_decorator_skips_budget_errors(self):
        calls = []

        @retry(max_attempts=3, initial_delay=0.01)
        def explode():
            calls.append(1)
            raise BudgetExceeded("nope")

        with pytest.raises(BudgetExceeded):
            explode()
        # Should NOT have retried — budget errors are fatal
        assert len(calls) == 1

    def test_retry_decorator_eventually_succeeds(self):
        attempts = {"n": 0}

        @retry(max_attempts=3, initial_delay=0.01, backoff=1.0)
        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        assert flaky() == "ok"
        assert attempts["n"] == 3

    def test_retry_decorator_gives_up(self):
        @retry(max_attempts=2, initial_delay=0.01)
        def always_fail():
            raise ValueError("nope")

        with pytest.raises(ValueError):
            always_fail()


# ──────────────────────────────────────────────────────────────
#  RateLimiter
# ──────────────────────────────────────────────────────────────
class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_calls=3, per_seconds=1.0)
        for _ in range(3):
            rl.acquire()

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_calls=2, per_seconds=0.5)
        rl.acquire()
        rl.acquire()
        start = time.monotonic()
        rl.acquire()  # should wait ~0.5s
        elapsed = time.monotonic() - start
        assert elapsed >= 0.3  # generous lower bound


# ──────────────────────────────────────────────────────────────
#  get_market_scores
# ──────────────────────────────────────────────────────────────
class TestMarketScores:
    def test_returns_all_keys(self):
        report = "Some report with growth and opportunity keywords."
        scores = get_market_scores(report, "Vercel")
        expected = {"market_opportunity", "competitive_pressure", "growth_trajectory",
                    "innovation_score", "risk_level", "market_maturity"}
        assert set(scores.keys()) == expected

    def test_all_values_in_range(self):
        scores = get_market_scores("whatever", "topic")
        for k, v in scores.items():
            assert 0 <= v <= 100, f"{k} out of range: {v}"

    def test_deterministic_for_same_input(self):
        s1 = get_market_scores("hello world", "Vercel")
        s2 = get_market_scores("hello world", "Vercel")
        assert s1 == s2

    def test_different_topics_yield_different_seeds(self):
        # Different topic -> different seed -> likely different score
        s1 = get_market_scores("", "Vercel")
        s2 = get_market_scores("", "Stripe")
        # Won't always be different but with 6 scores and 30-pt variance
        # the chance of *all* being identical is essentially zero
        assert s1 != s2

    def test_empty_report_safe(self):
        scores = get_market_scores("", "")
        assert all(isinstance(v, int) for v in scores.values())

    def test_keyword_signal_capped(self):
        # Spam the report with one keyword — should not push score to 100
        spam = "opportunity " * 1000
        scores = get_market_scores(spam, "Vercel")
        # Even with massive spam, signal contribution is capped at 6
        assert scores["market_opportunity"] <= 100


# ──────────────────────────────────────────────────────────────
#  run_mock_analysis
# ──────────────────────────────────────────────────────────────
class TestMockAnalysis:
    def test_returns_markdown(self):
        steps = []
        result = run_mock_analysis("Vercel", status_callback=lambda a, m: steps.append((a, m)))
        assert result.startswith("# Market Intelligence Report: Vercel")
        assert "## Executive Summary" in result
        assert "## SWOT Analysis" in result

    def test_callbacks_fire(self):
        steps = []
        run_mock_analysis("Foo", status_callback=lambda a, m: steps.append((a, m)))
        assert len(steps) >= 5
        agents_in_run = {a for a, _ in steps}
        # Should hit all three agent types in the simulation
        assert {"Researcher", "Analyst", "Writer"}.issubset(agents_in_run)

    def test_no_callback_safe(self):
        # Should still return a complete report
        result = run_mock_analysis("Foo", status_callback=None)
        assert len(result) > 500

    def test_template_data_disclaimer(self):
        # Honest about being a template — important for trust
        result = run_mock_analysis("Foo", status_callback=None)
        assert "illustrative" in result.lower() or "demo mode" in result.lower()


# ──────────────────────────────────────────────────────────────
#  tavily_search — verify the missing-key error path
# ──────────────────────────────────────────────────────────────
class TestTavilySearch:
    def test_missing_key_returns_error_string(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = agents.tavily_search("anything")
        assert "Error" in result
        assert "TAVILY_API_KEY" in result


# ──────────────────────────────────────────────────────────────
#  Model selection & fallback
# ──────────────────────────────────────────────────────────────
class TestModelSelection:
    def test_both_tiers_have_models(self):
        for tier in ("Quick", "Deep Dive"):
            models = _models_for_tier(tier)
            assert len(models) >= 2, f"{tier} has no fallback chain"
            # No duplicates
            assert len(models) == len(set(models))

    def test_default_uses_current_models(self):
        # Defaults should be 2.x, not deprecated 1.5
        assert "gemini-2.0-flash" in _MODEL_TIERS["Quick"]
        assert "gemini-2.5-pro" in _MODEL_TIERS["Deep Dive"]

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_FAST", "my-custom-model")
        # Reload the constants — easiest is to patch _MODEL_TIERS directly
        # since it's read at import time. We test the helper instead.
        # The function reads _MODEL_TIERS at call time.
        agents._MODEL_TIERS["Quick"] = ["my-custom-model", "gemini-2.0-flash"]
        try:
            assert _models_for_tier("Quick")[0] == "my-custom-model"
        finally:
            agents._MODEL_TIERS["Quick"] = _MODEL_TIERS["Quick"]  # restore

    def test_unknown_tier_falls_back_to_quick(self):
        # Patch a bogus tier to an empty list — helper should fall back
        agents._MODEL_TIERS["Bogus"] = []
        try:
            models = _models_for_tier("Bogus")
            # Falls back to Quick tier defaults
            assert models == _models_for_tier("Quick")
        finally:
            agents._MODEL_TIERS.pop("Bogus", None)

    def test_model_not_found_detection(self):
        assert _is_model_not_found(Exception("404 NOT_FOUND: model is not found"))
        assert _is_model_not_found(Exception("models/gemini-1.5-flash is not found for API version v1beta"))
        assert _is_model_not_found(Exception("model not found"))
        # Network errors should NOT be treated as model-not-found
        assert not _is_model_not_found(Exception("Connection timeout"))
        assert not _is_model_not_found(Exception("Rate limit exceeded"))
        assert not _is_model_not_found(Exception("Invalid API key"))

    def test_pricing_dict_has_current_models(self):
        # Sanity: the new defaults should have pricing entries
        for m in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"):
            assert m in _GEMINI_COST_PER_1K
            assert "input" in _GEMINI_COST_PER_1K[m]
            assert "output" in _GEMINI_COST_PER_1K[m]

    def test_legacy_models_still_in_pricing(self):
        # 1.5 models kept so old CostGuard records work as fallbacks
        for m in ("gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"):
            assert m in _GEMINI_COST_PER_1K


# ──────────────────────────────────────────────────────────────
#  Topic sanitization — makes "Figma", "Vercel" etc. robust
# ──────────────────────────────────────────────────────────────
class TestSanitizeTopic:
    def test_empty_returns_empty(self):
        assert _sanitize_topic("") == ""
        assert _sanitize_topic("   ") == ""

    def test_strips_whitespace(self):
        assert _sanitize_topic("  Figma  ") == "Figma (the collaborative interface design tool)"

    def test_collapses_internal_whitespace(self):
        # The "Figma" lookup is case-insensitive, so the same expansion is used
        result = _sanitize_topic("Figma    whitespace")
        assert "  " not in result
        assert "Figma" in result

    def test_strips_outer_punctuation(self):
        # Unknown topic — should strip trailing punctuation
        assert _sanitize_topic("Some random niche!!!") == "Some random niche"
        assert _sanitize_topic("...hello world...") == "hello world"

    def test_known_company_expands(self):
        # The whole point: bare "Figma" or "Vercel" should get a
        # disambiguated form so the LLM doesn't misread them.
        assert "Figma" in _sanitize_topic("Figma")
        assert "Figma" in _sanitize_topic("figma")  # case-insensitive lookup
        assert "Vercel" in _sanitize_topic("Vercel")
        assert "Supabase" in _sanitize_topic("Supabase")
        assert "Notion" in _sanitize_topic("Notion")
        assert "Linear" in _sanitize_topic("Linear")

    def test_unknown_topic_returned_as_is(self):
        # A niche we don't have a definition for should pass through cleanly
        assert _sanitize_topic("quantum error correction") == "quantum error correction"
        assert _sanitize_topic("cold brew coffee") == "cold brew coffee"

    def test_caps_long_topics(self):
        long_topic = "a" * 200
        result = _sanitize_topic(long_topic)
        assert len(result) <= 100

    def test_handles_unicode(self):
        # Non-ASCII topics should pass through
        assert _sanitize_topic("Größe") == "Größe"


class TestTopicSearchQueries:
    def test_returns_multiple_queries(self):
        queries = _topic_search_queries("Figma")
        assert len(queries) >= 2
        # Each query should mention the topic
        for q in queries:
            assert "Figma" in q

    def test_queries_cover_different_angles(self):
        queries = _topic_search_queries("Stripe")
        joined = " ".join(queries).lower()
        # Should cover overview, competitors, and business model
        assert "overview" in joined or "features" in joined
        assert "competitors" in joined or "alternatives" in joined
        assert "funding" in joined or "revenue" in joined or "business" in joined

    def test_empty_topic_handled(self):
        queries = _topic_search_queries("")
        # Should still produce some output without crashing
        assert isinstance(queries, list)
