"""Tests for cost tracker."""
import pytest
from saga.cost_tracker import CostTracker, UsageRecord, get_pricing, PRICING


class TestPricing:
    def test_exact_model_match(self):
        p = get_pricing("claude-haiku-4-5-20251001")
        assert p["input"] == 0.80
        assert p["cache_read"] == 0.08

    def test_prefix_fallback(self):
        p = get_pricing("claude-haiku-some-future-version")
        assert p["input"] == 0.80

    def test_unknown_model_returns_zero(self):
        p = get_pricing("totally-unknown-model-xyz")
        assert p["input"] == 0.0
        assert p["output"] == 0.0

    def test_gemini_free_tier(self):
        p = get_pricing("gemini-2.5-flash-lite")
        assert p["input"] == 0.0
        assert p["output"] == 0.0


class TestCostCalculation:
    def setup_method(self):
        self.tracker = CostTracker.__new__(CostTracker)

    def test_basic_cost(self):
        cost, savings = self.tracker.calculate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=10000,
            output_tokens=1000,
            cache_read_tokens=0,
            cache_create_tokens=0,
        )
        # input: 10000/1M * 0.80 = 0.008
        # output: 1000/1M * 4.0 = 0.004
        assert cost == pytest.approx(0.012, abs=0.001)
        assert savings == 0.0

    def test_cache_savings(self):
        cost, savings = self.tracker.calculate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=10000,
            output_tokens=1000,
            cache_read_tokens=8000,
            cache_create_tokens=0,
        )
        # cache_read saves: 8000/1M * (0.80 - 0.08) = 0.00576
        assert savings == pytest.approx(0.00576, abs=0.0001)
        # cost should be less than without cache
        assert cost < 0.012

    def test_cache_create_cost(self):
        cost, savings = self.tracker.calculate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=10000,
            output_tokens=1000,
            cache_read_tokens=0,
            cache_create_tokens=5000,
        )
        # cache_create: 5000/1M * 1.0 = 0.005 (1.25x of input)
        assert cost > 0.012  # More expensive due to cache creation

    def test_zero_tokens(self):
        cost, savings = self.tracker.calculate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=0, output_tokens=0,
        )
        assert cost == 0.0
        assert savings == 0.0

    def test_sonnet_pricing(self):
        cost, savings = self.tracker.calculate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=100000,
            output_tokens=5000,
            cache_read_tokens=80000,
        )
        # Sonnet is much more expensive
        # savings from cache: 80000/1M * (3.0 - 0.30) = 0.216
        assert savings == pytest.approx(0.216, abs=0.01)


class TestUsageRecord:
    def test_defaults(self):
        r = UsageRecord()
        assert r.model == ""
        assert r.input_tokens == 0
        assert r.cost_usd == 0.0

    def test_with_values(self):
        r = UsageRecord(
            model="claude-haiku-4-5-20251001",
            input_tokens=1000,
            output_tokens=500,
            session_id="test-sess",
            call_type="main",
        )
        assert r.model == "claude-haiku-4-5-20251001"
        assert r.session_id == "test-sess"
