import pytest

from app.core.ai_reviewer import AIReviewer
from app.core.indicators import QuantSignal, SignalAction


def test_anthropic_provider_without_key_falls_back_to_rule_based():
    reviewer = AIReviewer(provider="anthropic", api_key=None)
    assert reviewer.is_live is False

    signal = QuantSignal(
        action=SignalAction.BUY,
        score=0.5,
        reasons=["SMA20 crossed above SMA50 (bullish golden cross)."],
        indicators={"volatility_pct": 1.2},
    )
    review = reviewer.review("AAPL", signal, [100.0, 101.0, 102.0])
    assert review.source == "rule_based_fallback"
    assert 0.0 <= review.confidence <= 1.0


def test_bedrock_provider_constructs_client_with_region():
    reviewer = AIReviewer(provider="bedrock", aws_region="us-east-1")
    assert reviewer.is_live is True
    assert reviewer.model  # a default model id is set


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        AIReviewer(provider="not-a-real-provider")


def test_default_provider_is_anthropic_when_unset(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reviewer = AIReviewer()
    assert reviewer.provider == "anthropic"
    assert reviewer.is_live is False  # no key configured in this env
