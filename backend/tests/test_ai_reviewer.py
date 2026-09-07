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


def test_bedrock_provider_constructs_client_with_resolved_credentials(tmp_path, monkeypatch):
    creds_file = tmp_path / "credentials"
    creds_file.write_text(
        "[test-profile]\n"
        "aws_access_key_id = AKIA_TEST_FAKE\n"
        "aws_secret_access_key = test_secret\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds_file))
    monkeypatch.setenv("AWS_PROFILE", "test-profile")

    reviewer = AIReviewer(provider="bedrock", aws_region="us-east-1")
    assert reviewer.is_live is True
    assert reviewer.model  # a default model id is set


def test_bedrock_provider_falls_back_without_resolvable_credentials(monkeypatch):
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/credentials")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/config")

    reviewer = AIReviewer(provider="bedrock", aws_region="us-east-1")
    assert reviewer.is_live is False

    signal = QuantSignal(action=SignalAction.HOLD, score=0.0, reasons=["No signal."], indicators={})
    review = reviewer.review("AAPL", signal, [100.0])
    assert review.source == "rule_based_fallback"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        AIReviewer(provider="not-a-real-provider")


def test_default_provider_is_anthropic_when_unset(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reviewer = AIReviewer()
    assert reviewer.provider == "anthropic"
    assert reviewer.is_live is False  # no key configured in this env
