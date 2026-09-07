import numpy as np
import pandas as pd
import pytest

from app.core.indicators import SignalAction, generate_signal


def _make_trend_df(n=80, start=100.0, daily_return_pct=0.006, noise=0.003, seed=1):
    """Generates a synthetic OHLCV series using multiplicative (percentage)
    daily returns so prices stay realistic (and positive) even under a
    sustained trend, unlike additive drift."""
    rng = np.random.default_rng(seed)
    daily_returns = daily_return_pct + rng.normal(0, noise, n)
    closes = start * np.cumprod(1 + daily_returns)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": closes - 0.2,
            "High": closes + 0.5,
            "Low": closes - 0.5,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )
    return df


def test_insufficient_history_returns_hold():
    df = _make_trend_df(n=10)
    signal = generate_signal(df, min_bars=50)
    assert signal.action == SignalAction.HOLD
    assert "Insufficient history" in signal.reasons[0]


def test_strong_uptrend_biases_toward_buy():
    df = _make_trend_df(n=100, daily_return_pct=0.012, noise=0.002)
    signal = generate_signal(df)
    assert signal.score > 0
    assert signal.action in (SignalAction.BUY, SignalAction.HOLD)


def test_downtrend_with_recent_stabilization_biases_toward_sell():
    # A decline followed by a flat stretch: SMA50 (slow) stays above SMA20
    # (bearish trend, since it lags the decline) while RSI normalizes back
    # toward neutral over the flat period rather than bottoming out. This
    # isolates the trend component without triggering the RSI mean-reversion
    # counter-signal tested separately above.
    rng = np.random.default_rng(2)
    decline = 60 * np.cumprod(1 + (-0.01 + rng.normal(0, 0.003, 60)))
    flat = decline[-1] * np.cumprod(1 + rng.normal(0, 0.003, 40))
    closes = np.concatenate([decline, flat])
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame(
        {
            "Open": closes - 0.1,
            "High": closes + 0.3,
            "Low": closes - 0.3,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 5_000_000, len(closes)),
        },
        index=dates,
    )
    signal = generate_signal(df)
    assert signal.score < 0
    assert signal.action in (SignalAction.SELL, SignalAction.HOLD)


def test_score_is_bounded():
    df = _make_trend_df(n=100, daily_return_pct=0.03, noise=0.001)
    signal = generate_signal(df)
    assert -1.0 <= signal.score <= 1.0


def test_indicators_snapshot_has_expected_keys():
    df = _make_trend_df(n=100)
    signal = generate_signal(df)
    for key in ("close", "sma_20", "sma_50", "rsi_14", "macd_hist", "volatility_pct"):
        assert key in signal.indicators
