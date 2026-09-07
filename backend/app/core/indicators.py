"""
Quantitative signal engine.

Pure technical-indicator logic — no AI, no I/O. Deterministic and unit
testable in isolation, which matters: the AI layer should be reviewing this
signal, not generating the numbers itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class QuantSignal:
    action: SignalAction
    score: float  # -1.0 (strong sell) to +1.0 (strong buy)
    reasons: list[str]
    indicators: dict[str, float]


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, 12)
    ema_slow = _ema(series, 26)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds indicator columns to a copy of the OHLCV dataframe."""
    out = df.copy()
    out["sma_20"] = _sma(out["Close"], 20)
    out["sma_50"] = _sma(out["Close"], 50)
    out["ema_12"] = _ema(out["Close"], 12)
    out["rsi_14"] = _rsi(out["Close"], 14)
    macd_line, signal_line, hist = _macd(out["Close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["atr_14"] = _atr(out, 14)
    out["volatility_pct"] = (out["atr_14"] / out["Close"]) * 100
    return out


def generate_signal(df: pd.DataFrame, min_bars: int = 50) -> QuantSignal:
    """
    Combines SMA crossover, RSI, and MACD into a single scored signal.

    This is intentionally simple and explainable — a hiring reviewer (or the
    LLM layer above it) can see exactly why a signal fired.
    """
    if len(df) < min_bars:
        return QuantSignal(
            action=SignalAction.HOLD,
            score=0.0,
            reasons=[f"Insufficient history ({len(df)} bars, need {min_bars})."],
            indicators={},
        )

    ind = compute_indicators(df)
    latest = ind.iloc[-1]
    prev = ind.iloc[-2]

    score = 0.0
    reasons: list[str] = []

    # --- Trend: SMA20 vs SMA50 crossover ---
    if pd.notna(latest["sma_20"]) and pd.notna(latest["sma_50"]):
        if latest["sma_20"] > latest["sma_50"] and prev["sma_20"] <= prev["sma_50"]:
            score += 0.4
            reasons.append("SMA20 crossed above SMA50 (bullish golden cross).")
        elif latest["sma_20"] < latest["sma_50"] and prev["sma_20"] >= prev["sma_50"]:
            score -= 0.4
            reasons.append("SMA20 crossed below SMA50 (bearish death cross).")
        elif latest["sma_20"] > latest["sma_50"]:
            score += 0.15
            reasons.append("Price trend remains bullish (SMA20 > SMA50).")
        else:
            score -= 0.15
            reasons.append("Price trend remains bearish (SMA20 < SMA50).")

    # --- Momentum: RSI ---
    rsi = latest["rsi_14"]
    if pd.notna(rsi):
        if rsi < 30:
            score += 0.3
            reasons.append(f"RSI at {rsi:.1f} indicates oversold conditions.")
        elif rsi > 70:
            score -= 0.3
            reasons.append(f"RSI at {rsi:.1f} indicates overbought conditions.")

    # --- Momentum confirmation: MACD histogram ---
    if pd.notna(latest["macd_hist"]) and pd.notna(prev["macd_hist"]):
        if latest["macd_hist"] > 0 and prev["macd_hist"] <= 0:
            score += 0.3
            reasons.append("MACD histogram turned positive (bullish momentum shift).")
        elif latest["macd_hist"] < 0 and prev["macd_hist"] >= 0:
            score -= 0.3
            reasons.append("MACD histogram turned negative (bearish momentum shift).")

    score = max(-1.0, min(1.0, score))

    if score >= 0.3:
        action = SignalAction.BUY
    elif score <= -0.3:
        action = SignalAction.SELL
    else:
        action = SignalAction.HOLD
        if not reasons:
            reasons.append("No strong directional signal from current indicators.")

    indicators_snapshot = {
        "close": round(float(latest["Close"]), 2),
        "sma_20": round(float(latest["sma_20"]), 2) if pd.notna(latest["sma_20"]) else None,
        "sma_50": round(float(latest["sma_50"]), 2) if pd.notna(latest["sma_50"]) else None,
        "rsi_14": round(float(rsi), 2) if pd.notna(rsi) else None,
        "macd_hist": round(float(latest["macd_hist"]), 4) if pd.notna(latest["macd_hist"]) else None,
        "volatility_pct": round(float(latest["volatility_pct"]), 2) if pd.notna(latest["volatility_pct"]) else None,
    }

    return QuantSignal(action=action, score=round(score, 3), reasons=reasons, indicators=indicators_snapshot)
