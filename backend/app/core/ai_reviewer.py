"""
AI decision-review layer.

Important design point: the AI does NOT generate the trading signal or the
price data — it reviews a signal that was already produced deterministically
by core/indicators.py, and it never overrides core/risk.py. Its job is to
add interpretive context (does this signal make sense given the trend? what
could go wrong?) the way a second pair of eyes on a trading desk would.

If no ANTHROPIC_API_KEY is configured, falls back to a deterministic
rule-based explanation so the app is fully runnable out of the box.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from app.core.indicators import QuantSignal

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


@dataclass
class AIReview:
    narrative: str
    confidence: float  # 0.0 - 1.0, AI's confidence in the quant signal
    risk_flags: list[str]
    source: str  # "claude" or "rule_based_fallback"


SYSTEM_PROMPT = """You are a risk-aware trading assistant reviewing an \
algorithmic signal for a PAPER TRADING SIMULATOR (no real money, for \
educational/portfolio purposes only).

You will be given a symbol, a quantitative signal (action, score, and the \
technical reasons that produced it), and recent price context. Your job is \
NOT to invent a new signal. Review the given signal and respond ONLY with \
JSON in this exact shape:

{
  "narrative": "2-3 sentence plain-English explanation of what the signal means and whether the technical evidence is convincing",
  "confidence": 0.0-1.0,
  "risk_flags": ["short phrase", "short phrase"]
}

Lower confidence when indicators conflict, history is short, or volatility \
is high. risk_flags should call out things a trader should watch for \
(e.g. "signal conflicts with longer-term trend", "high volatility widens \
stop distance", "single-indicator signal, weak confirmation"). Return only \
the JSON object, nothing else."""


class AIReviewer:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None
        if self.api_key and anthropic is not None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def review(self, symbol: str, signal: QuantSignal, recent_close_prices: list[float]) -> AIReview:
        if self._client is None:
            return self._fallback_review(signal)

        user_content = json.dumps(
            {
                "symbol": symbol,
                "action": signal.action.value,
                "score": signal.score,
                "reasons": signal.reasons,
                "indicators": signal.indicators,
                "recent_close_prices": recent_close_prices[-10:],
            }
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            parsed = json.loads(text)
            return AIReview(
                narrative=parsed["narrative"],
                confidence=float(parsed["confidence"]),
                risk_flags=list(parsed.get("risk_flags", [])),
                source="claude",
            )
        except Exception:  # noqa: BLE001 - never let the AI layer crash the app
            return self._fallback_review(signal, degraded=True)

    @staticmethod
    def _fallback_review(signal: QuantSignal, degraded: bool = False) -> AIReview:
        """Deterministic stand-in used when no API key is configured, or if
        the live call fails. Keeps the app fully functional offline."""
        confidence = 0.4 + min(abs(signal.score), 1.0) * 0.4
        risk_flags = []
        if len(signal.reasons) <= 1:
            risk_flags.append("Single-indicator signal, weak confirmation.")
        if signal.indicators.get("volatility_pct", 0) and signal.indicators["volatility_pct"] > 3:
            risk_flags.append("Elevated volatility widens expected stop distance.")

        prefix = "[offline rule-based review] " if not degraded else "[AI call failed, fallback] "
        narrative = (
            f"{prefix}{signal.action.value} signal (score {signal.score:+.2f}) "
            f"based on: {'; '.join(signal.reasons)}."
        )
        return AIReview(
            narrative=narrative,
            confidence=round(confidence, 2),
            risk_flags=risk_flags,
            source="rule_based_fallback",
        )
