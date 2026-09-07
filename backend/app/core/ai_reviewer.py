"""
AI decision-review layer.

Important design point: the AI does NOT generate the trading signal or the
price data — it reviews a signal that was already produced deterministically
by core/indicators.py, and it never overrides core/risk.py. Its job is to
add interpretive context (does this signal make sense given the trend? what
could go wrong?) the way a second pair of eyes on a trading desk would.

Supports two providers, selected via the AI_PROVIDER env var:
  - "anthropic" (default): direct Anthropic API, needs ANTHROPIC_API_KEY
  - "bedrock": AWS Bedrock. Credentials are resolved explicitly via a
    boto3.Session (supporting AWS_PROFILE, including SSO/assume-role
    profiles defined in ~/.aws/config -- set AWS_SDK_LOAD_CONFIG=1 for
    those to be read at all) rather than left to the Anthropic SDK's
    internal lazy resolution, so failures are visible and debuggable
    instead of silently falling back.

If no credentials are available for the selected provider, or the live call
fails for any reason, falls back to a deterministic rule-based explanation
so the app is fully runnable out of the box either way.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from app.core.indicators import QuantSignal

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
# Bedrock model IDs are region/account-specific and dated (not the same
# naming as the direct API) -- verify the exact ID enabled for your account
# under Bedrock > Model access, and override via BEDROCK_MODEL_ID if this
# default isn't available to you.
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"


@dataclass
class AIReview:
    narrative: str
    confidence: float  # 0.0 - 1.0, AI's confidence in the quant signal
    risk_flags: list[str]
    source: str  # "claude_anthropic" / "claude_bedrock" / "rule_based_fallback"


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
the JSON object, nothing else, with no markdown code fences."""


def _strip_markdown_fences(text: str) -> str:
    """Claude occasionally wraps JSON replies in ```json ... ``` even when
    told not to. Strip that before parsing rather than failing and falling
    back on a response that was actually fine."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


class AIReviewer:
    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        aws_region: str | None = None,
    ):
        self.provider = (provider or os.environ.get("AI_PROVIDER", "anthropic")).lower()
        self.model = model
        self._client = None

        if anthropic is None:
            logger.warning("Anthropic SDK not installed; AI review will use the rule-based fallback.")
            return

        if self.provider == "bedrock":
            self._init_bedrock(aws_region)
        elif self.provider == "anthropic":
            self._init_anthropic(api_key)
        else:
            raise ValueError(f"Unknown AI_PROVIDER '{self.provider}': expected 'anthropic' or 'bedrock'.")

    def _init_anthropic(self, api_key: str | None) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = self.model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info("AI reviewer: direct Anthropic client ready (model=%s).", self.model)
        else:
            logger.info("AI reviewer: no ANTHROPIC_API_KEY set; using rule-based fallback.")

    def _init_bedrock(self, aws_region: str | None) -> None:
        import boto3

        self.model = self.model or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
        region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        profile = os.environ.get("AWS_PROFILE")

        logger.info("AI reviewer: initialising Bedrock client (profile=%s, region=%s, model=%s).",
                    profile, region, self.model)

        try:
            # Resolve credentials explicitly via boto3.Session rather than
            # letting the Anthropic SDK do it lazily -- this respects
            # AWS_PROFILE (including SSO/assume-role profiles defined in
            # ~/.aws/config, which requires AWS_SDK_LOAD_CONFIG=1 to be
            # read at all) and surfaces credential problems immediately
            # instead of an opaque failure on the first API call.
            #
            # Trade-off: this snapshots credentials at startup. For SSO or
            # other short-lived session tokens, credentials can expire in a
            # long-running process -- if you hit auth errors after the app
            # has been up for hours, restart it to pick up a fresh session
            # (or switch to long-lived IAM user keys / an instance role for
            # anything that needs to stay up unattended).
            session = boto3.Session(profile_name=profile, region_name=region)
            creds = session.get_credentials()
            if not creds:
                logger.warning("AI reviewer: no AWS credentials resolved for profile=%s; using rule-based fallback.", profile)
                return

            self._client = anthropic.AnthropicBedrock(
                aws_access_key=creds.access_key,
                aws_secret_key=creds.secret_key,
                aws_session_token=creds.token,
                aws_region=region,
            )
            logger.info("AI reviewer: Bedrock client ready.")
        except Exception:  # noqa: BLE001 - missing/invalid AWS setup falls back gracefully
            logger.exception("AI reviewer: failed to initialise Bedrock client; using rule-based fallback.")
            self._client = None

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
            )
            text = _strip_markdown_fences(text)
            parsed = json.loads(text)
            return AIReview(
                narrative=parsed["narrative"],
                confidence=float(parsed["confidence"]),
                risk_flags=list(parsed.get("risk_flags", [])),
                source=f"claude_{self.provider}",
            )
        except Exception:  # noqa: BLE001 - never let the AI layer crash the app
            logger.exception("AI reviewer: live call failed for %s; using rule-based fallback.", symbol)
            return self._fallback_review(signal, degraded=True)

    @staticmethod
    def _fallback_review(signal: QuantSignal, degraded: bool = False) -> AIReview:
        """Deterministic stand-in used when no provider is configured, or if
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
