"""
AI decision-review layer.

Important design point: the AI does NOT generate the trading signal or the
price data — it reviews a signal that was already produced deterministically
by core/indicators.py, and it never overrides core/risk.py. Its job is to
add interpretive context (does this signal make sense given the trend? what
could go wrong?) the way a second pair of eyes on a trading desk would.

Supports two providers, selected via the AI_PROVIDER env var:
  - "anthropic" (default): direct Anthropic API, needs ANTHROPIC_API_KEY
  - "bedrock": AWS Bedrock, using your normal AWS credential chain
    (env vars, ~/.aws/credentials, instance/task role, SSO, etc.) — no
    Anthropic API key needed

If no credentials are available for the selected provider, or the live call
fails for any reason, falls back to a deterministic rule-based explanation
so the app is fully runnable out of the box either way.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from dataclasses import dataclass

from app.core.indicators import QuantSignal

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
# Bedrock model IDs are region/account-specific and change as Anthropic ships
# new versions — verify the exact ID available to you in the Bedrock console
# (Model access page) and override via BEDROCK_MODEL_ID if this default isn't
# enabled for your account.
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-6-v1:0"


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
            print("⚠️ [AIReviewer] Anthropic SDK is not installed.", file=sys.stderr)
            return

        if self.provider == "bedrock":
            self.model = self.model or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
            region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

             # Explicitly load via boto3 session to capture AWS_PROFILE accurately
            import boto3
            profile = os.environ.get("AWS_PROFILE")
            
            print(f"🔄 [AIReviewer] Initialising Bedrock client (Profile: {profile}, Region: {region}, Model: {self.model})...")

            
            try:
                # AnthropicBedrock signs requests using boto3's normal
                # credential chain — nothing to pass explicitly beyond region
                # if you already have AWS credentials configured (env vars,
                # ~/.aws/credentials, SSO, or an instance/task role).
                session = boto3.Session(profile_name=profile, region_name=region)
                creds = session.get_credentials()
                
                if not creds:
                    print("⚠️ [AIReviewer] No credentials returned from boto3 session profile.", file=sys.stderr)
                    return

                self._client = anthropic.AnthropicBedrock(
                    aws_access_key=creds.access_key,
                    aws_secret_key=creds.secret_key,
                    aws_session_token=creds.token,
                    aws_region=region
                )
                print("✅ [AIReviewer] Bedrock SDK client generated successfully.")
            except Exception as e:
                print(f"❌ [AIReviewer] Bedrock Client Setup Error: {e}", file=sys.stderr)
                traceback.print_exc()
                self._client = None

        elif self.provider == "anthropic":
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.model = self.model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
                print(f"✅ [AIReviewer] Direct Anthropic client initialised using model {self.model}.")
        else:
            raise ValueError(f"Unknown AI_PROVIDER '{self.provider}': expected 'anthropic' or 'bedrock'.")

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
            
            # --- ADD THIS LINE TO DEBUG EMPTY OR WEIRD RESPONSES ---
            print(f"🔍 [AIReviewer] Raw Model Response Text: {repr(text)}")

            # --- ADD THIS CLEANUP SECTION TO STRIP MARKDOWN WRAPPERS ---
            if text.startswith("```"):
                # Strip leading ```json or ``` and trailing ```
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            # -----------------------------------------------------------

            

            parsed = json.loads(text)
            return AIReview(
                narrative=parsed["narrative"],
                confidence=float(parsed["confidence"]),
                risk_flags=list(parsed.get("risk_flags", [])),
                source=f"claude_{self.provider}",
            )
        except Exception as err:  # noqa: BLE001 - never let the AI layer crash the app
            print(f"❌ [AIReviewer] Live Bedrock API Call Failed: {err}", file=sys.stderr)
            traceback.print_exc()  # Prints full crash trace context to your server log terminal
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
