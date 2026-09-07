"""
Risk control layer.

Deliberately decoupled from signal generation: no matter how confident the
quant signal or the AI layer is, trades must pass through here. This mirrors
how real trading systems separate "alpha" from "risk" so a bad model can
never bypass position limits or blow through a drawdown ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REDUCED = "REDUCED"  # approved but at a smaller size than requested


@dataclass
class RiskLimits:
    max_position_pct: float = 0.20       # max % of portfolio in a single symbol
    max_portfolio_risk_pct: float = 0.02  # max % of portfolio risked per trade (stop-loss distance)
    max_drawdown_pct: float = 0.15       # circuit breaker: halt new entries beyond this drawdown
    stop_loss_pct: float = 0.05          # default stop distance from entry
    take_profit_pct: float = 0.10        # default target distance from entry
    max_open_positions: int = 8


@dataclass
class RiskCheckResult:
    decision: RiskDecision
    approved_quantity: float
    reasons: list[str] = field(default_factory=list)
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def evaluate_entry(
        self,
        *,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        requested_quantity: float,
        portfolio_equity: float,
        current_position_value: float,
        open_position_count: int,
        current_drawdown_pct: float,
        volatility_pct: float | None = None,
    ) -> RiskCheckResult:
        reasons: list[str] = []

        # 1. Circuit breaker: drawdown too deep, halt new entries entirely.
        if current_drawdown_pct >= self.limits.max_drawdown_pct:
            return RiskCheckResult(
                decision=RiskDecision.REJECTED,
                approved_quantity=0.0,
                reasons=[
                    f"Circuit breaker: portfolio drawdown {current_drawdown_pct:.1%} "
                    f"exceeds max {self.limits.max_drawdown_pct:.1%}. No new entries allowed."
                ],
            )

        # 2. Max open positions.
        if open_position_count >= self.limits.max_open_positions:
            return RiskCheckResult(
                decision=RiskDecision.REJECTED,
                approved_quantity=0.0,
                reasons=[
                    f"Max open positions reached ({open_position_count}/"
                    f"{self.limits.max_open_positions})."
                ],
            )

        # 3. Position sizing: volatility-adjusted, capped by max_position_pct
        #    and by max_portfolio_risk_pct given the stop-loss distance.
        stop_distance_pct = self.limits.stop_loss_pct
        if volatility_pct:
            # Widen the stop for volatile names so it isn't shaken out by noise,
            # but never below the configured floor.
            stop_distance_pct = max(self.limits.stop_loss_pct, volatility_pct / 100 * 2)

        risk_dollars = portfolio_equity * self.limits.max_portfolio_risk_pct
        max_qty_by_risk = risk_dollars / (price * stop_distance_pct) if price > 0 else 0

        max_position_dollars = portfolio_equity * self.limits.max_position_pct
        available_position_dollars = max(0.0, max_position_dollars - current_position_value)
        max_qty_by_exposure = available_position_dollars / price if price > 0 else 0

        approved_quantity = min(requested_quantity, max_qty_by_risk, max_qty_by_exposure)
        approved_quantity = max(0.0, round(approved_quantity, 4))

        if approved_quantity <= 0:
            return RiskCheckResult(
                decision=RiskDecision.REJECTED,
                approved_quantity=0.0,
                reasons=[
                    "No capacity: position or per-trade risk limit already reached "
                    f"for {symbol}."
                ],
            )

        decision = RiskDecision.APPROVED
        if approved_quantity < requested_quantity:
            decision = RiskDecision.REDUCED
            reasons.append(
                f"Requested {requested_quantity}, approved {approved_quantity} "
                f"(capped by risk/exposure limits)."
            )

        if side.upper() == "BUY":
            stop_loss_price = round(price * (1 - stop_distance_pct), 2)
            take_profit_price = round(price * (1 + self.limits.take_profit_pct), 2)
        else:
            stop_loss_price = round(price * (1 + stop_distance_pct), 2)
            take_profit_price = round(price * (1 - self.limits.take_profit_pct), 2)

        reasons.append(
            f"Stop-loss set at {stop_distance_pct:.1%} "
            f"({'volatility-adjusted' if volatility_pct else 'default'})."
        )

        return RiskCheckResult(
            decision=decision,
            approved_quantity=approved_quantity,
            reasons=reasons,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )
