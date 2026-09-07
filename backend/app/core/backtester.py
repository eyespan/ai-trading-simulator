"""
Backtesting engine.

Wires the full pipeline together:
  historical bars -> quant signal -> AI review -> risk gate -> execution

This is the same pipeline the live simulation endpoint uses, just replayed
over history instead of the latest bar. Keeping one pipeline for both modes
avoids "backtest vs live" logic drift, which is a classic real-world trading
system bug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core.ai_reviewer import AIReviewer
from app.core.indicators import SignalAction, generate_signal
from app.core.portfolio import Portfolio
from app.core.risk import RiskEngine


@dataclass
class BacktestResult:
    equity_curve: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    ai_reviews_sample: list[dict] = field(default_factory=list)


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    *,
    starting_cash: float = 100_000.0,
    use_ai_review: bool = True,
    ai_confidence_floor: float = 0.35,
    min_bars: int = 50,
) -> BacktestResult:
    portfolio = Portfolio(starting_cash=starting_cash, cash=starting_cash)
    risk_engine = RiskEngine()
    reviewer = AIReviewer() if use_ai_review else None

    equity_curve: list[dict] = []
    ai_reviews_sample: list[dict] = []

    for i in range(min_bars, len(df)):
        window = df.iloc[: i + 1]
        bar = window.iloc[-1]
        price = float(bar["Close"])
        current_prices = {symbol: price}

        portfolio.check_stops(current_prices)

        signal = generate_signal(window, min_bars=min_bars)

        confidence = 1.0
        review_source = "quant_only"
        if reviewer is not None and signal.action != SignalAction.HOLD:
            review = reviewer.review(
                symbol, signal, window["Close"].tail(10).tolist()
            )
            confidence = review.confidence
            review_source = review.source
            if len(ai_reviews_sample) < 5:
                ai_reviews_sample.append(
                    {
                        "date": str(window.index[-1].date()),
                        "action": signal.action.value,
                        "narrative": review.narrative,
                        "confidence": review.confidence,
                        "risk_flags": review.risk_flags,
                        "source": review.source,
                    }
                )

        if signal.action != SignalAction.HOLD and confidence >= ai_confidence_floor:
            eq = portfolio.equity(current_prices)
            dd = portfolio.drawdown_pct(current_prices)
            existing_value = portfolio.position_value(symbol, price)
            open_count = len(portfolio.positions)

            # Simple sizing intent: risk engine will clamp this to real limits.
            intended_qty = (eq * 0.10) / price if price > 0 else 0

            result = risk_engine.evaluate_entry(
                symbol=symbol,
                side=signal.action.value,
                price=price,
                requested_quantity=intended_qty,
                portfolio_equity=eq,
                current_position_value=existing_value,
                open_position_count=open_count,
                current_drawdown_pct=dd,
                volatility_pct=signal.indicators.get("volatility_pct"),
            )

            side = "BUY" if signal.action == SignalAction.BUY else "SELL"
            if side == "SELL" and symbol not in portfolio.positions:
                pass  # nothing to sell / no short-selling in this simulator
            elif result.approved_quantity > 0:
                portfolio.apply_fill(
                    symbol=symbol,
                    side=side,
                    quantity=result.approved_quantity,
                    price=price,
                    decision=result.decision,
                    reasons=signal.reasons + result.reasons,
                    stop_loss_price=result.stop_loss_price,
                    take_profit_price=result.take_profit_price,
                    timestamp=str(window.index[-1]),
                )

        equity_curve.append(
            {"date": str(window.index[-1].date()), "equity": round(portfolio.equity(current_prices), 2)}
        )

    final_prices = {symbol: float(df["Close"].iloc[-1])}
    metrics = _compute_metrics(equity_curve, portfolio, starting_cash)

    return BacktestResult(
        equity_curve=equity_curve,
        trades=[
            {
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "decision": t.decision,
                "reasons": t.reasons,
            }
            for t in portfolio.trade_log
        ],
        metrics=metrics,
        ai_reviews_sample=ai_reviews_sample,
    )


def _compute_metrics(equity_curve: list[dict], portfolio: Portfolio, starting_cash: float) -> dict:
    if not equity_curve:
        return {}

    equities = np.array([e["equity"] for e in equity_curve], dtype=float)
    returns = np.diff(equities) / equities[:-1]
    returns = returns[np.isfinite(returns)]

    total_return_pct = ((equities[-1] - starting_cash) / starting_cash) * 100

    running_max = np.maximum.accumulate(equities)
    drawdowns = (running_max - equities) / running_max
    max_drawdown_pct = float(np.max(drawdowns) * 100) if len(drawdowns) else 0.0

    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 1e-9:
        sharpe = float(np.mean(returns) / np.std(returns) * math.sqrt(252))

    return {
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": len(portfolio.trade_log),
        "final_equity": round(float(equities[-1]), 2),
        "starting_cash": starting_cash,
    }
