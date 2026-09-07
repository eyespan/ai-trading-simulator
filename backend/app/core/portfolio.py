"""
Paper trading portfolio: tracks cash, positions, and trade history.

This is the execution layer. It only acts on what the risk engine has
approved — it never re-derives quantity or price itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.core.risk import RiskDecision


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    stop_loss_price: float | None = None
    take_profit_price: float | None = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_entry_price  # revalued externally with live price


@dataclass
class Trade:
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    decision: str
    reasons: list[str]


@dataclass
class Portfolio:
    starting_cash: float = 100_000.0
    cash: float = 100_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    trade_log: list[Trade] = field(default_factory=list)
    peak_equity: float = 100_000.0

    def equity(self, current_prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos.avg_entry_price)
            value += pos.quantity * price
        return value

    def drawdown_pct(self, current_prices: dict[str, float]) -> float:
        eq = self.equity(current_prices)
        self.peak_equity = max(self.peak_equity, eq)
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - eq) / self.peak_equity)

    def position_value(self, symbol: str, current_price: float) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity * current_price if pos else 0.0

    def apply_fill(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        decision: RiskDecision,
        reasons: list[str],
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        timestamp: str | None = None,
    ) -> None:
        if quantity <= 0:
            return

        cost = quantity * price
        if side.upper() == "BUY":
            if cost > self.cash:
                quantity = round(self.cash / price, 4)
                cost = quantity * price
            self.cash -= cost
            existing = self.positions.get(symbol)
            if existing:
                total_qty = existing.quantity + quantity
                existing.avg_entry_price = (
                    (existing.avg_entry_price * existing.quantity) + cost
                ) / total_qty
                existing.quantity = total_qty
                existing.stop_loss_price = stop_loss_price or existing.stop_loss_price
                existing.take_profit_price = take_profit_price or existing.take_profit_price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                )
        else:  # SELL / close
            existing = self.positions.get(symbol)
            if not existing:
                return
            sell_qty = min(quantity, existing.quantity)
            self.cash += sell_qty * price
            existing.quantity -= sell_qty
            if existing.quantity <= 1e-9:
                del self.positions[symbol]

        self.trade_log.append(
            Trade(
                timestamp=timestamp or datetime.utcnow().isoformat(),
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                price=price,
                decision=decision.value,
                reasons=reasons,
            )
        )

    def check_stops(self, current_prices: dict[str, float]) -> list[Trade]:
        """Auto-close any position that has hit its stop-loss or take-profit."""
        closed: list[Trade] = []
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = current_prices.get(symbol)
            if price is None:
                continue
            hit_stop = pos.stop_loss_price and price <= pos.stop_loss_price
            hit_target = pos.take_profit_price and price >= pos.take_profit_price
            if hit_stop or hit_target:
                reason = "Stop-loss triggered." if hit_stop else "Take-profit triggered."
                qty = pos.quantity
                self.apply_fill(
                    symbol=symbol,
                    side="SELL",
                    quantity=qty,
                    price=price,
                    decision=RiskDecision.APPROVED,
                    reasons=[reason],
                )
                closed.append(self.trade_log[-1])
        return closed

    def summary(self, current_prices: dict[str, float]) -> dict:
        eq = self.equity(current_prices)
        return {
            "cash": round(self.cash, 2),
            "equity": round(eq, 2),
            "starting_cash": self.starting_cash,
            "total_return_pct": round(((eq - self.starting_cash) / self.starting_cash) * 100, 2),
            "drawdown_pct": round(self.drawdown_pct(current_prices) * 100, 2),
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_entry_price": round(p.avg_entry_price, 2),
                    "current_price": round(current_prices.get(p.symbol, p.avg_entry_price), 2),
                    "unrealized_pnl": round(
                        (current_prices.get(p.symbol, p.avg_entry_price) - p.avg_entry_price)
                        * p.quantity,
                        2,
                    ),
                    "stop_loss_price": p.stop_loss_price,
                    "take_profit_price": p.take_profit_price,
                }
                for p in self.positions.values()
            ],
            "trade_count": len(self.trade_log),
        }
