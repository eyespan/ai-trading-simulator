from __future__ import annotations

from pydantic import BaseModel


class QuoteResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    as_of: str


class SignalResponse(BaseModel):
    symbol: str
    action: str
    score: float
    reasons: list[str]
    indicators: dict
    ai_narrative: str
    ai_confidence: float
    ai_risk_flags: list[str]
    ai_source: str


class BacktestRequest(BaseModel):
    symbol: str
    period: str = "1y"
    interval: str = "1d"
    starting_cash: float = 100_000.0
    use_ai_review: bool = True


class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
