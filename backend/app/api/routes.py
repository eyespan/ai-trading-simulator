from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.ai_reviewer import AIReviewer
from app.core.backtester import run_backtest
from app.core.indicators import generate_signal
from app.core.market_data import MarketDataError, get_market_data_client
from app.models.schemas import BacktestRequest

router = APIRouter()

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "SPY"]


@router.get("/symbols")
def list_symbols():
    return {"symbols": WATCHLIST}


@router.get("/data/{symbol}")
def get_price_history(symbol: str, period: str = "6mo", interval: str = "1d"):
    client = get_market_data_client()
    try:
        df = client.get_history(symbol.upper(), period=period, interval=interval)
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    bars = [
        {
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }
        for idx, row in df.iterrows()
    ]
    return {"symbol": symbol.upper(), "bars": bars}


@router.get("/quote/{symbol}")
def get_quote(symbol: str):
    client = get_market_data_client()
    try:
        return client.get_quote_summary(symbol.upper())
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/signal/{symbol}")
def get_signal(symbol: str, period: str = "6mo", interval: str = "1d"):
    client = get_market_data_client()
    try:
        df = client.get_history(symbol.upper(), period=period, interval=interval)
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    signal = generate_signal(df)
    reviewer = AIReviewer()
    review = reviewer.review(symbol.upper(), signal, df["Close"].tail(10).tolist())

    return {
        "symbol": symbol.upper(),
        "action": signal.action.value,
        "score": signal.score,
        "reasons": signal.reasons,
        "indicators": signal.indicators,
        "ai_narrative": review.narrative,
        "ai_confidence": review.confidence,
        "ai_risk_flags": review.risk_flags,
        "ai_source": review.source,
        "ai_live": reviewer.is_live,
    }


@router.post("/backtest")
def post_backtest(request: BacktestRequest):
    client = get_market_data_client()
    try:
        df = client.get_history(request.symbol.upper(), period=request.period, interval=request.interval)
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = run_backtest(
        request.symbol.upper(),
        df,
        starting_cash=request.starting_cash,
        use_ai_review=request.use_ai_review,
    )
    return {
        "symbol": request.symbol.upper(),
        "equity_curve": result.equity_curve,
        "trades": result.trades,
        "metrics": result.metrics,
        "ai_reviews_sample": result.ai_reviews_sample,
    }
