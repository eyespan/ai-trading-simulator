"""
Market data ingestion layer.

Wraps yfinance so the rest of the app never talks to a data vendor directly.
Swapping to Alpha Vantage / Polygon / a broker API later means changing only
this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import yfinance as yf


class MarketDataError(Exception):
    """Raised when market data cannot be retrieved or is malformed."""


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketDataClient:
    """Fetches OHLCV data and caches it briefly to avoid hammering the API."""

    def __init__(self, cache_ttl_seconds: int = 60):
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}

    def get_history(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by timestamp with columns:
        Open, High, Low, Close, Volume.
        """
        cache_key = f"{symbol}:{period}:{interval}"
        now = datetime.utcnow()
        cached = self._cache.get(cache_key)
        if cached and (now - cached[0]).total_seconds() < self.cache_ttl_seconds:
            return cached[1]

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise MarketDataError(f"Failed to fetch data for {symbol}: {exc}") from exc

        if df is None or df.empty:
            raise MarketDataError(
                f"No data returned for symbol '{symbol}'. It may be delisted "
                f"or the symbol may be invalid."
            )

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        self._cache[cache_key] = (now, df)
        return df

    def get_latest_price(self, symbol: str) -> float:
        df = self.get_history(symbol, period="5d", interval="1d")
        return float(df["Close"].iloc[-1])

    def get_quote_summary(self, symbol: str) -> dict:
        df = self.get_history(symbol, period="1mo", interval="1d")
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change = float(latest["Close"] - prev["Close"])
        change_pct = (change / prev["Close"]) * 100 if prev["Close"] else 0.0
        return {
            "symbol": symbol,
            "price": round(float(latest["Close"]), 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest["Volume"]),
            "as_of": df.index[-1].isoformat(),
        }


@lru_cache(maxsize=1)
def get_market_data_client() -> MarketDataClient:
    return MarketDataClient()
