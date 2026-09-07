# AI Trading Simulator

A paper-trading simulator that separates three concerns real trading systems
keep separate: **signal generation**, **AI-assisted decision review**, and
**risk control** — then ties them together with a backtester and a live
dashboard.

Built as an engineering portfolio project to demonstrate AI/data-systems
work applied to market data, trading signals, and risk logic. **No real
money, no live orders, no financial advice** — this is a simulator.

## Why it's structured this way

Most "AI trading bot" demos let a single model both invent a trading idea
and size the trade. That's not how real systems are built, and it's not
resilient: if the model is wrong, nothing stops it. This project deliberately
keeps three layers independent, in a strict pipeline:

```
   market data  →  quant signal  →  AI review  →  risk gate  →  execution
   (yfinance)      (indicators)     (Claude)       (limits)      (paper)
```

1. **Quant signal (`app/core/indicators.py`)** — deterministic technical
   indicators (SMA crossover, RSI, MACD) produce a scored BUY / SELL / HOLD
   signal with explicit, listed reasons. No AI involved. Fully unit tested.
2. **AI review (`app/core/ai_reviewer.py`)** — Claude reviews the *already
   generated* signal (it does not invent a new one) and returns a plain-English
   rationale, a confidence score, and risk flags — the way a second pair of
   eyes on a desk would sanity-check a model's output. Supports either the
   direct Anthropic API or AWS Bedrock, selected via `AI_PROVIDER` (see
   [AI provider setup](#ai-provider-setup)). If no credentials are
   configured for the selected provider, or the call fails, this falls back
   to a deterministic rule-based explanation so the app is always fully
   runnable either way.
3. **Risk gate (`app/core/risk.py`)** — independent of both of the above.
   Enforces max position size, per-trade risk (stop-loss-based sizing),
   a portfolio drawdown circuit breaker, and a max open-positions limit.
   No signal, however confident, can bypass this layer.
4. **Execution (`app/core/portfolio.py`)** — a paper-trading ledger that only
   acts on what the risk engine approved.
5. **Backtester (`app/core/backtester.py`)** — replays this exact same
   pipeline over historical bars (not a separate "backtest-only" code path),
   producing an equity curve, trade log, and performance metrics (total
   return, max drawdown, Sharpe ratio).

## Stack

- **Backend:** FastAPI, pandas/numpy for indicators, yfinance for market data,
  Anthropic SDK for the AI review layer
- **Frontend:** vanilla HTML/CSS/JS dashboard (no build step), Chart.js for
  price and equity charts
- **Tests:** pytest, covering the signal engine and risk engine in isolation

## Running it

**Note** this was specifcally tested on **Python 3.13.13**

```bash
cd backend
python -m venv venv && source venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt

cp .env.example .env   # optional: configure an AI provider, see below

uvicorn app.main:app --reload
```

Then open `http://localhost:8000`.

The app runs fully offline / without any AI credentials at all — the AI
review layer degrades gracefully to a deterministic rule-based explanation
if no provider is configured or a live call fails.

### AI provider setup

Set `AI_PROVIDER` in `.env` to choose how the review layer talks to Claude:

**Direct Anthropic API** (`AI_PROVIDER=anthropic`, the default)
```
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

**AWS Bedrock** (`AI_PROVIDER=bedrock`) — uses your existing AWS credentials
(env vars, `~/.aws/credentials`, SSO, or an instance/task role), no
Anthropic API key needed:
```
AI_PROVIDER=bedrock
AWS_PROFILE=your-aws-profile-name
AWS_REGION=us-east-1
AWS_SDK_LOAD_CONFIG=1
# credentials via env vars, aws configure, or an assumed role -- whatever
# you already use for AWS CLI/SDK access
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0   # verify this against
                                                      # Bedrock > Model access
                                                      # for your account/region
```

If you're already authenticated to AWS locally, you generally only need
`AI_PROVIDER=bedrock`, `AWS_REGION`, and `AWS_PROFILE` — the app resolves
credentials via `boto3.Session(profile_name=...)` at startup, the same way
the AWS CLI would.

Note: credentials are resolved once at startup, not refreshed per-request.
If you're using short-lived SSO credentials and the app has been running for
a long time, restart it to pick up a fresh session.


### Running tests

```bash
cd backend
pytest tests/ -v
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/symbols` | Watchlist |
| `GET /api/quote/{symbol}` | Latest price/change |
| `GET /api/data/{symbol}?period=6mo&interval=1d` | OHLCV history |
| `GET /api/signal/{symbol}` | Quant signal + AI review for the latest bar |
| `POST /api/backtest` | Runs the full pipeline over historical data, returns equity curve, trade log, metrics |

Interactive docs at `/docs` (FastAPI's built-in Swagger UI).

## What this is (and isn't)

This is a portfolio/demo project showing how AI can be layered into an
existing data-driven decision pipeline without replacing the parts that need
to stay deterministic and auditable (signal math, risk limits). It uses paper
cash only, real market data (via yfinance) for realism, and includes explicit
risk controls because that's a first-class requirement in any real trading
system, not an afterthought.

It is **not**:
- Connected to a broker or any real execution venue
- Financial advice or a recommendation to trade any security
- A production-grade risk system — the limits here (position sizing,
  drawdown circuit breaker, stop-loss/take-profit) are illustrative and would
  need significant hardening (slippage/fees modeling, corporate actions,
  multi-asset correlation, etc.) before being anywhere near real capital

## Possible extensions

- Multi-symbol portfolio backtesting with correlation-aware position sizing
- Walk-forward / out-of-sample validation instead of a single backtest window
- Additional signal sources (order-book imbalance, sentiment from news APIs)
- WebSocket streaming for live intraday updates instead of polling
- Persisting backtests to compare strategies over time
