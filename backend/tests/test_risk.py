from app.core.risk import RiskDecision, RiskEngine, RiskLimits


def make_engine(**overrides):
    limits = RiskLimits(**overrides) if overrides else RiskLimits()
    return RiskEngine(limits)


def test_circuit_breaker_rejects_when_drawdown_exceeded():
    engine = make_engine(max_drawdown_pct=0.10)
    result = engine.evaluate_entry(
        symbol="AAPL",
        side="BUY",
        price=100.0,
        requested_quantity=10,
        portfolio_equity=90_000,
        current_position_value=0,
        open_position_count=0,
        current_drawdown_pct=0.15,
    )
    assert result.decision == RiskDecision.REJECTED
    assert result.approved_quantity == 0
    assert "Circuit breaker" in result.reasons[0]


def test_rejects_when_max_open_positions_reached():
    engine = make_engine(max_open_positions=3)
    result = engine.evaluate_entry(
        symbol="AAPL",
        side="BUY",
        price=100.0,
        requested_quantity=10,
        portfolio_equity=100_000,
        current_position_value=0,
        open_position_count=3,
        current_drawdown_pct=0.0,
    )
    assert result.decision == RiskDecision.REJECTED
    assert "Max open positions" in result.reasons[0]


def test_position_size_capped_by_exposure_limit():
    engine = make_engine(max_position_pct=0.10, max_portfolio_risk_pct=0.50)
    result = engine.evaluate_entry(
        symbol="AAPL",
        side="BUY",
        price=100.0,
        requested_quantity=1000,  # way more than 10% of equity allows
        portfolio_equity=100_000,
        current_position_value=0,
        open_position_count=0,
        current_drawdown_pct=0.0,
    )
    assert result.decision in (RiskDecision.APPROVED, RiskDecision.REDUCED)
    # 10% of 100,000 = 10,000 / price 100 = 100 shares max
    assert result.approved_quantity <= 100


def test_position_size_capped_by_per_trade_risk_limit():
    engine = make_engine(max_position_pct=0.90, max_portfolio_risk_pct=0.01, stop_loss_pct=0.05)
    result = engine.evaluate_entry(
        symbol="AAPL",
        side="BUY",
        price=100.0,
        requested_quantity=1000,
        portfolio_equity=100_000,
        current_position_value=0,
        open_position_count=0,
        current_drawdown_pct=0.0,
    )
    # risk_dollars = 100,000 * 0.01 = 1,000; stop distance 5% of price 100 = 5
    # max_qty_by_risk = 1000 / (100 * 0.05) = 200
    assert result.approved_quantity <= 200


def test_stop_and_target_prices_set_correctly_for_buy():
    engine = make_engine(stop_loss_pct=0.05, take_profit_pct=0.10)
    result = engine.evaluate_entry(
        symbol="AAPL",
        side="BUY",
        price=100.0,
        requested_quantity=10,
        portfolio_equity=100_000,
        current_position_value=0,
        open_position_count=0,
        current_drawdown_pct=0.0,
    )
    assert result.stop_loss_price == 95.0
    assert result.take_profit_price == 110.0


def test_no_capacity_when_position_limit_already_filled():
    engine = make_engine(max_position_pct=0.10)
    result = engine.evaluate_entry(
        symbol="AAPL",
        side="BUY",
        price=100.0,
        requested_quantity=10,
        portfolio_equity=100_000,
        current_position_value=10_000,  # already at the 10% cap
        open_position_count=1,
        current_drawdown_pct=0.0,
    )
    assert result.decision == RiskDecision.REJECTED
    assert result.approved_quantity == 0
