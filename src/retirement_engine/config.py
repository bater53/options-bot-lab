# src/retirement_engine/config.py

"""
Global configuration for the retirement engine + retirement simulation.

Engine-level settings = risk caps, symbols, etc.
Simulation settings = starting balance, contributions, returns, withdrawals, inflation.

This keeps your "engine guardrails" AND gives run_retirement_sim.py what it needs.
"""

RETIREMENT_CONFIG = {
    # ---------- Engine / Guardrails ----------
    "mode": "LAB",

    # Notional account size (also used as starting_balance by default)
    "account_size": 25_000.0,

    "max_risk_per_trade": 0.005,  # 0.5% of account per trade
    "max_risk_per_day": 0.02,     # 2% of account per day
    "max_open_trades": 3,

    "symbols": ["SPY"],

    # ---------- Simulation (Retirement) ----------
    # If not provided, run_retirement_sim will default starting_balance to account_size.
    "starting_balance": 25_000.0,

    # Accumulation contributions
    "monthly_contribution": 0.0,

    # Total sim horizon in years
    "years": 30,

    # Returns: choose ONE (monthly preferred)
    # Example: 0.006 ~= 0.6%/mo (~7.4% annual geometric)
    "expected_monthly_return": 0.006,
    # "expected_annual_return": 0.07,

    # Optional volatility for simple Monte Carlo later (not used yet)
    # "monthly_volatility": 0.03,
    
        # ---------- Trading-income model (bot-driven monthly P&L) ----------
    "use_trade_model": True,

    # Repeatable randomness (set None to be different each run)
    "random_seed": 42,

    # How often the bot trades
    "trades_per_week": 6,          # conservative: ~1–2/day
    "weeks_per_month": 4.33,       # average weeks per month

    # Performance assumptions (starting point — we’ll tune)
    "win_rate": 0.65,
    "avg_win_R": 0.70,
    "avg_loss_R": 1.00,            # HARD CAPPED (B foundation)

    # Risk & costs
    "risk_per_trade_pct": 0.0025,  # 0.25% of current balance per trade
    "cost_per_trade_usd": 3.00,    # fees + slippage estimate


    # When do withdrawals begin?
    "retire_after_years": 0,   # start withdrawals after 20 years of contributions

    # Spending in "today dollars" (inflation-adjusted if spend_inflation_adjusted=True)
    "monthly_spend": 0.0,
    "inflation_annual": 0.03,
    "spend_inflation_adjusted": True,

    # Stop contributing once withdrawals start?
    "contribute_during_retirement": False,

    # Output
    "output_path": "data/retirement_engine/sim_results.csv",
    "start_date": None,  # e.g. "2026-01" (optional)
    
    "stop_on_failure": False,   # set False if you want it to always run full 30 years

}
