"""
Run a full simulation for the Retirement Engine.

Usage:
    python -m src.retirement_engine.run_retirement_sim
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import random


from src.retirement_engine.config import RETIREMENT_CONFIG
from src.retirement_engine.ruleset import evaluate_ruleset


def _monthly_rate(cfg: Dict[str, Any]) -> float:
    if cfg.get("expected_monthly_return") is not None:
        return float(cfg["expected_monthly_return"])
    if cfg.get("expected_annual_return") is not None:
        annual = float(cfg["expected_annual_return"])
        return (1.0 + annual) ** (1.0 / 12.0) - 1.0
    raise KeyError("Set expected_monthly_return or expected_annual_return in RETIREMENT_CONFIG")


def _start_date(cfg: Dict[str, Any]) -> pd.Timestamp:
    v = cfg.get("start_date")
    if v:
        ts = pd.Timestamp(v)
        return pd.Timestamp(year=ts.year, month=ts.month, day=1)
    now = pd.Timestamp(datetime.now())
    return pd.Timestamp(year=now.year, month=now.month, day=1)


@dataclass(frozen=True)
class SimResult:
    df: pd.DataFrame
    survived: bool
    failure_month: Optional[int]
    final_balance: float
    total_contributed: float
    total_withdrawn: float


def simulate_retirement(
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> SimResult:
    cfg = cfg or RETIREMENT_CONFIG

    start_balance = float(cfg.get("starting_balance", cfg["account_size"]))
    monthly_contrib = float(cfg.get("monthly_contribution", 0.0))
    years = int(cfg.get("years", 30))

    retire_after_years = int(cfg.get("retire_after_years", years))  # default: never retire
    retire_start_month = retire_after_years * 12

    monthly_spend = float(cfg.get("monthly_spend", 0.0))
    inflation_annual = float(cfg.get("inflation_annual", 0.0))
    spend_infl_adj = bool(cfg.get("spend_inflation_adjusted", True))
    contribute_during_retirement = bool(cfg.get("contribute_during_retirement", False))

    stop_on_failure = bool(cfg.get("stop_on_failure", True))

    # If trade-model is off, we fall back to the fixed-return model
    use_trade_model = bool(cfg.get("use_trade_model", False))
    base_monthly_return = _monthly_rate(cfg) if not use_trade_model else 0.0

    start_date = _start_date(cfg)

    # Deterministic randomness (if seed set)
    seed = cfg.get("random_seed", 42)
    rng = random.Random(seed) if seed is not None else random.Random()

    balance = start_balance
    total_contrib = 0.0
    total_withdrawn = 0.0
    total_costs = 0.0
    failure_month: Optional[int] = None

    history = []
    n_months = years * 12

    # Trade model params
    trades_per_week = float(cfg.get("trades_per_week", 0.0))
    weeks_per_month = float(cfg.get("weeks_per_month", 4.33))
    win_rate = float(cfg.get("win_rate", 0.5))
    avg_win_R = float(cfg.get("avg_win_R", 0.7))
    avg_loss_R = float(cfg.get("avg_loss_R", 1.0))
    risk_per_trade_pct = float(cfg.get("risk_per_trade_pct", 0.0025))
    cost_per_trade_usd = float(cfg.get("cost_per_trade_usd", 0.0))

    for m in range(n_months):
        date = start_date + pd.DateOffset(months=m)
        month_num = m + 1

        in_retirement = m >= retire_start_month

        contrib = monthly_contrib if (not in_retirement or contribute_during_retirement) else 0.0

        # Withdrawal (inflation-adjusted spending)
        if in_retirement and monthly_spend > 0:
            years_into_ret = (m - retire_start_month) / 12.0
            infl_factor = (1.0 + inflation_annual) ** years_into_ret if spend_infl_adj else 1.0
            withdrawal_planned = monthly_spend * infl_factor
        else:
            withdrawal_planned = 0.0

        balance_before = balance
        rule_action = evaluate_ruleset(balance, m, cfg=cfg)

        # Map rule_action -> risk dial multiplier
        risk_mult = 1.0
        if rule_action == "reduce_risk":
            risk_mult = 0.5
        elif rule_action == "increase_risk":
            risk_mult = 1.2

        trades = wins = losses = 0
        gross_pnl = costs = net_pnl = 0.0
        risk_per_trade_usd = 0.0

        if failure_month is None and balance > 0:
            if use_trade_model:
                # Trades per month (rounded to int)
                trades = int(round(trades_per_week * weeks_per_month))
                if trades < 0:
                    trades = 0

                # Risk per trade (USD)
                risk_per_trade_usd = balance * (risk_per_trade_pct * risk_mult)

                # Generate wins/losses
                wins = sum(1 for _ in range(trades) if rng.random() < win_rate)
                losses = trades - wins

                gross_pnl = (wins * (avg_win_R * risk_per_trade_usd)) - (losses * (avg_loss_R * risk_per_trade_usd))
                costs = trades * cost_per_trade_usd
                net_pnl = gross_pnl - costs

                # Apply trade P&L
                balance += net_pnl
                total_costs += costs

            else:
                # Fixed-return fallback (original behavior)
                applied_return = base_monthly_return * risk_mult
                balance = balance * (1.0 + applied_return)

        # Contributions applied after trading/returns
        balance += contrib
        total_contrib += contrib

        # Cap withdrawal to what exists (realistic)
        withdrawal_actual = min(withdrawal_planned, max(balance, 0.0))
        balance -= withdrawal_actual
        total_withdrawn += withdrawal_actual

        # Failure occurs when we cannot meet the planned withdrawal
        if (withdrawal_actual < withdrawal_planned) and failure_month is None:
            failure_month = month_num
            balance = 0.0  # clamp

        history.append(
            {
                "month": month_num,
                "date": date.strftime("%Y-%m"),
                "phase": "retirement" if in_retirement else "accumulation",
                "rule_action": rule_action,
                "risk_mult": risk_mult,

                # Trade-model diagnostics
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "risk_per_trade_usd": round(risk_per_trade_usd, 2),
                "gross_pnl": round(gross_pnl, 2),
                "costs": round(costs, 2),
                "net_pnl": round(net_pnl, 2),

                # Cashflows
                "contribution": round(contrib, 2),
                "withdrawal_planned": round(withdrawal_planned, 2),
                "withdrawal_actual": round(withdrawal_actual, 2),

                # Balances
                "balance_before": round(balance_before, 2),
                "balance": round(balance, 2),
                "total_contributed": round(total_contrib, 2),
                "total_withdrawn": round(total_withdrawn, 2),
                "total_costs": round(total_costs, 2),

                "failed": failure_month is not None,
            }
        )

        if failure_month is not None and stop_on_failure:
            break

    df = pd.DataFrame(history)
    survived = failure_month is None
    final_balance = float(df["balance"].iloc[-1]) if len(df) else 0.0

    return SimResult(
        df=df,
        survived=survived,
        failure_month=failure_month,
        final_balance=final_balance,
        total_contributed=total_contrib,
        total_withdrawn=total_withdrawn,
    )

def main():
    print("=== Retirement Engine Simulation ===")
    res = simulate_retirement(cfg=RETIREMENT_CONFIG)
    
    df = res.df
    print(df.tail(12).to_string(index=False))
    
    status = "SURVIVED ✅" if res.survived else f"FAILED ❌ at month {res.failure_month}"
    print(f"\nStatus: {status}")
    print(f"Final balance: ${res.final_balance:,.2f}")
    print(f"Total contributed: ${res.total_contributed:,.2f}")
    print(f"Total withdrawn: ${res.total_withdrawn:,.2f}")
    
    if not res.survived:
        months_total = int(RETIREMENT_CONFIG["years"]) * 12
        months_short = months_total - int(res.failure_month or months_total)
        print(f"Ran full horizon: {months_total} months (30 years)")
        print(f"Failed at month {res.failure_month} → {months_short} months short of horizon")

    out_path = Path(RETIREMENT_CONFIG.get("output_path", "data/retirement_engine/sim_results.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved simulation results to {out_path.as_posix()}")
    

if __name__ == "__main__":
    main()
