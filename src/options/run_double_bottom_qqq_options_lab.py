"""
LAB: Project QQQ double-bottom underlying trades into option trade plans.

Usage:
    python -m src.run_double_bottom_qqq_options_lab
"""


from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Literal
import statistics  # <-- add this

import numpy as np
import pandas as pd
import yfinance as yf



# ── Option-side data models ────────────────────────────────────

OptionType = Literal["CALL", "PUT"]


@dataclass
class UnderlyingSignal:
    """
    A clean description of the underlying trade we want to express in options.
    """
    rule_id: str                 # e.g. "DB_QQQ_5M_LAB_V1"
    ticker: str                  # "QQQ"
    direction: Literal["LONG", "SHORT"]
    signal_time: pd.Timestamp    # when the entry is triggered
    entry_price: float
    stop_price: float
    target_price: float


@dataclass
class OptionContractSpec:
    """
    Describes which option contract we intend to trade.
    """
    underlying: str              # "QQQ"
    option_type: OptionType      # "CALL" or "PUT"
    expiry: date
    strike: float
    rule_id: str
    notes: str = ""


@dataclass
class OptionTradePlan:
    """
    Full option trade expression derived from an underlying signal.
    """
    signal: UnderlyingSignal
    contract: OptionContractSpec
    contracts: int = 1           # LAB: fixed 1 contract for now
    meta: Optional[dict] = None  # room for extra LAB info later


def _next_friday_on_or_after(d: date) -> date:
    """
    Return the next Friday on or after the given date.
    Monday=0, Sunday=6, Friday=4.
    """
    weekday = d.weekday()
    days_ahead = (4 - weekday) % 7
    return d + timedelta(days=days_ahead)


def choose_weekly_expiry(
    signal_time: pd.Timestamp,
    min_days_out: int = 5,
    max_days_out: int = 15,
) -> date:
    """
    Simple, broker-agnostic expiry rule for LAB:

    - Start from the signal date.
    - Pick the first Friday that is at least `min_days_out` days away.
    - If that Friday is more than `max_days_out` days away, we still use it
      (fine for LAB; can be refined later).
    """
    d0 = signal_time.date()
    first_friday = _next_friday_on_or_after(d0)

    if (first_friday - d0).days < min_days_out:
        candidate = first_friday + timedelta(days=7)
    else:
        candidate = first_friday

    return candidate


def round_strike_qqq_spy(underlying_price: float) -> float:
    """
    For QQQ / SPY, strikes are usually in 1.0 increments (sometimes 0.5).
    LAB: we keep it simple and round to the nearest whole number.
    """
    return float(round(underlying_price))


def map_underlying_to_call_option(
    signal: UnderlyingSignal,
    *,
    otm_pct: float = 1.0,
    min_dte: int = 5,
    max_dte: int = 15,
) -> OptionTradePlan:
    """
    Map a LONG underlying signal to a simple call option trade plan.

    Rule (LAB version):
      - Option type: CALL (double-bottom is bullish).
      - Expiry: next suitable weekly Friday with ~5–15 days to expiry.
      - Strike: slightly OTM by `otm_pct` (%) from the underlying entry price,
                rounded to a QQQ/SPY-style strike increment.
    """
    if signal.direction != "LONG":
        raise ValueError("map_underlying_to_call_option currently only supports LONG signals.")

    # Choose expiry
    expiry = choose_weekly_expiry(
        signal.signal_time,
        min_days_out=min_dte,
        max_days_out=max_dte,
    )

    # Choose strike: slightly OTM
    otm_price = signal.entry_price * (1.0 + otm_pct / 100.0)
    strike = round_strike_qqq_spy(otm_price)

    contract = OptionContractSpec(
        underlying=signal.ticker,
        option_type="CALL",
        expiry=expiry,
        strike=strike,
        rule_id=signal.rule_id,
        notes=f"{signal.ticker} call, ~{min_dte}-{max_dte} DTE, ~{otm_pct:.1f}% OTM",
    )

    return OptionTradePlan(
        signal=signal,
        contract=contract,
        contracts=1,
        meta=None,
    )
# ── QQQ double-bottom core (copied from your LAB runner) ──────

@dataclass
class DoubleBottomPattern:
    idx1: int
    idx2: int
    mid_idx: int
    low1: float
    low2: float
    mid_high: float


@dataclass
class PlannedTrade:
    entry_idx: int
    stop_idx: int
    target_idx: int
    entry: float
    stop: float
    target: float
    R: Optional[float] = None
    hit_target: bool = False
    hit_stop: bool = False


def load_qqq_5m(period: str = "60d") -> pd.DataFrame:
    df = yf.download(
        "QQQ",
        interval="5m",
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError("Failed to download QQQ data (empty dataframe).")

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))

        if "QQQ" in lvl0:
            df = df.xs("QQQ", axis=1, level=0)
        elif "QQQ" in lvl1:
            df = df.xs("QQQ", axis=1, level=1)
        else:
            first = list(df.columns.levels[0])[0]
            df = df.xs(first, axis=1, level=0)

    cols_lower = {str(c).lower(): c for c in df.columns}
    needed = ["open", "high", "low", "close"]
    missing = [name for name in needed if name not in cols_lower]
    if missing:
        raise RuntimeError(
            f"Unexpected columns from yfinance. "
            f"Needed {needed}, got {list(df.columns)}"
        )

    rename_map = {
        cols_lower["open"]: "Open",
        cols_lower["high"]: "High",
        cols_lower["low"]: "Low",
        cols_lower["close"]: "Close",
    }
    if "adj close" in cols_lower:
        rename_map[cols_lower["adj close"]] = "AdjClose"
    if "volume" in cols_lower:
        rename_map[cols_lower["volume"]] = "Volume"

    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return df


def find_local_minima(lows: pd.Series, left: int = 3, right: int = 3) -> np.ndarray:
    arr = lows.to_numpy()
    n = len(arr)
    idxs: List[int] = []

    for i in range(left, n - right):
        window = arr[i - left : i + right + 1]
        center = arr[i]
        if center == window.min() and center < arr[i - 1] and center < arr[i + 1]:
            idxs.append(i)

    return np.array(idxs, dtype=int)


def detect_double_bottoms(
    df: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
    max_bars_between: int = 48,
    low_tolerance_pct: float = 0.25,
    min_mid_bounce_pct: float = 0.8,
) ->List[DoubleBottomPattern]:
    lows = df["Low"].to_numpy()
    highs = df["High"].to_numpy()
    local_min_idx = find_local_minima(df["Low"], left=left, right=right)

    patterns: List[DoubleBottomPattern] = []
    if len(local_min_idx) < 2:
        return patterns

    for i_pos in range(len(local_min_idx)):
        i = local_min_idx[i_pos]
        for j_pos in range(i_pos + 1, len(local_min_idx)):
            j = local_min_idx[j_pos]

            if j - i > max_bars_between:
                break

            low1 = lows[i]
            low2 = lows[j]

            if abs(low2 - low1) / low1 * 100.0 > low_tolerance_pct:
                continue

            mid_slice = slice(i, j + 1)
            mid_high = highs[mid_slice].max()
            min_low = min(low1, low2)
            bounce_pct = (mid_high - min_low) / min_low * 100.0

            if bounce_pct < min_mid_bounce_pct:
                continue

            mid_candidates = np.arange(len(df))[mid_slice]
            local_highs = highs[mid_slice]
            mid_idx = int(mid_candidates[local_highs.argmax()])

            patterns.append(
                DoubleBottomPattern(
                    idx1=int(i),
                    idx2=int(j),
                    mid_idx=mid_idx,
                    low1=float(low1),
                    low2=float(low2),
                    mid_high=float(mid_high),
                )
            )
            break

    return patterns


def plan_trades_from_double_bottoms(
    df: pd.DataFrame,
    patterns: List[DoubleBottomPattern],
    *,
    rr_target: float = 1.5,
    entry_buffer_pct: float = 0.05,
    stop_buffer_pct: float = 0.05,
) -> List[PlannedTrade]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    trades: List[PlannedTrade] = []

    for p in patterns:
        entry_level = p.mid_high * (1.0 + entry_buffer_pct / 100.0)
        stop_level = min(p.low1, p.low2) * (1.0 - stop_buffer_pct / 100.0)

        risk_per_share = entry_level - stop_level
        if risk_per_share <= 0:
            continue

        target_level = entry_level + rr_target * risk_per_share

        entry_idx: Optional[int] = None
        hit_stop = False
        hit_target = False
        stop_idx: Optional[int] = None
        target_idx: Optional[int] = None

        for k in range(p.idx2 + 1, len(df)):
            bar_low = lows[k]
            bar_high = highs[k]

            if entry_idx is None:
                if bar_low <= entry_level <= bar_high:
                    entry_idx = k
                else:
                    continue

            if bar_low <= stop_level:
                hit_stop = True
                stop_idx = k
                break
            if bar_high >= target_level:
                hit_target = True
                target_idx = k
                break

        if entry_idx is None:
            continue

        R: Optional[float] = None
        if hit_target:
            R = rr_target
        elif hit_stop:
            R = -1.0

        trades.append(
            PlannedTrade(
                entry_idx=entry_idx,
                stop_idx=stop_idx or entry_idx,
                target_idx=target_idx or entry_idx,
                entry=float(entry_level),
                stop=float(stop_level),
                target=float(target_level),
                R=R,
                hit_target=hit_target,
                hit_stop=hit_stop,
            )
        )

    return trades


# ── Map QQQ underlying trades to option plans ─────────────────

def underlying_trade_to_signal(
    df: pd.DataFrame,
    trade: PlannedTrade,
    rule_id: str = "DB_QQQ_5M_LAB_V1",
) -> UnderlyingSignal:
    """
    Convert an underlying PlannedTrade into a clean UnderlyingSignal
    for options mapping.
    """
    idx = df.index[trade.entry_idx]
    return UnderlyingSignal(
        rule_id=rule_id,
        ticker="QQQ",
        direction="LONG",
        signal_time=idx,
        entry_price=trade.entry,
        stop_price=trade.stop,
        target_price=trade.target,
    )

def build_option_plans_for_qqq() -> List[OptionTradePlan]:
    period = "60d"
    print(f"Loading QQQ 5m candles for about the last {period}…")
    df = load_qqq_5m(period)
    print(f"Loaded {len(df)} bars.")

    patterns = detect_double_bottoms(df)
    print(f"[debug] detected {len(patterns)} candidate double bottoms")

    trades = plan_trades_from_double_bottoms(df, patterns)
    print(f"[debug] planned {len(trades)} underlying trades")

    plans: List[OptionTradePlan] = []

    for t in trades:
        if t.R is None:
            continue

        signal = underlying_trade_to_signal(df, t)
        plan = map_underlying_to_call_option(
            signal,
            otm_pct=1.0,
            min_dte=5,
            max_dte=15,
        )

        plan.meta = (plan.meta or {})
        plan.meta.update(
            {
                "underlying_R": t.R,
                "hit_target": t.hit_target,
                "hit_stop": t.hit_stop,
            }
        )

        plans.append(plan)

    return plans

#-------------Summary Function ----------------------------------

def summarize_variant(
    plans: List[OptionTradePlan],
    name: str,
    risk_min: Optional[float] = None,
    risk_max: Optional[float] = None,
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
) -> None:
    """
    Print a LAB-style summary for a subset of plans that satisfy
    the given metric bounds (risk_pct, dte).
    """
    subset = []

    for plan in plans:
        meta = plan.meta or {}
        risk = meta.get("risk_pct")
        dte = meta.get("dte")

        ok = True

        if risk_min is not None:
            if risk is None or risk < risk_min:
                ok = False
        if risk_max is not None:
            if risk is None or risk > risk_max:
                ok = False
        if dte_min is not None:
            if dte is None or dte < dte_min:
                ok = False
        if dte_max is not None:
            if dte is None or dte > dte_max:
                ok = False

        if ok:
            subset.append(plan)

    print(f"\n--- Variant: {name} ---")
    print(f"Trades: {len(subset)}")

    if not subset:
        return

    # pull metrics out of meta
    risks = []
    rewards = []
    Rs = []
    dtes = []

    for plan in subset:
        meta = plan.meta or {}
        if (rp := meta.get("risk_pct")) is not None:
            risks.append(rp)
        if (rw := meta.get("reward_pct")) is not None:
            rewards.append(rw)
        if (rr := meta.get("R")) is not None:
            Rs.append(rr)
        if (dd := meta.get("dte")) is not None:
            dtes.append(dd)

    if dtes:
        print(
            f"DTE: median={statistics.median(dtes):.1f}d, "
            f"min={min(dtes)}d, max={max(dtes)}d"
        )

    if risks:
        print(
            f"Risk %: median={statistics.median(risks):.2f}%, "
            f"min={min(risks):.2f}%, max={max(risks):.2f}%"
        )

    if rewards:
        print(
            f"Reward %: median={statistics.median(rewards):.2f}%, "
            f"min={min(rewards):.2f}%, max={max(rewards):.2f}%"
        )

    if Rs:
        print(
            f"R multiple: median={statistics.median(Rs):.2f}R, "
            f"min={min(Rs):.2f}R, max={max(Rs):.2f}R"
        )


#------------- main function ---------------------------------

def main() -> None:
    plans = build_option_plans_for_qqq()
    
    print()
    print("=== QQQ Double-Bottom → Call Option LAB Plans ===")

    # ---------- per-trade printout ----------

    for i, plan in enumerate(plans, start=1):
        sig = plan.signal
        con = plan.contract

        print(
            f"{i:2d}. {sig.rule_id} | {sig.signal_time} | "
            f"{sig.ticker} entry={sig.entry_price:.2f}, "
            f"stop={sig.stop_price:.2f}, "
            f"target={sig.target_price:.2f}"
        )
        print(
            f"    → {con.underlying} {con.option_type} {con.expiry} {con.strike:.0f} "
            f"(notes: {con.notes})"
        )

    print(f"\nTotal option plans generated: {len(plans)}")

    # ---------- LAB summary ----------

    if not plans:
        return

    stats_rows = []

    for plan in plans:
        sig = plan.signal
        con = plan.contract

        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price

        # risk/reward in %
        risk_pct = (entry - stop) / entry * 100 if entry and stop else 0.0
        reward_pct = (target - entry) / entry * 100 if entry and target else 0.0
        R = (reward_pct / risk_pct) if risk_pct > 0 else None

        # DTE (days to expiry)
        entry_date = sig.signal_time.date()          # pd.Timestamp -> date
        expiry_date = con.expiry                    # already a date
        dte = (expiry_date - entry_date).days if expiry_date and entry_date else None


        # store in meta so variants can reuse
        meta = {
            "risk_pct": risk_pct,
            "reward_pct": reward_pct,
            "R": R,
            "dte": dte,
        }
        if plan.meta is None:
            plan.meta = meta
        else:
            plan.meta.update(meta)

        stats_rows.append(meta)
            
    risks = [r["risk_pct"] for r in stats_rows if r["risk_pct"] > 0]
    rewards = [r["reward_pct"] for r in stats_rows if r["reward_pct"] != 0]
    Rs = [r["R"] for r in stats_rows if r["R"] is not None]
    dtes = [r["dte"] for r in stats_rows if r["dte"] is not None]

    dte_0_7 = [d for d in dtes if d <= 7]
    dte_8_15 = [d for d in dtes if 8 <= d <= 15]
    dte_16_plus = [d for d in dtes if d >= 16]

    print("\n=== LAB Summary ===")
    print(f"Total option plans: {len(plans)}")

    if dtes:
        print(
            f"DTE buckets: "
            f"≤7d: {len(dte_0_7)}, "
            f"8–15d: {len(dte_8_15)}, "
            f"≥16d: {len(dte_16_plus)}"
        )

    if risks:
        print(
            f"Risk % (entry→stop): "
            f"median={statistics.median(risks):.2f}%, "
            f"min={min(risks):.2f}%, max={max(risks):.2f}%"
        )

    if rewards:
        print(
            f"Reward % (entry→target): "
            f"median={statistics.median(rewards):.2f}%, "
            f"min={min(rewards):.2f}%, max={max(rewards):.2f}%"
        )

    if Rs:
        print(
            f"R multiple (target/risk): "
            f"median={statistics.median(Rs):.2f}R, "
            f"min={min(Rs):.2f}R, max={max(Rs):.2f}R"
        )

    # ---------- Rule-variant summaries ----------

    summarize_variant(
        plans,
        "Risk 0.8–1.3%",
        risk_min=0.8,
        risk_max=1.3,
    )

    summarize_variant(
        plans,
        "DTE 5–10 days",
        dte_min=5,
        dte_max=10,
    )

    summarize_variant(
        plans,
        "Risk 0.8–1.3% & DTE 5–10 days",
        risk_min=0.8,
        risk_max=1.3,
        dte_min=5,
        dte_max=10,
    )

    # ---------- Export plans to CSV for offline inspection ----------
    import os

    os.makedirs("data", exist_ok=True)

    rows = []
    for plan in plans:
        sig = plan.signal
        con = plan.contract
        meta = plan.meta or {}

        rows.append(
            {
                "rule_id": sig.rule_id,
                "ticker": sig.ticker,
                "signal_time": sig.signal_time,
                "entry": sig.entry_price,
                "stop": sig.stop_price,
                "target": sig.target_price,
                "expiry": con.expiry,
                "strike": con.strike,
                "option_type": con.option_type,
                "risk_pct": meta.get("risk_pct"),
                "reward_pct": meta.get("reward_pct"),
                "R": meta.get("R"),
                "dte": meta.get("dte"),
            }
        )

    df_export = pd.DataFrame(rows)
    out_path = "data/lab_qqq_double_bottom_options.csv"
    df_export.to_csv(out_path, index=False)
    print(f"\n[LAB] Exported QQQ option plans to {out_path}")

    
if __name__ == "__main__":
    main()

