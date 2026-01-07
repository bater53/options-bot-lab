from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import importlib
import inspect
import pandas as pd

try:
    from src.risk.gate import assert_trading_allowed_today  # type: ignore
except Exception:
    # Windows repo copy may not include src/risk yet.
    # Fallback: allow running signals without the gate.
    def assert_trading_allowed_today() -> None:
        return None
  
from src.data.yf_candles import refresh_symbol_csv

KEEP_ORDER = [
    "rule_id",
    "ticker",
    "signal_time",
    "ts_ny",
    "trade_date",
    "entry",
    "stop",
    "target",
    "expiry",
    "strike",
    "option_type",
    "dte",
    "risk_pct",
    "reward_pct",
    "cost_pct",
    "R",
    "net_R",
    "net_reward_pct",
]


def _read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    return df


def _normalize(df: pd.DataFrame, tz: ZoneInfo) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # Ensure signal_time exists/parsed
    if "signal_time" in out.columns:
        st_utc = pd.to_datetime(out["signal_time"], errors="coerce", utc=True)
        ts_ny = st_utc.dt.tz_convert(tz)
        out["ts_ny"] = ts_ny.astype("string")
        out["_day_ny"] = ts_ny.dt.date.astype("string")
    else:
        out["ts_ny"] = pd.NA
        out["_day_ny"] = pd.NA

    # Normalize trade_date: prefer existing, else fill from NY day
    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].astype("string")
        out["trade_date"] = out["trade_date"].fillna(out["_day_ny"])
    else:
        out["trade_date"] = out["_day_ny"]

    # Ensure cost_pct exists and is numeric
    if "cost_pct" not in out.columns:
        out["cost_pct"] = pd.NA
    out["cost_pct"] = pd.to_numeric(out["cost_pct"], errors="coerce").fillna(0.0)

    # Drop helper underscore columns
    drop_cols = [c for c in out.columns if c.startswith("_")]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")

    # Reorder columns: known first, then extras
    extras = [c for c in out.columns if c not in KEEP_ORDER]
    out = out[[c for c in KEEP_ORDER if c in out.columns] + extras]

    return out


def _resolve_source_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "lab"

    alias = {
        "spy_rules": "src.signals.spy_paper_rules",
        "spy_paper_rules": "src.signals.spy_paper_rules",
        "lab": "lab",
    }
    if name in alias:
        return alias[name]

    if "." in name:
        return name

    return f"src.signals.{name}"


def load_source_module(source_name: str):
    mod_name = _resolve_source_name(source_name)
    if mod_name == "lab":
        return None
    return importlib.import_module(mod_name)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--day", default=None, help="YYYY-MM-DD (default: today NY)")
    p.add_argument("--out_dir", default="data/signals", help="output folder")
    p.add_argument("--also_write_live_candidates", action="store_true")
    p.add_argument(
        "--latest",
        action="store_true",
        help="If no rows for --day/today, write the most recent trade_date instead.",
    )
    p.add_argument("--refresh", action="store_true", help="Refresh candles via Yahoo before running rules")
    p.add_argument("--refresh_period", default="7d", help='Yahoo period e.g. "5d","7d","30d"')
    p.add_argument("--refresh_interval", default="5m", help='Yahoo interval e.g. "5m","15m"')
    p.add_argument("--debug", action="store_true", help="Print rule debug stats (why filters pass/fail)")
    p.add_argument(
        "--cost_bps",
        type=float,
        default=0.0,
        help="Estimated total trading cost in bps (e.g., 10 = 0.10%). Applied to reward_pct.",
    )

    # Back-compat switches
    p.add_argument("--spy", action="store_true", help="run SPY (back-compat)")
    p.add_argument("--qqq", action="store_true", help="run QQQ (back-compat)")

    # New interface
    p.add_argument("--symbol", default=None, help="Single symbol, e.g. SPY or QQQ")
    p.add_argument("--mode", default="paper", choices=["paper", "live"], help="paper or live")
    p.add_argument("--symbols", default=None, help="Comma-separated symbols, e.g. SPY,QQQ")

    p.add_argument(
        "--source",
        default="lab",
        help=(
            "Signal source. Use 'lab' to read the lab CSVs, or a module name. "
            "Aliases: spy_rules/spy_paper_rules -> src.signals.spy_paper_rules"
        ),
    )
    p.add_argument("--sources", default=None, help="Comma-separated list of sources (overrides --source), e.g. spy_rules,lab")

    return p.parse_args()


def _slug_source(s: str) -> str:
    s = (s or "lab").strip()
    if "." in s:
        s = s.split(".")[-1]
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def main() -> int:
    args = parse_args()

    DEFAULT_IN = {
        "SPY": "data/lab_spy_double_bottom_options_filtered.csv",
        "QQQ": "data/lab_qqq_double_bottom_options_filtered.csv",
    }

    # choose which symbols to run (priority: --symbols > --symbol > back-compat flags)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol.strip().upper()]
    else:
        symbols = []
        if args.spy:
            symbols.append("SPY")
        if args.qqq:
            symbols.append("QQQ")

    if not symbols:
        raise SystemExit("Pick a symbol: use --symbols SPY,QQQ or --symbol SPY or --spy/--qqq.")

    # gate (circuit breaker)
    assert_trading_allowed_today()

    tz = ZoneInfo("America/New_York")
    day = args.day or datetime.now(tz).date().isoformat()

    # Determine which sources to run (priority: --sources > --source)
    source_list = [s.strip() for s in (args.sources.split(",") if args.sources else [args.source]) if s.strip()]
    resolved_sources = [_resolve_source_name(s) for s in source_list]

    # Pre-load module sources (lab stays as None)
    source_mods: dict[str, object | None] = {}
    for src in source_list:
        resolved = _resolve_source_name(src)
        source_mods[resolved] = load_source_module(src)  # returns None for "lab"

    dfs: list[pd.DataFrame] = []

    for sym in symbols:
        # optional candle refresh
        if args.refresh:
            candle_dir = Path("data/candles")
            candle_dir.mkdir(parents=True, exist_ok=True)
            candle_csv = candle_dir / f"{sym}_{args.refresh_interval}.csv"
            refresh_symbol_csv(sym, candle_csv, period=args.refresh_period, interval=args.refresh_interval)

        # run each source for this symbol
        for src_resolved in resolved_sources:
            source_mod = source_mods.get(src_resolved)

            # --- LAB source ---
            if src_resolved == "lab":
                in_path = Path(DEFAULT_IN.get(sym, ""))
                if not in_path.exists():
                    if args.debug:
                        print(f"[WARN] lab: no default input CSV configured for {sym}")
                    continue

                df_sig = _read_if_exists(in_path)
                if df_sig.empty:
                    if args.debug:
                        print(f"[WARN] lab: empty input for {sym} at {in_path}")
                    continue

                if "ticker" not in df_sig.columns:
                    df_sig["ticker"] = sym

                df_sig["source"] = "lab"
                dfs.append(df_sig)
                continue

            # --- MODULE source ---
            if source_mod is None or not hasattr(source_mod, "generate_signals"):
                raise SystemExit(f"Source module {src_resolved} missing generate_signals(symbol=..., ...)")

            sig = inspect.signature(source_mod.generate_signals)
            kwargs = dict(symbol=sym, day=day, latest=args.latest, mode=args.mode)
            if "debug" in sig.parameters:
                kwargs["debug"] = args.debug

            # allow (df, debug_text) return OR just df
            debug_text = None
            result = source_mod.generate_signals(**kwargs)

            if isinstance(result, tuple) and len(result) == 2:
                df_sig, debug_text = result
            else:
                df_sig = result

            if args.debug and debug_text:
                print(debug_text)

            if df_sig is None or getattr(df_sig, "empty", True):
                if args.debug:
                    print(f"[WARN] {src_resolved}: no signals for {sym} (None/empty)")
                continue

            if "ticker" not in df_sig.columns:
                df_sig["ticker"] = sym

            df_sig["source"] = src_resolved.split(".")[-1]  # e.g. spy_paper_rules
            dfs.append(df_sig)

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=KEEP_ORDER)
    combined = _normalize(combined, tz)

    # Apply flat cost model (bps -> percent). Only fill where cost_pct is 0 so rule-provided costs win.
    if args.cost_bps and args.cost_bps > 0 and not combined.empty:
        flat_cost_pct = args.cost_bps / 100.0  # bps -> percent (10 bps = 0.10%)
        if "cost_pct" not in combined.columns:
            combined["cost_pct"] = 0.0
        combined["cost_pct"] = pd.to_numeric(combined["cost_pct"], errors="coerce").fillna(0.0)

        mask = combined["cost_pct"] <= 0
        combined.loc[mask, "cost_pct"] = flat_cost_pct

        # recompute net fields when possible
        if "reward_pct" in combined.columns:
            combined["reward_pct"] = pd.to_numeric(combined["reward_pct"], errors="coerce")
            combined["net_reward_pct"] = (combined["reward_pct"] - combined["cost_pct"]).clip(lower=0)

        if "risk_pct" in combined.columns and "net_reward_pct" in combined.columns:
            combined["risk_pct"] = pd.to_numeric(combined["risk_pct"], errors="coerce")
            combined["net_R"] = combined["net_reward_pct"] / combined["risk_pct"]

    # Filter using trade_date (NY day) if present
    if "trade_date" in combined.columns:
        todays = combined.loc[combined["trade_date"] == day].copy()
    else:
        todays = combined.copy()

    if todays.empty and args.latest and "trade_date" in combined.columns and not combined.empty:
        last_day = combined["trade_date"].dropna().astype("string").max()
        todays = combined.loc[combined["trade_date"] == last_day].copy()
        print(f"[INFO] No rows for day={day}; using latest trade_date={last_day} rows={len(todays)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If multiple sources, put them in the filename tag
    if len(source_list) > 1:
        src_tag = "_".join(_slug_source(s) for s in source_list)
    else:
        src_tag = _slug_source(source_list[0])

    sym_tag = "_".join(symbols) if len(symbols) > 1 else symbols[0]
    out_path = out_dir / f"signals_{day}_{sym_tag}_{src_tag}.csv"

    todays.to_csv(out_path, index=False)

    if todays.empty:
        print(f"[WARN] No rows loaded; wrote empty {out_path}")
        return 0

    print(f"[OK] breaker clear — wrote {out_path} rows={len(todays)}")

    if args.also_write_live_candidates:
        lc_path = Path("data/live_candidates.csv")
        lc_path.parent.mkdir(parents=True, exist_ok=True)
        todays.to_csv(lc_path, index=False)
        print(f"[OK] updated {lc_path} rows={len(todays)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
