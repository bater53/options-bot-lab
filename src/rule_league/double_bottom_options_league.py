"""
Double Bottom Options 'Rule League' dashboard.

This is a first-pass league table that:
- Looks for CSVs with trade or summary stats
- Computes n_trades, win_rate, avg_R, total_R
- Ranks rules so we can see which variants are promotion candidates.

Run from project root:

    (.venv) python -m src.rule_league.double_bottom_options_league
"""

import argparse
import pathlib
from typing import List, Dict, Any

import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Project root is two levels above this file:
#   <root>/src/rule_league/double_bottom_options_league.py
#   parents[0] = rule_league
#   parents[1] = src
#   parents[2] = <root>
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Absolute path to the league directory, independent of where we run python from
DEFAULT_LEAGUE_DIR = PROJECT_ROOT / "data" / "rule_league"

# If no --files are passed, we’ll glob this pattern under DEFAULT_LEAGUE_DIR
DEFAULT_GLOB = "*.csv"

R_COL_CANDIDATES = ["R", "r", "r_multiple", "R_multiple"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Double Bottom Options Rule League Dashboard"
    )
    parser.add_argument(
        "--files",
        type=str,
        default="",
        help=(
            "Comma-separated list of CSV files. "
            "If omitted, scan data/rule_league/*.csv."
        ),
    )
    parser.add_argument(
        "--min_trades",
        type=int,
        default=5,
        help="Minimum trades required for a rule to be shown.",
    )
    return parser.parse_args()

def discover_csv_paths(args: argparse.Namespace) -> List[pathlib.Path]:
    # If explicit files are passed, just trust those.
    if args.files:
        return [pathlib.Path(p.strip()) for p in args.files.split(",") if p.strip()]

    league_dir = DEFAULT_LEAGUE_DIR

    # Debug info so we can see exactly what we're doing
    print(f"[debug] cwd = {pathlib.Path().resolve()}")
    print(f"[debug] league dir = {league_dir} (exists={league_dir.exists()})")

    if not league_dir.exists():
        print(f"[warn] Default league dir {league_dir} does not exist yet.")
        return []

    # Manually list all files and pick .csv / .CSV / .CsV etc.
    all_entries = list(league_dir.iterdir())
    print("[debug] entries in league dir:")
    for e in all_entries:
        print(f"    - {e.name} (is_file={e.is_file()}, suffix={e.suffix!r})")

    csv_paths = [
        p for p in all_entries
        if p.is_file() and p.suffix.lower() == ".csv"
    ]

    print(f"[debug] found {len(csv_paths)} CSV(s) in league dir after filtering.")
    return sorted(csv_paths)

def detect_r_col(df: pd.DataFrame) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in R_COL_CANDIDATES:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def summarize_per_trade(df: pd.DataFrame, file_path: pathlib.Path) -> Dict[str, Any] | None:
    """
    For CSVs that have one row per trade with an R multiple.
    We'll aggregate into one summary for the entire file.
    """
    r_col = detect_r_col(df)
    if r_col is None:
        return None

    cols_lower = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    symbol_col = pick("symbol", "ticker", "underlying")

    n_trades = len(df)
    wins = df[r_col] > 0
    n_wins = int(wins.sum())
    n_losses = n_trades - n_wins
    win_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0
    avg_R = float(df[r_col].mean()) if n_trades > 0 else 0.0
    total_R = float(df[r_col].sum()) if n_trades > 0 else 0.0

    # ----- symbol detection: filename FIRST, column only if filename is unknown -----
    stem_upper = file_path.stem.upper()
    if "SPY" in stem_upper:
        symbol = "SPY"
    elif "QQQ" in stem_upper:
        symbol = "QQQ"
    else:
        symbol = "UNKNOWN"

    # Only let the CSV override if the filename didn't give us SPY/QQQ
    if symbol == "UNKNOWN" and symbol_col and not df[symbol_col].empty:
        try:
            col_symbol = str(df[symbol_col].mode().iloc[0]).upper()
            if col_symbol in {"SPY", "QQQ"}:
                symbol = col_symbol
        except Exception:
            pass

    return {
        "rule_id": file_path.stem,
        "symbol": symbol,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "total_R": total_R,
        "source_file": str(file_path),
    }

    # ----- symbol detection: filename first, then column -----
    stem_upper = file_path.stem.upper()

    if "SPY" in stem_upper:
        symbol = "SPY"
    elif "QQQ" in stem_upper:
        symbol = "QQQ"
    else:
        symbol = "UNKNOWN"

    if symbol_col and not df[symbol_col].empty:
        try:
            col_symbol = str(df[symbol_col].mode().iloc[0]).upper()
            # Only override if it looks like a real underlying we care about
            if col_symbol in {"SPY", "QQQ"}:
                symbol = col_symbol
        except Exception:
            # If anything goes weird, just stick with filename-based symbol
            pass

    return {
        "rule_id": file_path.stem,
        "symbol": symbol,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "total_R": total_R,
        "source_file": str(file_path),
    }

    # Default symbol from filename, then override with column if usef
    
    stem_upper = file_path.stem.upper()
    if "SPY" in stem_upper:
        symbol = "SPY"
    elif "QQQ" in stem_upper:
        symbol = "QQQ"
    else:
        symbol = "UNKNOWN"

    if symbol_col and not df[symbol_col].empty:
        try:
            col_symbol = str(df[symbol_col].mode().iloc[0]).upper()
            # Only override if the column is something sensible like SPY/QQQ
            if col_symbol in {"SPY", "QQQ"}:
                symbol = col_symbol
        except Exception:
            pass

    return {
        "rule_id": file_path.stem,
        "symbol": symbol,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "total_R": total_R,
        "source_file": str(file_path),
    }


def summarize_file(file_path: pathlib.Path) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[warn] Failed to read {file_path}: {e}")
        return []

    # For now we only implement per-trade summarization.
    pt = summarize_per_trade(df, file_path)
    return [pt] if pt is not None else []


def print_league_table(rows: List[Dict[str, Any]], min_trades: int) -> None:
    if not rows:
        print("No rules found for league table.")
        return

    df = pd.DataFrame(rows)
    df = df[df["n_trades"] >= min_trades].copy()
    if df.empty:
        print(f"No rules with at least {min_trades} trades.")
        return

    df.sort_values(
        by=["avg_R", "total_R", "n_trades"],
        ascending=[False, False, False],
        inplace=True,
    )

    display_cols = [
        "rule_id",
        "symbol",
        "n_trades",
        "win_rate",
        "avg_R",
        "total_R",
        "source_file",
    ]
    df_display = df[display_cols].copy()
    df_display["win_rate"] = (df_display["win_rate"] * 100).map("{:.1f}%".format)
    df_display["avg_R"] = df_display["avg_R"].map("{:.2f}".format)
    df_display["total_R"] = df_display["total_R"].map("{:.2f}".format)

    print("\n=== Double Bottom Options Rule League ===")
    print(f"(min_trades >= {min_trades})\n")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_colwidth", 60,
        "display.width", 140,
    ):
        print(df_display.to_string(index=False))
    print()


def main() -> None:
    args = parse_args()
    paths = discover_csv_paths(args)

    if not paths:
        print(
            "[info] No CSVs found. Once your LAB runs write stats into "
            "data/rule_league/, rerun this command."
        )
        return

    print("[info] Loading rule stats from:")
    for p in paths:
        print(f"  - {p}")

    all_rows: List[Dict[str, Any]] = []
    for p in paths:
        rows = summarize_file(p)
        if not rows:
            print(f"[warn] No usable stats in {p} (skipping).")
            continue
        all_rows.extend(rows)

    print_league_table(all_rows, min_trades=args.min_trades)


if __name__ == "__main__":
    main()
