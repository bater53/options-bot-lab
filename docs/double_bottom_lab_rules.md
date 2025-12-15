# Double-Bottom LAB Rules – Snapshot

Date: (fill in current date)

This file is a snapshot of the current, working LAB rules for SPY and QQQ
double-bottom breakouts on 5-minute candles, using ~60 days of history from
Yahoo Finance (yfinance).

It exists to help rebuild quickly if code is changed or broken.

---

## Common Structure

- Interval: 5-minute candles
- Lookback: 60d (limited by Yahoo 5m history)
- Data source: yfinance (no auto-adjust, intraday)
- Pattern detection:
  - local minima on `Low` with `left=3`, `right=3`
  - max bars between lows: 48 (tighter than default 60)
  - low tolerance: lows must be within 0.25% of each other
  - mid bounce: high between lows at least 0.8% above the lows
- Trade model (underlying-only R-multiple):
  - Entry: breakout above mid-high + 0.05% buffer
  - Stop: below lower low − 0.05% buffer
  - Target: entry + 1.5 * (entry − stop)
  - First touch after entry of stop or target wins (intrabar)

---

## Rule: DB_SPY_5M_LAB_V1

**Instrument:** SPY  
**Timeframe:** 5m  
**Lookback:** 60d

### Parameters

- `left = 3`
- `right = 3`
- `max_bars_between = 48`
- `low_tolerance_pct = 0.25`
- `min_mid_bounce_pct = 0.8`
- `rr_target = 1.5`
- `entry_buffer_pct = 0.05`
- `stop_buffer_pct = 0.05`

### Latest observed stats (example run)

- Bars loaded: 4606
- Candidate patterns: 22
- Trades planned: 22
- Resolved trades (hit stop or target): 20
  - Wins: 11
  - Losses: 9
- Unresolved trades at end: 2
- Win rate (resolved only): **55.0%**
- Avg R (resolved only): **+0.38**

**Expectancy check:**

- Geometry: +1.5R on win, −1R on loss
- Break-even win rate: 40%
- Empirical: 55% win, ~0.38R avg → healthy positive edge.

---

## Rule: DB_QQQ_5M_LAB_V1

**Instrument:** QQQ  
**Timeframe:** 5m  
**Lookback:** 60d

### Parameters

- `left = 3`
- `right = 3`
- `max_bars_between = 48`
- `low_tolerance_pct = 0.25`
- `min_mid_bounce_pct = 0.8`
- `rr_target = 1.5`
- `entry_buffer_pct = 0.05`
- `stop_buffer_pct = 0.05`

### Latest observed stats (example run)

- Bars loaded: ~4607
- Candidate patterns: 30
- Trades planned: 27
- Resolved trades (hit stop or target): 20
  - Wins: 9
  - Losses: 11
- Unresolved trades at end: 7
- Win rate (resolved only): **45.0%**
- Avg R (resolved only): **+0.12**

**Expectancy check:**

- Same geometry as SPY (1.5R / −1R)
- Break-even win rate: 40%
- Empirical: 45% win, ~0.12R avg → positive but smaller edge than SPY.

---

## Notes

- These rules are currently implemented directly in:

  - `src/backtest/run_double_bottom_spy_lab.py`
  - `src/backtest/run_double_bottom_qqq_lab.py`

- The Rule League should treat these rule definitions as the “source of truth”
  for configuration, even if implementations evolve.

- If you ever rebuild, you can reconstruct the logic and thresholds from this doc
  and re-run 60d 5m tests to validate you are “back” to this baseline.
