import pandas as pd
import numpy as np

def summarize(trades: list):
    if not trades:
        return {}

    pnl_R = np.array([t.pnl_R for t in trades])
    wins = pnl_R[pnl_R > 0]
    losses = pnl_R[pnl_R <= 0]

    win_rate = len(wins) / len(pnl_R)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0  # negative

    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    equity = pnl_R.cumsum()
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = drawdown.min() if len(drawdown) else 0.0

    return {
        "trades": len(pnl_R),
        "win_rate": win_rate,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "expectancy_R": expectancy,
        "max_drawdown_R": max_dd,
        "total_R": pnl_R.sum(),
    }
