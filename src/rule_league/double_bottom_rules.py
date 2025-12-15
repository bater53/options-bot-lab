"""
Config definitions for double-bottom LAB rules.

These capture the parameters and identity of each rule
(e.g. for the Rule League to iterate over).
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class DoubleBottomRuleConfig:
    rule_id: str
    ticker: str
    interval: str
    period: str
    left: int
    right: int
    max_bars_between: int
    low_tolerance_pct: float
    min_mid_bounce_pct: float
    rr_target: float
    entry_buffer_pct: float
    stop_buffer_pct: float


# Baseline configurations matching the current SPY/QQQ LAB runners
DB_SPY_5M_LAB_V1 = DoubleBottomRuleConfig(
    rule_id="DB_SPY_5M_LAB_V1",
    ticker="SPY",
    interval="5m",
    period="60d",
    left=3,
    right=3,
    max_bars_between=48,
    low_tolerance_pct=0.25,
    min_mid_bounce_pct=0.8,
    rr_target=1.5,
    entry_buffer_pct=0.05,
    stop_buffer_pct=0.05,
)

DB_QQQ_5M_LAB_V1 = DoubleBottomRuleConfig(
    rule_id="DB_QQQ_5M_LAB_V1",
    ticker="QQQ",
    interval="5m",
    period="60d",
    left=3,
    right=3,
    max_bars_between=48,
    low_tolerance_pct=0.25,
    min_mid_bounce_pct=0.8,
    rr_target=1.5,
    entry_buffer_pct=0.05,
    stop_buffer_pct=0.05,
)


ALL_DOUBLE_BOTTOM_RULES: Dict[str, DoubleBottomRuleConfig] = {
    DB_SPY_5M_LAB_V1.rule_id: DB_SPY_5M_LAB_V1,
    DB_QQQ_5M_LAB_V1.rule_id: DB_QQQ_5M_LAB_V1,
}
