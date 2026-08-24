"""
risk_education.py

Deliberately NOT a signal generator. This module computes Average True
Range (ATR) - a standard, well-established volatility measure - and
returns it alongside illustrative examples of how it's commonly used
for stop-loss/take-profit distance and position sizing.

What this does NOT do, on purpose: it never says "buy here, stop here,
take profit here" for the current forecast. The methodology here is
completely decoupled from the model's directional prediction - ATR-based
risk sizing works the same whether the model says up or down, because
it's about managing risk on a trade you decide to make, not about
telling you to make one. The distinction matters: this project's whole
premise is showing honestly how small the model's real edge is (often
~1-2 percentage points over baseline, sometimes none at all) - bolting
confident entry/stop/target numbers onto that would manufacture false
precision the backtest results explicitly don't support.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - the average of the 'true range' (the largest
    of: high-low, |high-prev_close|, |low-prev_close|) over `period` bars.
    Standard Wilder's smoothing (an exponential moving average variant),
    the same method most charting platforms use."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing: equivalent to an EMA with alpha = 1/period
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


@dataclass
class RiskExample:
    stop_multiplier: float
    target_multiplier: float
    label: str
    stop_distance: float
    target_distance: float
    risk_reward_ratio: float


def build_risk_education(df: pd.DataFrame, current_price: float) -> dict:
    """Returns the current ATR plus a table of illustrative stop/target
    distances at common ATR multiples - explicitly a reference table of
    how the methodology works, not a single prescribed number."""
    atr_series = compute_atr(df)
    current_atr = float(atr_series.iloc[-1])
    atr_pct_of_price = round((current_atr / current_price) * 100, 2)

    # Common, widely-taught ATR multiples - not "the" right answer, just
    # the standard reference points most risk-management education uses.
    combos = [
        (1.0, 2.0, "Tight stop"),
        (1.5, 3.0, "Common default"),
        (2.0, 4.0, "Wider stop"),
    ]
    examples = [
        RiskExample(
            stop_multiplier=stop_mult, target_multiplier=target_mult, label=label,
            stop_distance=round(current_atr * stop_mult, 4),
            target_distance=round(current_atr * target_mult, 4),
            risk_reward_ratio=round(target_mult / stop_mult, 2),
        )
        for stop_mult, target_mult, label in combos
    ]

    return {
        "current_atr": round(current_atr, 4),
        "atr_period": 14,
        "atr_pct_of_price": atr_pct_of_price,
        "current_price": round(current_price, 4),
        "examples": [e.__dict__ for e in examples],
        "note": (
            "This is risk-sizing methodology, not a trade recommendation - it works the same "
            "regardless of the model's forecast direction above. ATR measures how much this "
            "instrument typically moves; these are standard reference multiples for stop/target "
            "distance, not a prescription. Decide the trade yourself; use a calculator like this "
            "only to size the risk on it."
        ),
    }


def position_size(account_size: float, risk_pct: float, stop_distance: float, price_per_unit: float = 1.0) -> dict:
    """The actual position-sizing formula: given how much of your account
    you're willing to risk and how far your stop is, how many units can
    you hold? risk_amount = account_size * (risk_pct/100); units =
    risk_amount / stop_distance. Pure arithmetic, no model involved -
    this is the same formula any risk-management course teaches."""
    if stop_distance <= 0:
        raise ValueError("stop_distance must be positive")
    risk_amount = account_size * (risk_pct / 100)
    units = risk_amount / stop_distance
    position_value = units * price_per_unit
    return {
        "risk_amount": round(risk_amount, 2),
        "units": round(units, 4),
        "position_value": round(position_value, 2),
    }
