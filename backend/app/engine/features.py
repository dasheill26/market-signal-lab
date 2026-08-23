"""
features.py

Technical indicators computed manually with pandas rather than pulled from
a library like `ta` - deliberately, so every number here is something I
can actually explain and defend, not a black-box import. Each function is
a standard, well-known indicator; nothing exotic.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a DataFrame with columns [Open, High, Low, Close, Volume]
    (standard OHLCV, one row per period) and adds technical indicator
    columns used as model features. Returns a new DataFrame - does not
    mutate the input."""
    df = df.copy()
    close = df["Close"]

    # Moving averages - trend direction
    df["sma_10"] = close.rolling(10).mean()
    df["sma_30"] = close.rolling(30).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()

    # Momentum: RSI (Relative Strength Index), 14-period standard
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands - volatility bands around a 20-period mean
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Realized volatility - rolling std of returns
    returns = close.pct_change()
    df["volatility_10"] = returns.rolling(10).std()

    # Lagged returns - the model's memory of recent price action
    for lag in (1, 2, 3, 5):
        df[f"return_lag_{lag}"] = returns.shift(lag)

    # Volume change
    df["volume_change"] = df["Volume"].pct_change()

    # Price relative to moving averages (normalized, scale-invariant)
    df["close_vs_sma10"] = (close - df["sma_10"]) / df["sma_10"]
    df["close_vs_sma30"] = (close - df["sma_30"]) / df["sma_30"]

    return df


FEATURE_COLUMNS = [
    "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pct", "volatility_10",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "volume_change", "close_vs_sma10", "close_vs_sma30",
]


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Binary target: does the close price go up over the next `horizon`
    periods? This is a directional classification target, deliberately -
    predicting the exact future price is a much easier claim to overclaim
    and a much harder one to evaluate honestly than "up or down"."""
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_return > 0).astype(int)
