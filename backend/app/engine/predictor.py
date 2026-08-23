"""
predictor.py

The single entry point routes.py calls: given a symbol, fetch data, run
the full pipeline, and return chart data, a next-period forecast, and an
honest backtest report - all in one call, so the API layer doesn't need
to know about features/model/backtest internals separately.
"""

from dataclasses import dataclass, asdict

from .data_source import fetch_ohlcv
from .features import add_features, FEATURE_COLUMNS
from .model import prepare_dataset, train_model
from .backtest import walk_forward_backtest


@dataclass
class Forecast:
    symbol: str
    data_mode: str  # 'live' or 'cached_demo_data'
    direction: str  # 'up' or 'down'
    confidence_pct: float
    last_close: float
    last_date: str
    backtest_mean_accuracy: float
    backtest_naive_baseline: float
    beats_baseline: bool
    n_backtest_folds: int
    disclaimer: str = (
        "This is a statistical forecast for demonstration purposes, based on historical "
        "technical indicators. It is not financial advice, and the backtested accuracy above "
        "should tell you exactly how much (or little) to trust it - markets are close to "
        "efficient, and no model here has a real trading edge."
    )


def run_forecast(symbol: str, horizon: int = 1) -> tuple[Forecast, "pd.DataFrame"]:
    raw_df, mode = fetch_ohlcv(symbol)
    df = add_features(raw_df)

    X, y, full_df = prepare_dataset(raw_df, horizon=horizon)

    # Backtest first (honest performance context), using all data except
    # the most recent horizon rows (those don't have a known target yet).
    n_folds = 5 if len(full_df) > 1500 else 3
    min_train = max(200, len(full_df) // 3)
    try:
        report = walk_forward_backtest(full_df, n_folds=n_folds, min_train_size=min_train)
    except ValueError:
        # Not enough data for the requested fold count - fall back to fewer folds
        report = walk_forward_backtest(full_df, n_folds=2, min_train_size=max(100, len(full_df) // 3))

    # Train on ALL available history for the actual live forecast - more
    # data is strictly better for the deployed prediction itself; the
    # walk-forward folds above exist purely to measure honest accuracy,
    # not to produce the forecast.
    final_model = train_model(X, y)
    latest_features = df[FEATURE_COLUMNS].iloc[[-1]]
    pred = final_model.predict(latest_features)[0]
    proba = final_model.predict_proba(latest_features)[0]
    confidence = float(max(proba)) * 100

    forecast = Forecast(
        symbol=symbol,
        data_mode=mode,
        direction="up" if pred == 1 else "down",
        confidence_pct=round(confidence, 1),
        last_close=round(float(df["Close"].iloc[-1]), 4),
        last_date=str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        backtest_mean_accuracy=report.mean_model_accuracy,
        backtest_naive_baseline=report.mean_naive_baseline_accuracy,
        beats_baseline=report.beats_baseline,
        n_backtest_folds=report.n_folds,
    )
    return forecast, df
