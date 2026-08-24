"""
predictor.py

Two entry points, deliberately different speeds for different purposes:

  - run_forecast(): the fast path. Cached (5 min TTL) since recomputing
    an identical backtest on unchanged data on every page load measured
    at 5.3s - pure waste. This is what loads by default.
  - run_advanced_analysis(): the slow, honest-but-expensive path -
    model comparison across 3 model families, time-series-aware
    hyperparameter tuning, permutation feature importance, and a
    probability calibration check. ~15-20s combined. Deliberately NOT
    run automatically on every page load - it's triggered on demand,
    the same way a real ML system retrains/tunes periodically offline
    rather than on every single inference request.
"""

from dataclasses import dataclass, asdict

from .data_source import fetch_ohlcv
from .features import add_features, FEATURE_COLUMNS
from .model import prepare_dataset, train_model
from .backtest import walk_forward_backtest
from .model_comparison import compare_models
from .tuning import tune_hyperparameters
from .cache import get_cached, set_cached
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


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
    cache_key = f"forecast:{symbol}:{horizon}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    raw_df, mode = fetch_ohlcv(symbol)
    df = add_features(raw_df)

    X, y, full_df = prepare_dataset(raw_df, horizon=horizon)

    n_folds = 5 if len(full_df) > 1500 else 3
    min_train = max(200, len(full_df) // 3)
    try:
        report = walk_forward_backtest(full_df, n_folds=n_folds, min_train_size=min_train)
    except ValueError:
        report = walk_forward_backtest(full_df, n_folds=2, min_train_size=max(100, len(full_df) // 3))

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
    result = (forecast, df)
    set_cached(cache_key, result)
    return result


def run_advanced_analysis(symbol: str) -> dict:
    """The slow, comprehensive path - not cached as aggressively (1 hour,
    vs 5 min for the fast forecast) since it's already an explicit,
    deliberate on-demand action, not something a user triggers by
    casually switching symbols."""
    cache_key = f"analysis:{symbol}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    raw_df, mode = fetch_ohlcv(symbol)
    X, y, full_df = prepare_dataset(raw_df)

    n_folds = 3
    min_train = max(500, len(full_df) // 3)

    # 1. Model comparison - same honest methodology across 3 model families
    candidates = compare_models(full_df, n_folds=n_folds, min_train_size=min_train)

    # 2. Hyperparameter tuning via TimeSeriesSplit (temporally correct CV)
    tuning_result = tune_hyperparameters(X, y, n_splits=4, n_iter=15)

    # 3. Feature importance via permutation (model-agnostic, measures real
    #    predictive impact rather than an internal tree-splitting heuristic)
    split = int(len(X) * 0.8)
    model = train_model(X.iloc[:split], y.iloc[:split])
    importance_result = permutation_importance(
        model, X.iloc[split:], y.iloc[split:], n_repeats=10, random_state=42, n_jobs=-1,
    )
    feature_importance = sorted(
        [{"feature": f, "importance": round(float(imp), 5)}
         for f, imp in zip(FEATURE_COLUMNS, importance_result.importances_mean)],
        key=lambda x: -x["importance"],
    )

    # 4. Calibration check - is model confidence actually trustworthy?
    proba = model.predict_proba(X.iloc[split:])[:, 1]
    brier = brier_score_loss(y.iloc[split:], proba)
    prob_true, prob_pred = calibration_curve(y.iloc[split:], proba, n_bins=8, strategy="uniform")
    calibration_points = [
        {"predicted": round(float(p), 3), "actual": round(float(a), 3)}
        for p, a in zip(prob_pred, prob_true)
    ]

    result = {
        "symbol": symbol,
        "data_mode": mode,
        "model_comparison": [
            {"name": c.name, "description": c.description, "accuracy_pct": round(c.mean_accuracy * 100, 2),
             "beats_baseline": c.beats_baseline}
            for c in candidates
        ],
        "tuning": {
            "best_params": tuning_result["best_params"],
            "best_cv_accuracy_pct": round(tuning_result["best_cv_accuracy"] * 100, 2),
        },
        "feature_importance": feature_importance,
        "calibration": {
            "brier_score": round(float(brier), 4),
            "brier_score_naive_50_50": 0.25,
            "well_calibrated": brier < 0.25,
            "curve": calibration_points,
            "note": (
                "Brier score below 0.25 means the model's confidence percentages are more "
                "trustworthy than just always guessing 50/50 - above 0.25 means they're actively "
                "worse than not having a confidence score at all. Calibration (isotonic regression) "
                "is applied to the production model specifically because this was checked and, "
                "uncalibrated, it failed this test."
            ),
        },
    }
    set_cached(cache_key, result, ttl=3600)
    return result
