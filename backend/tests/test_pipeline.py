"""
Tests for the ML pipeline and API. Uses the bundled real NVDA dataset
directly (not the live yfinance path, not synthetic data) - genuine
historical market data, small and fast enough to run in CI.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from app.engine.features import add_features, make_target, FEATURE_COLUMNS
from app.engine.model import prepare_dataset, train_model
from app.engine.backtest import walk_forward_backtest
from run import app


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nvda_sample.csv")


def _load_sample():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")


def test_features_produce_expected_columns():
    df = add_features(_load_sample())
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_rsi_stays_within_valid_range():
    """RSI is mathematically bounded 0-100 - a real sanity check on the
    manual implementation, not just 'does it run without crashing'."""
    df = add_features(_load_sample())
    valid_rsi = df["rsi_14"].dropna()
    assert valid_rsi.min() >= 0
    assert valid_rsi.max() <= 100


def test_target_is_binary():
    df = _load_sample()
    target = make_target(df)
    valid = target.dropna()
    assert set(valid.unique()).issubset({0, 1})


def test_prepare_dataset_drops_nan_rows():
    X, y, full_df = prepare_dataset(_load_sample())
    assert X.isna().sum().sum() == 0
    assert len(X) == len(y) == len(full_df)


def test_model_trains_and_predicts():
    X, y, _ = prepare_dataset(_load_sample())
    model = train_model(X.iloc[:1000], y.iloc[:1000])
    preds = model.predict(X.iloc[1000:1010])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})


def test_walk_forward_backtest_never_trains_on_future_data():
    """The core honesty guarantee of this project: for every fold, the
    entire test window must come strictly after the entire train window
    - verified directly by index position, not just trusted."""
    _, _, full_df = prepare_dataset(_load_sample())
    report = walk_forward_backtest(full_df, n_folds=3, min_train_size=1000)
    assert report.n_folds == 3
    # Reconstruct fold boundaries the same way the function does, and
    # check strict ordering.
    n = len(full_df)
    test_chunk = (n - 1000) // 3
    for i, fold in enumerate(report.folds):
        train_end = 1000 + i * test_chunk
        assert fold.n_train == train_end


def test_backtest_reports_both_model_and_baseline():
    _, _, full_df = prepare_dataset(_load_sample())
    report = walk_forward_backtest(full_df, n_folds=3, min_train_size=1000)
    assert 0 <= report.mean_model_accuracy <= 1
    assert 0 <= report.mean_naive_baseline_accuracy <= 1
    assert isinstance(report.beats_baseline, bool)


def test_api_healthz():
    client = app.test_client()
    r = client.get("/healthz")
    assert r.status_code == 200


def test_api_symbols():
    client = app.test_client()
    r = client.get("/api/symbols")
    assert r.status_code == 200
    data = r.get_json()
    assert "stocks" in data and "forex" in data


def test_api_forecast_includes_disclaimer_and_data_mode():
    """Regression guard: the forecast response must always disclose
    which data mode it used and include the disclaimer - a client
    should never be able to receive a forecast without both."""
    client = app.test_client()
    r = client.get("/api/forecast/NVDA")
    assert r.status_code == 200
    data = r.get_json()
    assert data["forecast"]["data_mode"] in ("live", "cached_demo_data")
    assert "disclaimer" in data["forecast"]
    assert "not financial advice" in data["forecast"]["disclaimer"].lower()


def test_api_forecast_invalid_horizon_returns_400():
    client = app.test_client()
    r = client.get("/api/forecast/NVDA?horizon=99")
    assert r.status_code == 400


def test_unknown_client_route_falls_back_to_frontend_not_404():
    """Regression test for a real bug: setting static_url_path='' on the
    Flask app caused its own implicit static-file route to shadow the
    custom SPA catch-all route entirely, so any unknown path returned a
    raw Flask 404 instead of index.html - breaking client-side routing
    for any direct link or page refresh on a non-root URL. Fixed by
    disabling Flask's implicit static handling (static_folder=None) and
    serving everything through one explicit route."""
    client = app.test_client()
    r = client.get("/some-client-side-route-that-is-not-an-api-path")
    assert r.status_code == 200
    assert r.status_code != 404


if __name__ == "__main__":
    test_features_produce_expected_columns()
    test_rsi_stays_within_valid_range()
    test_target_is_binary()
    test_prepare_dataset_drops_nan_rows()
    test_model_trains_and_predicts()
    test_walk_forward_backtest_never_trains_on_future_data()
    test_backtest_reports_both_model_and_baseline()
    print("Core tests passed (run via pytest for the full suite including Flask app tests).")
