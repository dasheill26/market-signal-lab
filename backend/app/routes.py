"""
routes.py

REST endpoints. Every response includes data_mode ('live' or
'cached_demo_data') explicitly, in every payload that carries price
data - not just logged server-side - so the frontend can and does show
the user which mode they're looking at, rather than silently presenting
demo data as if it were live.
"""

from flask import Blueprint, jsonify, request

from app.engine.data_source import SUPPORTED_SYMBOLS
from app.engine.predictor import run_forecast, run_advanced_analysis
from app.engine.risk_education import build_risk_education, position_size as calc_position_size
from app.engine.data_source import fetch_ohlcv

bp = Blueprint("main", __name__)


@bp.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


@bp.route("/api/symbols")
def get_symbols():
    return jsonify(SUPPORTED_SYMBOLS)


@bp.route("/api/forecast/<symbol>")
def get_forecast(symbol):
    horizon = request.args.get("horizon", default=1, type=int)
    if horizon < 1 or horizon > 10:
        return jsonify({"error": "horizon must be between 1 and 10"}), 400

    try:
        forecast, df = run_forecast(symbol, horizon=horizon)
    except Exception as e:
        return jsonify({"error": f"Could not generate a forecast for '{symbol}': {e}"}), 422

    # Chart data: last 180 periods, kept small and JSON-friendly
    chart_df = df.tail(180)
    chart_data = [
        {
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]) if not pd_isna(row["Volume"]) else 0,
        }
        for idx, row in chart_df.iterrows()
    ]

    return jsonify({
        "forecast": forecast.__dict__,
        "chart": chart_data,
    })


@bp.route("/api/analysis/<symbol>")
def get_analysis(symbol):
    """The slow, comprehensive path: model comparison, hyperparameter
    tuning, feature importance, calibration check. ~15-20s uncached -
    deliberately not called automatically by the frontend on page load,
    only on explicit user action, since a page shouldn't force a 20s
    wait for something most visitors won't look at."""
    try:
        result = run_advanced_analysis(symbol)
    except Exception as e:
        return jsonify({"error": f"Could not run analysis for '{symbol}': {e}"}), 422
    return jsonify(result)


@bp.route("/api/risk-education/<symbol>")
def get_risk_education(symbol):
    """ATR-based risk-sizing reference data - deliberately decoupled from
    the forecast direction. Fast (no model training), just a volatility
    calculation plus illustrative reference multiples."""
    try:
        df, mode = fetch_ohlcv(symbol)
        current_price = float(df["Close"].iloc[-1])
        result = build_risk_education(df, current_price)
        result["symbol"] = symbol
        result["data_mode"] = mode
    except Exception as e:
        return jsonify({"error": f"Could not compute risk data for '{symbol}': {e}"}), 422
    return jsonify(result)


@bp.route("/api/position-size", methods=["POST"])
def api_position_size():
    """Pure arithmetic, no model or market data involved - the same
    formula any risk-management course teaches. A calculator, not a
    recommendation: the person supplies every input themselves."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        account_size = float(data.get("account_size", 0))
        risk_pct = float(data.get("risk_pct", 0))
        stop_distance = float(data.get("stop_distance", 0))
        price_per_unit = float(data.get("price_per_unit", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "account_size, risk_pct, stop_distance, and price_per_unit must be numbers"}), 400

    if account_size <= 0 or risk_pct <= 0 or risk_pct > 100:
        return jsonify({"error": "account_size must be positive and risk_pct must be between 0 and 100"}), 400

    try:
        result = calc_position_size(account_size, risk_pct, stop_distance, price_per_unit)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


def pd_isna(val):
    import pandas as pd
    return pd.isna(val)
