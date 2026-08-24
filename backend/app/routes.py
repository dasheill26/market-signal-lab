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


def pd_isna(val):
    import pandas as pd
    return pd.isna(val)
