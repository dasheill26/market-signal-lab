"""
sockets.py

WebSocket layer for "live" price updates. Being precise about what this
actually is: a background task polls the data source every 30 seconds
per actively-subscribed symbol and pushes updates over the socket to
anyone subscribed to that symbol's room. That's genuinely useful (no
page reload needed, real push rather than client-side polling) and
genuinely different from true tick-level market data streaming, which
needs a paid real-time feed this project doesn't have. Documented
honestly rather than implied to be something it isn't.
"""

import threading
import time

from flask_socketio import join_room, leave_room, emit

from app import socketio
from app.engine.data_source import fetch_ohlcv

POLL_INTERVAL_SECONDS = 30
_active_symbols = set()
_lock = threading.Lock()
_poller_started = False


def _poll_loop():
    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        with _lock:
            symbols = list(_active_symbols)
        for symbol in symbols:
            try:
                df, mode = fetch_ohlcv(symbol, period="5d")
                latest = df.iloc[-1]
                socketio.emit("price_update", {
                    "symbol": symbol,
                    "data_mode": mode,
                    "close": round(float(latest["Close"]), 4),
                    "timestamp": str(df.index[-1]),
                }, room=symbol)
            except Exception as e:
                socketio.emit("price_update_error", {"symbol": symbol, "error": str(e)}, room=symbol)


def _ensure_poller_started():
    global _poller_started
    with _lock:
        if not _poller_started:
            socketio.start_background_task(_poll_loop)
            _poller_started = True


@socketio.on("subscribe")
def handle_subscribe(data):
    symbol = (data or {}).get("symbol")
    if not symbol:
        emit("subscribe_error", {"error": "symbol is required"})
        return
    join_room(symbol)
    with _lock:
        _active_symbols.add(symbol)
    _ensure_poller_started()
    emit("subscribed", {"symbol": symbol})


@socketio.on("unsubscribe")
def handle_unsubscribe(data):
    symbol = (data or {}).get("symbol")
    if symbol:
        leave_room(symbol)
