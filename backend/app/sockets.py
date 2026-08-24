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

Real bug found and fixed: unsubscribe only called leave_room(), never
removing the symbol from _active_symbols - so the backend kept polling
every symbol ever visited in the app's lifetime, forever, even with
zero remaining subscribers. Fixed with per-symbol subscriber counting:
a symbol only leaves _active_symbols (stops being polled) once its
count reaches zero. Also handles the case a naive fix would miss - a
client closing the browser tab without ever calling unsubscribe -  by
tracking which symbols each connected session subscribed to and
cleaning those up on disconnect too.
"""

import threading
import time

from flask_socketio import join_room, leave_room, emit
from flask import request

from app import socketio
from app.engine.data_source import fetch_ohlcv

POLL_INTERVAL_SECONDS = 30
_active_symbols = set()
_symbol_subscriber_counts = {}   # symbol -> count of sessions subscribed
_session_symbols = {}            # session_id -> set of symbols that session subscribed to
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


def _add_subscription(session_id: str, symbol: str):
    with _lock:
        _session_symbols.setdefault(session_id, set()).add(symbol)
        _symbol_subscriber_counts[symbol] = _symbol_subscriber_counts.get(symbol, 0) + 1
        _active_symbols.add(symbol)


def _remove_subscription(session_id: str, symbol: str):
    with _lock:
        session_syms = _session_symbols.get(session_id)
        if session_syms and symbol in session_syms:
            session_syms.discard(symbol)
        count = _symbol_subscriber_counts.get(symbol, 0) - 1
        if count <= 0:
            _symbol_subscriber_counts.pop(symbol, None)
            _active_symbols.discard(symbol)
        else:
            _symbol_subscriber_counts[symbol] = count


@socketio.on("subscribe")
def handle_subscribe(data):
    symbol = (data or {}).get("symbol")
    if not symbol:
        emit("subscribe_error", {"error": "symbol is required"})
        return
    join_room(symbol)
    _add_subscription(request.sid, symbol)
    _ensure_poller_started()
    emit("subscribed", {"symbol": symbol})


@socketio.on("unsubscribe")
def handle_unsubscribe(data):
    symbol = (data or {}).get("symbol")
    if symbol:
        leave_room(symbol)
        _remove_subscription(request.sid, symbol)


@socketio.on("disconnect")
def handle_disconnect():
    """A client can close the tab without ever calling unsubscribe -
    clean up whatever that session was still subscribed to, so those
    symbols don't stay in _active_symbols forever with a phantom
    subscriber that will never unsubscribe itself."""
    session_id = request.sid
    with _lock:
        symbols = list(_session_symbols.get(session_id, set()))
    for symbol in symbols:
        _remove_subscription(session_id, symbol)
    with _lock:
        _session_symbols.pop(session_id, None)
