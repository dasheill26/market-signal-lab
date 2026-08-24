"""
cache.py

A plain in-memory TTL cache for forecast results, keyed by symbol.
Recomputing an identical walk-forward backtest + calibrated model on
unchanged data on every single page load or symbol switch is pure
waste - measured directly at 5.3s per request, not a guess. Same
principle as the hash-based change-detection skip logic in an earlier
project in this portfolio (lead-reconciliation-agent): don't redo work
when nothing has changed.

Deliberately simple (a dict with timestamps), not Redis or a real cache
backend - this app has one process and modest traffic; reaching for
external infrastructure here would be solving a problem this project
doesn't have, the same over-engineering trap "add every technology
possible" leads to.
"""

import time
import threading

_cache = {}
_lock = threading.Lock()
TTL_SECONDS = 300  # 5 minutes - long enough to avoid redundant recomputation on repeated views, short enough that "live" data doesn't go too stale


def get_cached(key: str):
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del _cache[key]
            return None
        return value


def set_cached(key: str, value, ttl: int = TTL_SECONDS):
    with _lock:
        _cache[key] = (value, time.time() + ttl)


def cache_stats() -> dict:
    with _lock:
        return {"entries": len(_cache), "keys": list(_cache.keys())}
