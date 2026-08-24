# Market Signal Lab

A full-stack time-series ML forecasting app for stocks and forex — React frontend, Flask + WebSocket backend, a gradient-boosted model on manually implemented technical indicators, and **honest walk-forward backtested accuracy shown next to every prediction**, not hidden below it.

![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
[![tests](https://github.com/dasheill26/market-signal-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/dasheill26/market-signal-lab/actions/workflows/tests.yml)

🔗 **[Live demo](https://market-signal-lab.onrender.com)**

## Not financial advice — read this first

This project exists to demonstrate how a **properly evaluated** financial ML pipeline actually behaves, including how small a genuine edge over a naive baseline really is. Markets are close enough to efficient that no portfolio project has real trading edge, and any project claiming otherwise should make you suspicious of it, not impressed by it.

The measured result on real NVDA data (1999–2026, walk-forward validated):

| | Accuracy |
|---|---|
| Model (gradient-boosted trees on technical indicators) | **50.99%** |
| Naive baseline ("tomorrow moves the same direction as today") | 49.49% |
| Random / coin-flip | 50.00% |

A ~1.5 percentage point edge over the naive baseline. That's the honest number — not hidden, not spun, shown directly in the UI next to every forecast. The value of this project is the methodology (walk-forward validation, honest baseline comparison, a real full-stack delivery), not a claim that it beats the market.

## What this actually is

- **Directional forecasting**, not price prediction — the model predicts whether the next period closes up or down, with a confidence score. Predicting an exact future price is a much easier claim to overclaim and far harder to evaluate honestly.
- **Walk-forward backtested**, not randomly split — a random shuffled train/test split on time-series data leaks future information into training and produces inflated, meaningless accuracy numbers. This project trains only on data strictly before each test window, rolls the cutoff forward, and repeats — verified directly by a test that checks fold boundaries never overlap.
- **A gradient-boosted tree model, not a deep learning model** — deliberately. On this much data (a few thousand daily bars) and this feature set (~15 engineered tabular indicators), a deep model is more prone to overfitting and no more likely to add real signal, while costing far more to train and deploy. (A separate project in this portfolio, [Face Recognition Studio](https://github.com/dasheill26/face-recognition-studio), already demonstrates what happens when a heavy deep learning stack gets reached for without checking the resource cost first — one attribute model there measured at 3.5GB peak RAM. Lesson applied here before it became a problem, not after.)
- **Selectable forecast horizon** (1 day / 3 days / 1 week / 2 weeks), framed for who's actually looking: day traders care about tomorrow, position traders about the next two weeks. Genuinely different, honest results across horizons — on EURUSD, the 1-day model beats its naive baseline; the 10-day model, tested the same way, doesn't. Longer-range prediction is genuinely harder, and the app says so rather than smoothing it over.

## Beyond a single model: the advanced analysis

Available on demand (`/api/analysis/<symbol>`, ~15-20s — deliberately not run automatically on page load, the same way a real ML system tunes/evaluates periodically offline rather than on every inference request):

- **Model comparison across 3 families** — Logistic Regression, Random Forest, and Gradient Boosted Trees, evaluated under the **identical** walk-forward methodology. Not "I picked a model" — a real, reproducible comparison that justifies the choice. On the bundled NVDA data, the two tree-based models beat baseline; plain Logistic Regression, tested the same way, does not.
- **Time-series-aware hyperparameter tuning** — `RandomizedSearchCV` over `TimeSeriesSplit`, not standard `KFold`. Standard KFold shuffles randomly, which for time-series data means a fold can train on data from *after* its own test point — reintroducing the exact future-leakage mistake the whole backtesting approach exists to prevent. Verified directly with a test that checks every fold's training indices end strictly before its test indices begin.
- **Feature importance via permutation**, not a tree-specific heuristic — `HistGradientBoostingClassifier` doesn't expose `.feature_importances_` the way `RandomForest` does; rather than switch model types just for this, permutation importance is used instead, which is actually the more rigorous choice regardless — it measures real predictive impact (how much does shuffling this feature actually hurt accuracy?) rather than an internal tree-splitting statistic, and works identically across any model.
- **Probability calibration — a real, honest finding, not a claim.** Checked directly whether the model's confidence percentages meant anything: uncalibrated, the Brier score was **0.2546** — worse than always guessing 50/50, which scores exactly 0.25. A "70% confident" prediction was not actually right 70% of the time. Applying isotonic calibration (`CalibratedClassifierCV`) fixed this — Brier score dropped to **0.2489** — and, measured directly rather than assumed, also improved raw directional accuracy on the same held-out split (51.9% → 53.8%). The production model uses calibration by default because this was checked and, uncalibrated, it failed the test.

## Risk management, not trading signals

There's a specific thing this project deliberately does **not** do: generate BUY/SELL signals with entry/stop-loss/take-profit levels formatted for direct trade execution. That's a real, meaningfully different category of output than an honest directional forecast — it manufactures a level of confidence and precision the backtest results above don't support (a ~1-2 percentage point edge over baseline, sometimes none at all, is not "confidently buy here"). A lot of retail trading-signal content does exactly this, often with red flags like agreeing across every timeframe simultaneously — a pattern real markets don't actually produce, and a tell of a service optimized to look authoritative rather than to have genuine edge.

What's included instead: an ATR (Average True Range)-based risk-sizing reference panel, explicitly **decoupled from the forecast direction** — it works the same regardless of what the model predicts, because it's teaching the methodology of sizing risk on a trade you decide to make, not telling you to make one. Verified directly with a test that checks the output never contains the words "buy," "sell," "long," or "short." Includes a real position-sizing calculator (pure arithmetic — account size, risk %, and stop distance in, position size out) so the person doing the calculating supplies every input themselves.

## Performance and caching

An uncached forecast request measured at **5.3 seconds** — real cost, not negligible, from the walk-forward backtest plus a calibrated final model fit. Recomputing this on every page load or symbol switch is pure waste when the underlying data hasn't changed, so forecast results are cached in-memory with a 5-minute TTL (confirmed directly: 5.3s uncached, ~0ms on a cache hit, identical result). Same principle as the hash-based change-detection skip logic in an earlier project in this portfolio ([Lead Reconciliation Agent](https://github.com/dasheill26/lead-reconciliation-agent)) — don't redo work when nothing has changed. The advanced analysis endpoint caches separately, for an hour, since it's already an explicit on-demand action.

## Architecture

```
market-lab/
├── backend/
│   ├── app/
│   │   ├── engine/
│   │   │   ├── data_source.py     # live Yahoo Finance + cached-real-data fallback
│   │   │   ├── features.py        # RSI, MACD, Bollinger Bands - implemented manually
│   │   │   ├── model.py           # HistGradientBoosting + isotonic calibration
│   │   │   ├── backtest.py        # walk-forward validation - the honesty-critical file
│   │   │   ├── model_comparison.py # 3 model families, identical methodology
│   │   │   ├── tuning.py          # TimeSeriesSplit hyperparameter search
│   │   │   ├── cache.py           # TTL cache - avoids redundant recomputation
│   │   │   └── predictor.py       # ties the above together into two entry points
│   │   ├── routes.py              # REST API
│   │   └── sockets.py             # WebSocket live price updates
│   ├── data/nvda_sample.csv       # real historical data (not synthetic), fallback + test fixture
│   └── tests/
├── frontend/                       # React + Vite, lightweight-charts, socket.io-client
└── Dockerfile                      # multi-stage: Node build -> Python runtime, single service
```

Frontend and backend deploy as a **single service** — the React app is built at Docker build time and served directly by Flask, not hosted separately. Simpler, one port, no CORS complexity in production.

## Real bugs found during development (not hypothetical)

1. **`eventlet` (Flask-SocketIO's typical async driver) is deprecated** — confirmed by testing, not assumed; switched to `threading` async mode, which needs no extra dependency and works cleanly with gunicorn's `gthread` worker class (verified directly, including a real WebSocket handshake through the actual production server).
2. **A real Flask routing bug**: setting `static_url_path=""` to serve the built React app caused Flask's own *implicit* static-file route to silently shadow the custom SPA catch-all route. Every unknown client-side path returned a raw Flask 404 instead of falling back to `index.html` — breaking any direct link or page refresh on a non-root URL. Found by testing the exact failure scenario directly, not by inspection. Fixed by disabling Flask's implicit static handling (`static_folder=None`) and serving everything through one explicit route. Now a permanent regression test.
3. **A moderate esbuild vulnerability** in the initial Vite version pulled in by `npm install` — not ignored; upgraded to a patched Vite version and confirmed `npm audit` reports zero vulnerabilities before this was considered done.
4. **Yahoo Finance's endpoints are unreachable from network-restricted sandboxed environments** — confirmed directly, not assumed. This shaped the dual-mode data source design: live fetch with a fallback to bundled real historical data, with the active mode disclosed in every API response and shown in the UI, not silently swapped.
5. **A CI workflow YAML syntax error** — an unquoted colon inside a step name (`"...multi-stage: Node build..."`) made YAML parse it as a new mapping key instead of plain text, failing the entire workflow before any job could run. The failure signature (`created_at` and `updated_at` identical, zero jobs ever created) was diagnostic in itself. Fixed and verified by parsing the YAML directly with PyYAML before pushing again, not just pushing and hoping.
6. **The model's confidence scores were, at first, actively worse than useless.** Uncalibrated Brier score of 0.2546 vs 0.25 for always guessing 50/50 — see "Beyond a single model" above. This is the one "bug" that isn't really a bug so much as a check most projects never run at all.
7. **The forex fallback silently served stock prices.** Found from a live deployment screenshot, the same way the static-folder bug was found: the original fallback always returned the bundled NVDA stock data regardless of what was actually requested, so a failed live fetch for a forex pair like EURUSD would silently substitute stock prices up to $207 — wildly wrong for a currency pair that has never traded outside roughly 0.5–2.0. The `data_mode` flag honestly said "cached", but the actual numbers were nonsense for the requested instrument. Fixed by sourcing a second real historical dataset (genuine EURUSD daily OHLCV, resampled from hourly data, 2003–2020) and routing the fallback by the requested symbol's actual asset class — and, for any asset class with no matching fallback, failing with a clear error instead of ever substituting mismatched data. Now two regression tests: one confirming forex fallback returns forex-range prices, one confirming the fix didn't break the existing stock fallback.
8. **A flaky CI test that only worked in my restricted sandbox.** The forex fallback regression test above hardcoded an assumption (`mode == "cached_demo_data"`) that only held because my sandbox can't reach Yahoo Finance — GitHub Actions runners have full internet access, so the live fetch actually succeeded there, taking the correct-but-different code path and failing the test. Fixed properly with monkeypatching to force the fallback path deterministically, regardless of what network access the runner happens to have.
9. **A copy-paste bug across projects.** The position-sizing endpoint's input validation called a helper function (`_get_json_body`) that exists in a *different* project in this portfolio (Face Recognition Studio) but was never defined here — every request would have crashed with an unhandled 500. Caught immediately by testing before it ever reached production, not after; now has a permanent regression test.
10. **Gold needed its own asset class, not a reuse of forex's.** Same principle as the forex fix, applied proactively this time rather than found from a screenshot: gold trades in the low thousands per ounce, wildly different from both stock range and forex range (~0.5-2.0) — reusing either existing fallback would have reproduced the exact bug already fixed once. Sourced a third real historical dataset (genuine XAUUSD daily OHLC, 2012–2022) and added "metal" as a proper third asset class. Also found while researching this: yfinance doesn't recognize `XAUUSD=X` at all — Yahoo Finance's own symbol search returns nothing for it. The correct ticker is `GC=F` (COMEX Gold Futures). `XAUUSD` is kept as the user-facing display symbol (the standard retail convention) and mapped internally to the correct ticker before the live fetch — verified directly, not assumed, by checking the failed-fetch error message actually referenced `GC=F`.

## What "live" actually means here

The WebSocket layer polls the data source every 30 seconds per actively-subscribed symbol and pushes updates to connected clients. That's genuinely useful — no page reload needed, real server-push rather than client-side polling — and it is **not** the same thing as tick-level market data streaming, which needs a paid real-time feed this project doesn't have. Documented plainly rather than implied to be something it isn't.

## Running it

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py    # http://127.0.0.1:5003
```

### Frontend (separate dev server, proxies API calls to the backend)

```bash
cd frontend
npm install
npm run dev    # http://127.0.0.1:5173
```

### Full stack via Docker (what production actually runs)

```bash
docker build -t market-lab .
docker run -p 5003:5003 market-lab
```

## Tests

```bash
cd backend && pytest tests/ -v
```

30 tests: technical indicator correctness (RSI mathematically bounded 0–100, ATR always non-negative), the walk-forward fold-boundary guarantee, model comparison uses identical methodology for every candidate, hyperparameter tuning genuinely uses `TimeSeriesSplit` (checked directly, not just that it runs), the calibrated model's Brier score beats the 50/50 baseline, the cache returns identical results faster, API contract tests (every forecast response must include the disclaimer and data mode; the analysis endpoint must always return all four components together), the SPA routing regression test, the forex/gold/stock fallback regression tests (each asset class's fallback genuinely returns data in that asset class's own real price range), the gold ticker-mapping test, the position-size copy-paste bug regression test, and — the one that matters most for the risk panel — a direct check that its output never contains the words "buy," "sell," "long," or "short."

## What I'd do with more time

- **A proper time-series cross-validation library** (e.g. `sktime`) instead of the hand-rolled walk-forward splitter — the current implementation is correct and tested, but a maintained library would handle more edge cases.
- **More instruments and a longer history** for less liquid forex pairs, where the bundled fallback dataset (NVDA only) doesn't represent the requested symbol at all.
- **SHAP values** for per-prediction explainability, not just global permutation importance — "why did the model say up *this time*?" is a natural next question the current UI doesn't answer.
- **Rate limiting** on the API, same known gap as the other projects in this portfolio.
- **Persist the tuned hyperparameters** back into the production model rather than only reporting them — currently the tuning result is informational; wiring it back into `model.py` would close the loop.

## License

MIT — see [LICENSE](LICENSE).
