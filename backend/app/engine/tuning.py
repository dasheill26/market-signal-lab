"""
tuning.py

Hyperparameter search using TimeSeriesSplit, not standard KFold cross-
validation. Standard KFold shuffles and splits randomly, which for
time-series data means a fold can train on data from AFTER the point
it's tested on - the same future-leakage mistake the whole backtest.py
module exists to avoid, just reintroduced through the back door via
whatever CV strategy tunes the hyperparameters. TimeSeriesSplit's fold
boundaries are verified directly (see the test for this module) to
never let training data come after test data in any fold.

RandomizedSearchCV rather than GridSearchCV - a reasonably-sized random
sample of the hyperparameter space in a fixed time budget, rather than
exhaustively evaluating every combination, which would be slow enough
to not be practical inside a live API call.
"""

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

PARAM_DISTRIBUTIONS = {
    "max_iter": [50, 100, 150, 200],
    "max_depth": [3, 4, 5, 6, None],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
}


def tune_hyperparameters(X, y, n_splits: int = 4, n_iter: int = 15, random_state: int = 42) -> dict:
    """Returns the best hyperparameter combination found, plus the CV
    score it achieved - both reported honestly to the caller rather than
    just silently baked into the final model."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_model = HistGradientBoostingClassifier(random_state=random_state)

    search = RandomizedSearchCV(
        base_model, PARAM_DISTRIBUTIONS,
        n_iter=n_iter, cv=tscv, scoring="accuracy",
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X, y)

    return {
        "best_params": search.best_params_,
        "best_cv_accuracy": round(float(search.best_score_), 4),
    }
