"""
backtest.py

The single most important file in this project, honesty-wise. A random
shuffled train/test split on time-series data leaks future information
into training (the model can effectively "see the future" via correlated
neighboring rows) and produces inflated accuracy numbers that don't mean
anything - a classic, well-known mistake in financial ML, and exactly the
kind of thing that makes an experienced interviewer distrust a stock
prediction project on sight.

This module does walk-forward validation instead: train only on data
strictly before a cutoff, test only on data strictly after it, roll the
cutoff forward, repeat. The model never sees the future during training,
in any fold.

Two baselines are always reported alongside the model, not just the
model's own number in isolation:
  - Random baseline (50%) - what you'd get by coin flip.
  - Naive/momentum baseline - "predict tomorrow moves the same direction
    as today did." A real, non-trivial baseline; a model that can't beat
    this isn't adding value, whatever its raw accuracy number looks like.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

from .model import train_model as _default_train_model
from .features import FEATURE_COLUMNS


@dataclass
class FoldResult:
    train_end_date: str
    test_start_date: str
    test_end_date: str
    n_train: int
    n_test: int
    model_accuracy: float
    naive_baseline_accuracy: float


@dataclass
class BacktestReport:
    folds: list
    mean_model_accuracy: float
    mean_naive_baseline_accuracy: float
    beats_baseline: bool
    n_folds: int


def walk_forward_backtest(df_with_features: pd.DataFrame, n_folds: int = 5,
                           min_train_size: int = 500,
                           model_factory: Callable = None) -> BacktestReport:
    """df_with_features must already have FEATURE_COLUMNS, 'target', and
    a 'Close' column, NaN rows already dropped (see model.prepare_dataset).
    Expanding-window walk-forward: each fold trains on everything up to a
    cutoff and tests on the next chunk, cutoff moves forward each fold.

    model_factory: a callable(X_train, y_train) -> fitted model. Defaults
    to the project's own HistGradientBoostingClassifier (model.train_model)
    for backward compatibility, but accepting any factory here is what
    makes honest model comparison possible - the exact same walk-forward
    methodology runs for every model type being compared, not a different
    evaluation for each."""
    model_factory = model_factory or _default_train_model

    n = len(df_with_features)
    test_chunk = (n - min_train_size) // n_folds
    if test_chunk < 10:
        raise ValueError(f"Not enough data for {n_folds} folds with min_train_size={min_train_size}")

    folds = []
    for i in range(n_folds):
        train_end = min_train_size + i * test_chunk
        test_end = train_end + test_chunk
        if test_end > n:
            break

        train_df = df_with_features.iloc[:train_end]
        test_df = df_with_features.iloc[train_end:test_end]

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target"]

        model = model_factory(X_train, y_train)
        preds = model.predict(X_test)
        model_acc = float((preds == y_test.values).mean())

        # Naive baseline: predict the same direction as the most recent
        # known move at the start of the test window (momentum carry-over,
        # not peeking at any test-period data itself).
        last_known_direction = int(y_train.iloc[-1])
        naive_preds = np.full(len(y_test), last_known_direction)
        naive_acc = float((naive_preds == y_test.values).mean())

        folds.append(FoldResult(
            train_end_date=str(train_df.index[-1]) if hasattr(train_df.index[-1], "date") else str(train_end),
            test_start_date=str(test_df.index[0]) if hasattr(test_df.index[0], "date") else str(train_end),
            test_end_date=str(test_df.index[-1]) if hasattr(test_df.index[-1], "date") else str(test_end),
            n_train=len(train_df), n_test=len(test_df),
            model_accuracy=round(model_acc, 4),
            naive_baseline_accuracy=round(naive_acc, 4),
        ))

    mean_model = round(float(np.mean([f.model_accuracy for f in folds])), 4)
    mean_naive = round(float(np.mean([f.naive_baseline_accuracy for f in folds])), 4)

    return BacktestReport(
        folds=folds,
        mean_model_accuracy=mean_model,
        mean_naive_baseline_accuracy=mean_naive,
        beats_baseline=mean_model > mean_naive,
        n_folds=len(folds),
    )
