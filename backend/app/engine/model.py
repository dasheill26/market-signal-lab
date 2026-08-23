"""
model.py

Deliberately NOT an LSTM/deep learning model, despite this being an
otherwise ML-heavy portfolio. Two real reasons:

1. The DeepFace project already proved what happens when you reach for
   a heavy deep learning stack without checking the resource cost first
   (the age model alone measured at 3.5GB peak RAM). A gradient-boosted
   tree model on ~15 engineered tabular features needs a few MB, trains
   in under a second, and - for this kind of tabular, non-sequential-raw
   feature set - generally performs comparably to or better than an LSTM
   in practice, without the training instability or resource cost.
2. It's more honest. A deep model on this little data (a few thousand
   daily bars) is far more prone to overfitting and looking impressive
   in-sample while adding no real signal out-of-sample - exactly the
   failure mode this whole project is designed to avoid pretending
   doesn't exist.
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from .features import add_features, make_target, FEATURE_COLUMNS


def prepare_dataset(raw_df: pd.DataFrame, horizon: int = 1):
    """Raw OHLCV -> (X, y, full_df_with_features), rows with any NaN
    feature (the first ~30 rows, from rolling windows) dropped."""
    df = add_features(raw_df)
    df["target"] = make_target(df, horizon=horizon)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    return X, y, df


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=150, max_depth=4, learning_rate=0.05, random_state=42,
    )
    model.fit(X_train, y_train)
    return model
