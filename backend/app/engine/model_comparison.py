"""
model_comparison.py

Compares Logistic Regression, Random Forest, and HistGradientBoosting
under the exact same walk-forward methodology - the same honesty
guarantee (no fold ever trains on future data) applied uniformly across
every candidate, not a different, possibly friendlier evaluation for
whichever model the project happens to ship.

This is what separates "I picked a model" from "I compared approaches
and can justify the choice" - the second is what an experienced ML
interviewer actually wants to see, and it's directly testable: a real
model selection process, not a claim about one.
"""

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from .backtest import walk_forward_backtest


@dataclass
class ModelCandidate:
    name: str
    description: str
    mean_accuracy: float
    beats_baseline: bool


def _logistic_regression_factory(X_train, y_train):
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)
    return model


def _random_forest_factory(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def _hist_gb_factory(X_train, y_train):
    model = HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    return model


MODEL_FACTORIES = {
    "logistic_regression": (
        "Logistic Regression",
        "A simple linear baseline - if a far more complex model can't beat this by a "
        "meaningful margin, the complexity isn't earning its keep.",
        _logistic_regression_factory,
    ),
    "random_forest": (
        "Random Forest",
        "An ensemble of decision trees, each trained on a random subset of data and features.",
        _random_forest_factory,
    ),
    "hist_gradient_boosting": (
        "Gradient Boosted Trees",
        "Trees trained sequentially, each correcting the previous ones' errors - "
        "generally the strongest of the three on tabular data like this.",
        _hist_gb_factory,
    ),
}


def compare_models(df_with_features, n_folds: int = 3, min_train_size: int = 1000) -> list[ModelCandidate]:
    """Runs the identical walk-forward backtest for every candidate model
    and returns results sorted best-first. Fewer folds than the main
    single-model backtest by default (3 vs 5) purely for speed - running
    this 3x (once per model) at full fold count would be needlessly slow
    for a live API call; the comparison itself is still honest, just a
    coarser read on accuracy than the full single-model backtest."""
    results = []
    for key, (name, description, factory) in MODEL_FACTORIES.items():
        report = walk_forward_backtest(
            df_with_features, n_folds=n_folds, min_train_size=min_train_size, model_factory=factory,
        )
        results.append(ModelCandidate(
            name=name, description=description,
            mean_accuracy=report.mean_model_accuracy,
            beats_baseline=report.beats_baseline,
        ))
    results.sort(key=lambda c: c.mean_accuracy, reverse=True)
    return results
