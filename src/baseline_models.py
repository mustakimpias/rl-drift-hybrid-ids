"""Static (batch) baseline classifiers: Random Forest, Extra Trees, Gradient
Boosting, Logistic Regression, and XGBoost/LightGBM when installed.

These serve as the "Static ML-only IDS" arm of the hybrid comparison and as
the classifier the rule/RL layers wrap in hybrid_ids.py.
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .evaluation import compute_metrics
from .utils import optional_import, timer

logger = logging.getLogger("rl_drift_ids")

xgboost = optional_import("xgboost")
lightgbm = optional_import("lightgbm")


def get_baseline_models(
    random_state: int = 42,
    n_estimators: int = 200,
    use_xgboost: bool = True,
    use_lightgbm: bool = True,
) -> dict[str, Any]:
    """Build the dict of {model_name: unfitted estimator} to benchmark.

    XGBoost/LightGBM are included only if both requested and importable —
    the pipeline must keep working when they aren't installed.
    """
    models: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
    }

    if use_xgboost and xgboost is not None:
        models["xgboost"] = xgboost.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    elif use_xgboost:
        logger.warning("xgboost not installed; skipping XGBoost baseline.")

    if use_lightgbm and lightgbm is not None:
        models["lightgbm"] = lightgbm.LGBMClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1, verbose=-1
        )
    elif use_lightgbm:
        logger.warning("lightgbm not installed; skipping LightGBM baseline.")

    return models


def train_and_evaluate_baselines(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    random_state: int = 42,
    n_estimators: int = 200,
    use_xgboost: bool = True,
    use_lightgbm: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit each baseline model, evaluate it, and return (metrics_rows, fitted_models)."""
    models = get_baseline_models(random_state, n_estimators, use_xgboost, use_lightgbm)
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        logger.info("Training baseline model: %s", name)
        with timer() as train_t:
            model.fit(X_train, y_train)
        with timer() as infer_t:
            y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["model"] = name
        metrics["train_time_sec"] = train_t["seconds"]
        metrics["inference_time_sec"] = infer_t["seconds"]
        rows.append(metrics)
        fitted[name] = model
        logger.info(
            "%s: f1=%.4f acc=%.4f fpr=%.4f train=%.2fs",
            name, metrics["f1_score"], metrics["accuracy"], metrics["false_positive_rate"],
            train_t["seconds"],
        )

    return rows, fitted
