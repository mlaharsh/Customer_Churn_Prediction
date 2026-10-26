"""Model training, evaluation, selection, and artifact serialization."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from churn.config import (
    FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    RISK_BANDS,
    TEST_SIZE,
)
from churn.features import build_preprocessor
from churn.insights import risk_tier

SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "extra_trees": "Extra trees",
}


def candidate_pipelines() -> dict[str, Pipeline]:
    """Return the candidate models with defensible, fixed hyperparameters."""

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=700,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.85,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=16,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _metric_summary(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    prevalence = float(np.mean(y_true))
    top_n = max(1, int(round(len(probabilities) * 0.10)))
    top_indices = np.argsort(probabilities)[-top_n:]
    top_capture = float(np.sum(np.asarray(y_true)[top_indices]) / max(1, np.sum(y_true)))
    top_precision = float(np.mean(np.asarray(y_true)[top_indices]))
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "log_loss": round(float(log_loss(y_true, probabilities)), 4),
        "specificity": round(float(tn / max(1, tn + fp)), 4),
        "negative_predictive_value": round(float(tn / max(1, tn + fn)), 4),
        "alert_rate": round(float(np.mean(predictions)), 4),
        "churners_flagged": int(tp),
        "churners_missed": int(fn),
        "non_churners_contacted": int(fp),
        "top_decile_capture": round(top_capture, 4),
        "top_decile_precision": round(top_precision, 4),
        "top_decile_lift": round(top_precision / prevalence, 2) if prevalence else 0.0,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "prevalence": round(prevalence, 4),
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_index = int(np.argmax(f1_scores[:-1]))
    return round(float(thresholds[best_index]), 3)


def train_and_select(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Cross-validate candidates, select by ROC-AUC, and evaluate on holdout."""

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    comparison: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in candidate_pipelines().items():
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(x_train, y_train)
        test_probabilities = pipeline.predict_proba(x_test)[:, 1]
        test_metrics = _metric_summary(y_test, test_probabilities, 0.5)
        record: dict[str, Any] = {
            "model": name,
            "display_name": MODEL_DISPLAY_NAMES[name],
            "cv_roc_auc": round(float(np.mean(scores["test_roc_auc"])), 4),
            "cv_roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 4),
            "cv_accuracy": round(float(np.mean(scores["test_accuracy"])), 4),
            "cv_precision": round(float(np.mean(scores["test_precision"])), 4),
            "cv_recall": round(float(np.mean(scores["test_recall"])), 4),
            "cv_f1": round(float(np.mean(scores["test_f1"])), 4),
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_f1_at_0_5": test_metrics["f1"],
            "test_recall_at_0_5": test_metrics["recall"],
        }
        comparison.append(record)
        fitted_models[name] = pipeline

    comparison = sorted(comparison, key=lambda item: item["cv_roc_auc"], reverse=True)
    selected_name = comparison[0]["model"]
    selected_pipeline = fitted_models[selected_name]

    oof_probabilities = cross_val_predict(
        clone(selected_pipeline),
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = _best_f1_threshold(y_train, oof_probabilities)

    test_probabilities = selected_pipeline.predict_proba(x_test)[:, 1]
    selected_test_metrics = _metric_summary(y_test, test_probabilities, threshold)
    comparison_by_model = {record["model"]: record for record in comparison}
    comparison_by_model[selected_name]["selected"] = True
    comparison_by_model[selected_name]["tuned_threshold"] = threshold
    comparison_by_model[selected_name]["test_f1_tuned"] = selected_test_metrics["f1"]
    comparison_by_model[selected_name]["test_recall_tuned"] = selected_test_metrics["recall"]

    importance = permutation_importance(
        selected_pipeline,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    importance_frame["importance_mean"] = importance_frame["importance_mean"].clip(lower=0)
    importance_frame = importance_frame.sort_values("importance_mean", ascending=False).reset_index(
        drop=True
    )
    total_importance = float(importance_frame["importance_mean"].sum())
    importance_frame["importance_share"] = (
        importance_frame["importance_mean"] / total_importance if total_importance else 0.0
    )

    return {
        "comparison": comparison,
        "fitted_models": fitted_models,
        "selected_name": selected_name,
        "selected_pipeline": selected_pipeline,
        "threshold": threshold,
        "test_probabilities": test_probabilities,
        "test_metrics": selected_test_metrics,
        "permutation_importance": importance_frame,
    }


def baselines_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute median/mode baselines used for local what-if explanations."""

    from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    baselines: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        baselines[column] = float(frame[column].median())
    for column in CATEGORICAL_FEATURES:
        baselines[column] = str(frame[column].mode(dropna=True).iloc[0])
    return baselines


def save_model_artifact(
    pipeline: Pipeline,
    *,
    threshold: float,
    selected_name: str,
    baselines: dict[str, Any],
    metrics: dict[str, Any],
    path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Fit the final model on all available data and serialize the bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "selected_model": selected_name,
        "selected_model_display": MODEL_DISPLAY_NAMES[selected_name],
        "baselines": baselines,
        "features": FEATURES,
        "risk_bands": RISK_BANDS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "metrics": metrics,
    }
    joblib.dump(bundle, path, compress=3)
    return bundle


def load_model_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m churn.train` first."
        )
    return joblib.load(path)


def write_metrics(
    *,
    selected_name: str,
    threshold: float,
    comparison: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    importance: pd.DataFrame,
    dataset_summary: dict[str, Any],
    segment_summary: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Write a JSON report consumed by the app, docs, and CI."""

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_summary,
        "selected_model": selected_name,
        "selected_model_display": MODEL_DISPLAY_NAMES[selected_name],
        "decision_threshold": round(float(threshold), 3),
        "risk_bands": RISK_BANDS,
        "model_comparison": comparison,
        "test_metrics": test_metrics,
        "segments": segment_summary,
        "feature_importance": importance.round(6).to_dict(orient="records"),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
