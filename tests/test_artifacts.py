"""Smoke tests for the committed model and evaluation artifacts."""

from __future__ import annotations

import json

import pandas as pd

from churn.config import (
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    TEST_PREDICTIONS_PATH,
)
from churn.modeling import load_model_bundle


def test_model_and_metrics_artifacts_exist() -> None:
    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()
    assert FEATURE_IMPORTANCE_PATH.exists()
    assert TEST_PREDICTIONS_PATH.exists()
    bundle = load_model_bundle()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert bundle["selected_model"] == metrics["selected_model"]
    assert bundle["threshold"] == metrics["decision_threshold"]
    assert 0 < bundle["threshold"] < 1
    assert len(bundle["features"]) == 22
    priority = metrics["dataset"]["priority_segment"]
    assert priority["customers"] == 2128
    assert priority["churners"] == 1162
    assert 0 < priority["share_of_churn_revenue"] < 1


def test_holdout_predictions_have_expected_shape() -> None:
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    assert len(predictions) == 1409
    assert predictions["customerID"].is_unique
    assert predictions["churn_probability"].between(0, 1).all()
    assert predictions["actual_churn"].isin([0, 1]).all()
    assert predictions["risk_tier"].isin(["Low", "Medium", "High", "Critical"]).all()


def test_risk_deciles_and_report_figures_exist() -> None:
    risk_deciles = pd.read_csv(TEST_PREDICTIONS_PATH.parent / "risk_deciles.csv")
    assert len(risk_deciles) == 10
    assert risk_deciles["churn_capture"].iloc[0] <= 1
    assert risk_deciles["churn_capture"].iloc[-1] == 1
    expected_figures = {
        "churn_overview.png",
        "segment_risk_profile.png",
        "model_comparison.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "confusion_matrix.png",
        "threshold_tradeoff.png",
        "feature_importance.png",
        "risk_distribution.png",
        "calibration_curve.png",
        "revenue_at_risk.png",
    }
    assert expected_figures.issubset({path.name for path in FIGURES_DIR.glob("*.png")})
    assert (TEST_PREDICTIONS_PATH.parent / "summary.md").exists()
    assert (TEST_PREDICTIONS_PATH.parent / "model_comparison.csv").exists()
