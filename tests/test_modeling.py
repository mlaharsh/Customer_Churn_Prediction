"""Modeling helper and serialization tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.config import FEATURES
from churn.modeling import (
    _metric_summary,
    baselines_from_frame,
    candidate_pipelines,
    load_model_bundle,
    save_model_artifact,
)


def test_metric_summary_counts_confusion_matrix() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.7, 0.6])
    metrics = _metric_summary(y_true, probabilities, 0.5)
    assert metrics["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 3,
    }
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["prevalence"] == pytest.approx(0.6)


def test_candidate_models_have_expected_structure() -> None:
    candidates = candidate_pipelines()
    assert set(candidates) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
        "extra_trees",
    }
    for pipeline in candidates.values():
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


def test_baselines_cover_every_model_feature(clean_frame: pd.DataFrame) -> None:
    baselines = baselines_from_frame(clean_frame.head(200))
    assert set(baselines) == set(FEATURES)
    assert baselines["Contract"] in set(clean_frame["Contract"])
    assert 0 <= baselines["tenure"] <= 72
    assert baselines["MonthlyCharges"] > 0


def test_model_bundle_round_trip(tmp_path, bundle: dict) -> None:
    destination = tmp_path / "model.joblib"
    save_model_artifact(
        bundle["pipeline"],
        threshold=bundle["threshold"],
        selected_name=bundle["selected_model"],
        baselines=bundle["baselines"],
        metrics=bundle["metrics"],
        path=destination,
    )
    assert destination.exists()
    loaded = load_model_bundle(destination)
    assert loaded["threshold"] == bundle["threshold"]
    assert loaded["selected_model"] == bundle["selected_model"]
    assert loaded["features"] == FEATURES
    probabilities = loaded["pipeline"].predict_proba(
        pd.DataFrame([{feature: bundle["baselines"][feature] for feature in FEATURES}])
    )[:, 1]
    assert 0 <= float(probabilities[0]) <= 1
