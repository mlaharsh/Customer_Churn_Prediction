"""Export the fitted churn pipeline for the static browser scorer.

The GitHub Pages deployment has no Python runtime. This module copies the
*exact* fitted preprocessing parameters and logistic-regression coefficients
into a small JavaScript data file so the public scorer can reproduce the native
model without retraining or calling a server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn.config import FEATURES, ROOT, TEST_PREDICTIONS_PATH
from churn.data import prepare_inference_data
from churn.insights import FEATURE_LABELS
from churn.modeling import load_model_bundle

BROWSER_DATA_PATH = ROOT / "docs" / "scorer_data.js"
BROWSER_DATA_PREFIX = "window.CHURN_MODEL = "
BROWSER_DATA_SUFFIX = ";\n"

# Raw customer profiles used by the UI as demos and by the parity checks. They
# deliberately cover very different risk profiles.
EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "new-fiber",
        "label": "New month-to-month fiber customer",
        "raw": {
            "gender": "Female",
            "SeniorCitizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 98.5,
            "TotalCharges": 295.5,
        },
    },
    {
        "id": "loyal",
        "label": "Long-tenure two-year customer",
        "raw": {
            "gender": "Male",
            "SeniorCitizen": "No",
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 66,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 35.0,
            "TotalCharges": 2310.0,
        },
    },
    {
        "id": "mid-term",
        "label": "One-year DSL customer",
        "raw": {
            "gender": "Female",
            "SeniorCitizen": "Yes",
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 28,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 59.9,
            "TotalCharges": 1677.2,
        },
    },
]


def _validate_bundle(bundle: dict[str, Any]) -> None:
    required = {"pipeline", "threshold", "baselines", "risk_bands", "features"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {missing}")
    pipeline = bundle["pipeline"]
    try:
        pipeline.named_steps["preprocessor"]
        pipeline.named_steps["classifier"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("Model bundle does not contain the expected pipeline steps.") from exc


def _cleaned_example_frame(pipeline: Any, examples: list[dict[str, Any]]) -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {"customerID": f"demo-{index + 1}", **example["raw"]}
            for index, example in enumerate(examples)
        ]
    )
    return prepare_inference_data(raw)


def build_browser_artifact(
    bundle: dict[str, Any] | None = None,
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable payload embedded in ``scorer_data.js``."""

    bundle = load_model_bundle() if bundle is None else bundle
    _validate_bundle(bundle)
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    coefficients = np.asarray(classifier.coef_, dtype=float).reshape(-1)
    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Encoded feature count does not match the classifier coefficient count."
        )
    coefficient_map = dict(zip(feature_names, coefficients, strict=True))

    categorical_transformer = preprocessor.named_transformers_["categorical"]
    numeric_transformer = preprocessor.named_transformers_["numeric"]
    imputer = numeric_transformer.named_steps["imputer"]
    scaler = numeric_transformer.named_steps["scaler"]

    categorical: list[dict[str, Any]] = []
    for feature, categories in zip(
        categorical_transformer.feature_names_in_,
        categorical_transformer.categories_,
        strict=True,
    ):
        feature = str(feature)
        categorical.append(
            {
                "name": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "coefficients": {
                    str(category): float(coefficient_map[f"{feature}_{category}"])
                    for category in categories
                },
            }
        )

    numeric: list[dict[str, Any]] = []
    for name, median, mean, scale in zip(
        numeric_transformer.feature_names_in_,
        imputer.statistics_,
        scaler.mean_,
        scaler.scale_,
        strict=True,
    ):
        name = str(name)
        numeric.append(
            {
                "name": name,
                "label": FEATURE_LABELS.get(name, name),
                "median": float(median),
                "mean": float(mean),
                "scale": float(scale),
                "coefficient": float(coefficient_map[name]),
            }
        )

    if predictions is None:
        predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    required_prediction_columns = {
        "churn_probability",
        "actual_churn",
        "MonthlyCharges",
    }
    missing_predictions = sorted(required_prediction_columns - set(predictions.columns))
    if missing_predictions:
        raise ValueError(f"Test predictions are missing columns: {missing_predictions}")
    if len(predictions) == 0:
        raise ValueError("Test predictions are empty.")

    cleaned_examples = _cleaned_example_frame(pipeline, EXAMPLES)
    examples: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for index, example in enumerate(EXAMPLES):
        features = {
            feature: cleaned_examples.iloc[index][feature]
            for feature in FEATURES
        }
        cleaned_features = {
            feature: (
                float(features[feature])
                if feature in numeric_transformer.feature_names_in_
                else str(features[feature])
            )
            for feature in FEATURES
        }
        expected_probability = float(
            pipeline.predict_proba(pd.DataFrame([cleaned_features], columns=FEATURES))[0, 1]
        )
        examples.append(
            {
                "id": example["id"],
                "label": example["label"],
                "raw": example["raw"],
            }
        )
        validation.append(
            {
                "id": example["id"],
                "label": example["label"],
                "features": cleaned_features,
                "expectedProbability": expected_probability,
            }
        )

    # The holdout rows are the untouched 20% evaluation split. Their probabilities
    # come from the selected pipeline fitted on the training split, so this is a
    # genuine out-of-sample threshold replay rather than an in-sample fit.
    holdout = {
        "n": int(len(predictions)),
        "churners": int(predictions["actual_churn"].sum()),
        "threshold": float(bundle["threshold"]),
        "probabilities": [
            round(float(value), 4) for value in predictions["churn_probability"].tolist()
        ],
        "labels": "".join(
            str(int(value)) for value in predictions["actual_churn"].tolist()
        ),
        "monthlyCharges": [
            round(float(value), 2) for value in predictions["MonthlyCharges"].tolist()
        ],
    }

    metrics = bundle.get("metrics", {})
    test_metrics = metrics.get("test_metrics", {})
    dataset = metrics.get("dataset", {})
    metadata = {
        "title": "Churn Signal live scorer",
        "selectedModel": str(bundle.get("selected_model_display", "Logistic regression")),
        "selectedModelKey": str(bundle.get("selected_model", "logistic_regression")),
        "threshold": float(bundle["threshold"]),
        "trainedAt": str(bundle.get("trained_at", "")),
        "sklearnVersion": str(bundle.get("sklearn_version", "")),
        "pythonVersion": str(bundle.get("python_version", "")),
        "riskBands": [
            {
                "name": str(band["name"]),
                "min": float(band["min"]),
                "max": float(band["max"]),
                "color": str(band["color"]),
            }
            for band in bundle["risk_bands"]
        ],
        "metrics": {
            "rocAuc": float(test_metrics.get("roc_auc", 0.0)),
            "prAuc": float(test_metrics.get("pr_auc", 0.0)),
            "precision": float(test_metrics.get("precision", 0.0)),
            "recall": float(test_metrics.get("recall", 0.0)),
            "f1": float(test_metrics.get("f1", 0.0)),
            "brier": float(test_metrics.get("brier", 0.0)),
            "alertRate": float(test_metrics.get("alert_rate", 0.0)),
            "churnersFlagged": int(test_metrics.get("churners_flagged", 0)),
            "nonChurnersContacted": int(test_metrics.get("non_churners_contacted", 0)),
            "churnersMissed": int(test_metrics.get("churners_missed", 0)),
        },
        "dataset": {
            "customers": int(dataset.get("customers", 0)),
            "churners": int(dataset.get("churners", 0)),
            "churnRate": float(dataset.get("churn_rate", 0.0)),
            "annualChurnRevenue": float(dataset.get("annual_churn_revenue", 0.0)),
        },
        "source": "models/churn_model.joblib",
        "note": (
            "Coefficients and preprocessing parameters are exported from the fitted "
            "scikit-learn pipeline. Holdout threshold metrics replay the saved "
            "out-of-sample evaluation predictions."
        ),
    }

    return {
        "metadata": metadata,
        "numeric": numeric,
        "categorical": categorical,
        "intercept": float(np.asarray(classifier.intercept_).reshape(-1)[0]),
        "baselines": {
            key: float(value) if key in numeric_transformer.feature_names_in_ else str(value)
            for key, value in bundle["baselines"].items()
            if key in FEATURES
        },
        "examples": examples,
        "validation": validation,
        "holdout": holdout,
    }


def render_browser_data(artifact: dict[str, Any]) -> str:
    """Serialize the artifact as a classic script so it also works from file://."""

    payload = json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
    return f"{BROWSER_DATA_PREFIX}{payload}{BROWSER_DATA_SUFFIX}"


def write_browser_data(
    path: Path = BROWSER_DATA_PATH,
    bundle: dict[str, Any] | None = None,
) -> Path:
    """Write the generated browser data file and return its path."""

    artifact = build_browser_artifact(bundle=bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_browser_data(artifact), encoding="utf-8", newline="\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if docs/scorer_data.js is out of date instead of rewriting it.",
    )
    args = parser.parse_args()

    if args.check:
        expected = render_browser_data(build_browser_artifact())
        current = BROWSER_DATA_PATH.read_text(encoding="utf-8")
        if current != expected:
            raise SystemExit(
                "docs/scorer_data.js is out of date. "
                "Run `python -m churn.browser_export` and commit the result."
            )
        print(f"Browser artifact is current: {BROWSER_DATA_PATH}")
        return

    path = write_browser_data()
    print(f"Wrote browser scorer artifact: {path}")


if __name__ == "__main__":
    main()
