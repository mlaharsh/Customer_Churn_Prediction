"""Scoring, risk banding, and business-facing retention insights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from churn.config import FEATURES, RISK_BANDS

FEATURE_LABELS = {
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "AvgMonthlySpend": "Average monthly spend",
    "ServiceCount": "Active services",
    "IsNewCustomer": "New customer",
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}

VALUE_LABELS = {
    "Month-to-month": "month-to-month",
    "One year": "one-year",
    "Two year": "two-year",
    "Electronic check": "electronic check",
    "Mailed check": "mailed check",
    "Bank transfer (automatic)": "bank transfer",
    "Credit card (automatic)": "credit card",
    "Fiber optic": "fiber optic",
}


def risk_tier(probability: float) -> str:
    """Map a churn probability to the configured business risk band."""

    for band in RISK_BANDS:
        if band["min"] <= probability < band["max"]:
            return str(band["name"])
    return "Critical"


def risk_color(probability: float) -> str:
    for band in RISK_BANDS:
        if band["min"] <= probability < band["max"]:
            return str(band["color"])
    return str(RISK_BANDS[-1]["color"])


def score_frame(
    pipeline: Any,
    frame: pd.DataFrame,
    *,
    threshold: float,
    include_target: bool = False,
) -> pd.DataFrame:
    """Score a cleaned feature frame and append decision-ready columns."""

    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot score data; missing features: {missing}")

    scored = frame.copy()
    raw_probabilities = pipeline.predict_proba(frame[FEATURES])[:, 1]
    probabilities = np.round(raw_probabilities, 4)
    scored["churn_probability"] = probabilities
    scored["churn_prediction"] = (raw_probabilities >= threshold).astype(int)
    scored["risk_tier"] = [risk_tier(value) for value in raw_probabilities]
    scored["expected_monthly_revenue_at_risk"] = np.round(
        scored["MonthlyCharges"].to_numpy() * probabilities, 2
    )
    scored["expected_annual_revenue_at_risk"] = np.round(
        scored["expected_monthly_revenue_at_risk"].to_numpy() * 12, 2
    )
    if include_target:
        scored["actual_churn"] = frame.get("Churn")
    return scored


def _baseline_row(row: pd.Series, baselines: dict[str, Any]) -> pd.Series:
    baseline = row.copy()
    for feature, value in baselines.items():
        if feature in baseline.index:
            baseline[feature] = value
    return baseline


def local_what_if(
    pipeline: Any,
    row: pd.Series,
    baselines: dict[str, Any],
    *,
    top_n: int = 6,
) -> pd.DataFrame:
    """Estimate one-at-a-time feature sensitivity for a single customer.

    This is a transparent what-if calculation, not a causal attribution. Each
    feature is reset to its training-set baseline (median/mode) while the other
    inputs stay fixed. The delta is the change in predicted churn probability.
    """

    actual = float(pipeline.predict_proba(pd.DataFrame([row[FEATURES]]))[0, 1])
    records: list[dict[str, Any]] = []
    for feature in FEATURES:
        candidate = _baseline_row(row, baselines)
        candidate[feature] = baselines[feature]
        probability = float(
            pipeline.predict_proba(pd.DataFrame([candidate[FEATURES]]))[0, 1]
        )
        records.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "current_value": _display_value(feature, row[feature]),
                "baseline_value": _display_value(feature, baselines[feature]),
                "probability_if_baseline": probability,
                "impact_pp": round((actual - probability) * 100, 1),
            }
        )

    result = pd.DataFrame(records)
    result["absolute_impact"] = result["impact_pp"].abs()
    return result.sort_values("absolute_impact", ascending=False).head(top_n).drop(
        columns=["absolute_impact"]
    )


def _display_value(feature: str, value: Any) -> str:
    if feature == "IsNewCustomer":
        return "Yes" if int(value) == 1 else "No"
    if feature in {"tenure"}:
        return f"{int(value)} months"
    if feature in {"MonthlyCharges", "TotalCharges", "AvgMonthlySpend"}:
        return f"${float(value):,.2f}"
    if feature == "ServiceCount":
        return f"{int(value)} active"
    text = str(value)
    return VALUE_LABELS.get(text, text)


def retention_actions(row: pd.Series, probability: float, threshold: float) -> list[str]:
    """Return explainable, rule-based retention plays for a scored customer.

    These are operational suggestions, not causal claims. They are intentionally
    tied to observable account facts so a retention agent can act on them.
    """

    actions: list[str] = []
    if row.get("Contract") == "Month-to-month":
        actions.append(
            "Offer a one-year contract with a loyalty credit; contract length is the strongest segment-level churn signal."
        )
    if int(row.get("IsNewCustomer", 0)) == 1:
        actions.append(
            "Trigger a 30-day onboarding check-in and confirm the customer has activated their core services."
        )
    if row.get("TechSupport") == "No" or row.get("OnlineSecurity") == "No":
        actions.append(
            "Bundle a no-cost tech-support or online-security trial; missing support is associated with higher churn."
        )
    if row.get("InternetService") == "Fiber optic":
        actions.append(
            "Run a fiber service-quality and price-to-value review; fiber customers show elevated churn in this dataset."
        )
    if row.get("PaymentMethod") == "Electronic check":
        actions.append(
            "Incentivize automatic bank or card payment to reduce friction and improve renewal stickiness."
        )
    if float(row.get("MonthlyCharges", 0)) >= 80:
        actions.append(
            "Review the plan against usage and offer a right-sized bundle rather than a blanket discount."
        )
    if probability >= threshold:
        actions.insert(
            0,
            "Assign a retention specialist within 48 hours and document the save offer outcome."
        )
    if not actions:
        actions.append(
            "Maintain standard engagement; monitor the next billing cycle for changes in tenure or service mix."
        )
    return actions[:4]


def segment_summary(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Summarize churn rate, customer count, and observed revenue loss by segment."""

    grouped = (
        frame.groupby(dimension, dropna=False, observed=True)
        .agg(
            customers=("customerID", "count"),
            churners=("Churn", "sum"),
            monthly_revenue=("MonthlyCharges", "sum"),
        )
        .reset_index()
    )
    # Revenue loss uses observed churn in the segment, not a model estimate.
    churned_revenue = (
        frame.loc[frame["Churn"] == 1]
        .groupby(dimension, dropna=False, observed=True)["MonthlyCharges"]
        .sum()
        .rename("monthly_churn_revenue")
        .reset_index()
    )
    grouped = grouped.merge(churned_revenue, on=dimension, how="left")
    grouped["monthly_churn_revenue"] = grouped["monthly_churn_revenue"].fillna(0.0)
    grouped["annual_churn_revenue"] = grouped["monthly_churn_revenue"] * 12
    grouped["churn_rate"] = grouped["churners"] / grouped["customers"]
    grouped = grouped.rename(columns={dimension: "segment"})
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)
