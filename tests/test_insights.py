"""Risk scoring and retention-insight tests."""

from __future__ import annotations

import pandas as pd
import pytest

from churn.config import FEATURES, RISK_BANDS
from churn.insights import (
    local_what_if,
    retention_actions,
    risk_color,
    risk_tier,
    score_frame,
    segment_summary,
)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "Low"),
        (0.2999, "Low"),
        (0.30, "Medium"),
        (0.5499, "Medium"),
        (0.55, "High"),
        (0.7499, "High"),
        (0.75, "Critical"),
        (1.0, "Critical"),
    ],
)
def test_risk_tiers_follow_configured_bands(probability: float, expected: str) -> None:
    assert risk_tier(probability) == expected
    assert risk_color(probability) in {band["color"] for band in RISK_BANDS}


def test_score_frame_adds_decision_columns(clean_frame: pd.DataFrame, bundle: dict) -> None:
    frame = clean_frame.head(25).copy()
    scored = score_frame(bundle["pipeline"], frame, threshold=bundle["threshold"], include_target=True)
    assert {
        "churn_probability",
        "churn_prediction",
        "risk_tier",
        "expected_monthly_revenue_at_risk",
        "expected_annual_revenue_at_risk",
        "actual_churn",
    }.issubset(scored.columns)
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["churn_prediction"].isin([0, 1]).all()
    assert set(scored["risk_tier"]).issubset({"Low", "Medium", "High", "Critical"})
    expected_monthly = (scored["MonthlyCharges"] * scored["churn_probability"]).round(2)
    expected_annual = (expected_monthly * 12).round(2)
    pd.testing.assert_series_equal(
        scored["expected_monthly_revenue_at_risk"].reset_index(drop=True),
        expected_monthly.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        scored["expected_annual_revenue_at_risk"].reset_index(drop=True),
        expected_annual.reset_index(drop=True),
        check_names=False,
    )


def test_local_what_if_returns_ranked_sensitivity(clean_frame: pd.DataFrame, bundle: dict) -> None:
    row = clean_frame.iloc[0]
    result = local_what_if(bundle["pipeline"], row, bundle["baselines"], top_n=6)
    assert len(result) == 6
    assert {
        "feature",
        "feature_label",
        "current_value",
        "baseline_value",
        "probability_if_baseline",
        "impact_pp",
    }.issubset(result.columns)
    assert result["feature"].isin(FEATURES).all()
    assert result["probability_if_baseline"].between(0, 1).all()
    assert result["impact_pp"].map(lambda value: isinstance(value, (int, float))).all()


def test_retention_actions_prioritize_flagged_customer(clean_frame: pd.DataFrame) -> None:
    row = clean_frame[
        (clean_frame["Contract"] == "Month-to-month")
        & (clean_frame["InternetService"] == "Fiber optic")
        & (clean_frame["TechSupport"] == "No")
    ].iloc[0]
    actions = retention_actions(row, probability=0.9, threshold=0.583)
    assert actions
    assert "specialist" in actions[0].lower()
    assert len(actions) <= 4


def test_segment_summary_is_internally_consistent(clean_frame: pd.DataFrame) -> None:
    frame = clean_frame.head(1200).copy()
    summary = segment_summary(frame, "Contract")
    assert {"segment", "customers", "churners", "churn_rate", "monthly_churn_revenue"}.issubset(
        summary.columns
    )
    assert summary["customers"].sum() == len(frame)
    assert summary["churners"].sum() == int(frame["Churn"].sum())
    assert summary["churn_rate"].between(0, 1).all()
    assert (summary["monthly_churn_revenue"] >= 0).all()
