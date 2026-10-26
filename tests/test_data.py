"""Data contract and cleaning tests."""

from __future__ import annotations

import pandas as pd
import pytest

from churn.config import FEATURES, ID_COLUMN, TARGET
from churn.data import (
    DataValidationError,
    RAW_COLUMNS,
    batch_scoring_template,
    clean_churn_data,
    prepare_inference_data,
)


def test_source_shape_and_schema(raw_frame: pd.DataFrame) -> None:
    assert raw_frame.shape == (7043, 21)
    assert list(raw_frame.columns) == RAW_COLUMNS
    assert raw_frame[ID_COLUMN].is_unique
    assert set(raw_frame[TARGET].unique()) == {"No", "Yes"}


def test_clean_frame_contract(clean_frame: pd.DataFrame) -> None:
    assert len(clean_frame) == 7043
    assert clean_frame[ID_COLUMN].is_unique
    assert clean_frame[TARGET].isin([0, 1]).all()
    assert not clean_frame[FEATURES].isna().any().any()
    assert set(clean_frame["SeniorCitizen"].unique()) == {"No", "Yes"}
    assert clean_frame["AvgMonthlySpend"].notna().all()
    assert clean_frame["ServiceCount"].between(0, 8).all()
    assert clean_frame["IsNewCustomer"].isin([0, 1]).all()


def test_blank_total_charges_are_handled(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame) -> None:
    blank_mask = raw_frame["TotalCharges"].astype(str).str.strip() == ""
    assert int(blank_mask.sum()) == 11
    assert (raw_frame.loc[blank_mask, "tenure"] == 0).all()
    assert (clean_frame.loc[blank_mask, "TotalCharges"] == 0).all()


def test_invalid_target_is_rejected(raw_frame: pd.DataFrame) -> None:
    broken = raw_frame.copy()
    broken.loc[broken.index[0], TARGET] = "Maybe"
    with pytest.raises(DataValidationError, match="Churn must contain Yes/No"):
        clean_churn_data(broken)


def test_duplicate_customer_ids_are_rejected(raw_frame: pd.DataFrame) -> None:
    broken = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="customerID must be unique"):
        clean_churn_data(broken)


def test_inference_without_id_or_target(clean_frame: pd.DataFrame) -> None:
    raw_features = clean_frame.drop(columns=["customerID", TARGET])
    prepared = prepare_inference_data(raw_features)
    assert prepared[ID_COLUMN].tolist() == [f"row-{index}" for index in range(1, len(prepared) + 1)]
    assert TARGET not in prepared.columns
    assert not prepared[FEATURES].isna().any().any()


def test_batch_template_is_safe_for_users() -> None:
    template = batch_scoring_template()
    assert len(template) == 25
    assert TARGET not in template.columns
    assert ID_COLUMN in template.columns
    assert list(template.columns) == [column for column in RAW_COLUMNS if column != TARGET]
