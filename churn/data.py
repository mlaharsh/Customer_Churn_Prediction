"""Data loading, validation, and feature preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    SERVICE_COLUMNS,
    TARGET,
)

RAW_COLUMNS = [
    ID_COLUMN,
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET,
]


class DataValidationError(ValueError):
    """Raised when the input data does not match the expected contract."""


def read_raw_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the Telco churn CSV without silently coercing values.

    ``TotalCharges`` contains 11 blank strings in the source file. Reading it
    without type coercion lets us validate and handle those records explicitly.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place the Telco Customer Churn CSV at data/raw/telco_churn.csv."
        )
    frame = pd.read_csv(path)
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    return frame


def _yes_no_from_binary(series: pd.Series) -> pd.Series:
    """Normalize the SeniorCitizen flag to Yes/No regardless of input dtype."""

    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map({1: "Yes", 0: "No"})
    text = series.astype(str).str.strip().str.title()
    return mapped.fillna(text.where(text.isin(["Yes", "No"]), "No"))


def clean_churn_data(
    raw: pd.DataFrame,
    *,
    require_target: bool = True,
) -> pd.DataFrame:
    """Apply the deterministic cleaning and feature engineering steps.

    Business rules
    --------------
    - ``TotalCharges`` is blank only for brand-new customers with ``tenure=0``.
      Those rows receive ``0``. For future batch data, a blank is imputed as
      ``MonthlyCharges * tenure`` when tenure is available.
    - ``AvgMonthlySpend`` is total charges divided by tenure; new customers use
      their current monthly charge.
    - ``ServiceCount`` counts active add-on services (a value of ``Yes``).
    - ``IsNewCustomer`` flags the first six months, a high-risk lifecycle stage.
    """

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataValidationError("Input data must be a non-empty pandas DataFrame.")

    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if require_target:
        required = RAW_COLUMNS
    else:
        required = [column for column in RAW_COLUMNS if column != TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str).str.strip()
    if frame[ID_COLUMN].duplicated().any():
        duplicates = int(frame[ID_COLUMN].duplicated().sum())
        raise DataValidationError(f"customerID must be unique; found {duplicates} duplicates.")

    frame["SeniorCitizen"] = _yes_no_from_binary(frame["SeniorCitizen"])

    for column in ["tenure", "MonthlyCharges"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[["tenure", "MonthlyCharges"]].isna().any().any():
        raise DataValidationError("tenure and MonthlyCharges must be numeric.")

    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    expected_total = frame["MonthlyCharges"] * frame["tenure"].clip(lower=0)
    frame["TotalCharges"] = frame["TotalCharges"].fillna(expected_total).fillna(0.0)

    frame["AvgMonthlySpend"] = np.where(
        frame["tenure"] > 0,
        frame["TotalCharges"] / frame["tenure"].replace(0, np.nan),
        frame["MonthlyCharges"],
    )
    frame["AvgMonthlySpend"] = pd.to_numeric(frame["AvgMonthlySpend"], errors="coerce").fillna(
        frame["MonthlyCharges"]
    )

    frame["ServiceCount"] = (frame[SERVICE_COLUMNS] == "Yes").sum(axis=1).astype(int)
    frame["IsNewCustomer"] = (frame["tenure"] <= 6).astype(int)

    if require_target:
        normalized_target = frame[TARGET].astype(str).str.strip().str.title()
        mapped_target = normalized_target.map({"Yes": 1, "No": 0})
        if mapped_target.isna().any():
            values = sorted(normalized_target[mapped_target.isna()].unique().tolist())
            raise DataValidationError(f"Churn must contain Yes/No values; found {values}")
        frame[TARGET] = mapped_target.astype(int)

    if frame[FEATURES].isna().any().any():
        missing_features = frame[FEATURES].columns[frame[FEATURES].isna().any()].tolist()
        raise DataValidationError(f"Missing values remain in features: {missing_features}")

    return frame


def load_clean_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Convenience loader for the canonical, cleaned training dataset."""

    return clean_churn_data(read_raw_data(path=path), require_target=True)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model feature matrix and the binary target."""

    if TARGET not in frame.columns:
        raise DataValidationError("The target column is required for model training.")
    return frame[FEATURES].copy(), frame[TARGET].astype(int).copy()


def prepare_inference_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean feature data for scoring while allowing Churn/customerID to be absent."""

    frame = raw.copy()
    if ID_COLUMN not in frame.columns:
        frame.insert(0, ID_COLUMN, [f"row-{i + 1}" for i in range(len(frame))])
    if TARGET in frame.columns:
        frame = frame.drop(columns=[TARGET])
    cleaned = clean_churn_data(frame.assign(Churn="No"), require_target=False)
    return cleaned.drop(columns=[TARGET], errors="ignore")


def batch_scoring_template(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a user-facing batch scoring template (features, no target)."""

    if raw is None:
        raw = read_raw_data()
    columns = [column for column in RAW_COLUMNS if column != TARGET]
    return raw[columns].head(25).copy()
