"""Shared pytest fixtures for churn tests."""

from __future__ import annotations

import pandas as pd
import pytest

from churn.data import clean_churn_data, read_raw_data
from churn.modeling import load_model_bundle


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return read_raw_data()


@pytest.fixture(scope="session")
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return clean_churn_data(raw_frame)


@pytest.fixture(scope="session")
def bundle() -> dict:
    return load_model_bundle()
