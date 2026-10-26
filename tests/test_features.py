"""Preprocessing contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import FEATURES
from churn.features import build_preprocessor


def test_preprocessor_ignores_unseen_categories(clean_frame: pd.DataFrame) -> None:
    frame = clean_frame[FEATURES].head(40).copy()
    preprocessor = build_preprocessor()
    preprocessor.fit(frame)
    transformed = preprocessor.transform(frame)
    assert transformed.shape[0] == len(frame)
    assert np.isfinite(transformed).all()

    unseen = frame.copy()
    unseen.loc[unseen.index[0], "Contract"] = "Lifetime"
    transformed_unseen = preprocessor.transform(unseen)
    assert transformed_unseen.shape == transformed.shape
    assert np.isfinite(transformed_unseen).all()
