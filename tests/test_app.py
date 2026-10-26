"""Render every page headlessly to catch app-only runtime regressions."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

PAGES = [
    "Overview",
    "Customer risk scorer",
    "Batch scoring",
    "Model performance",
    "Method & limits",
]


@pytest.fixture(scope="module")
def app() -> AppTest:
    instance = AppTest.from_file(APP_PATH, default_timeout=90).run()
    assert not instance.exception, instance.exception
    return instance


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(app: AppTest, page: str) -> None:
    app.radio[0].set_value(page).run()
    assert app.radio[0].value == page
    assert not app.exception, app.exception
