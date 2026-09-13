"""Smoke tests for the Streamlit demo (the `ui` dependency group)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def _app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    return at


def test_app_loads_without_exceptions():
    _app()


def test_reconciliation_tab_shows_coherent_results():
    at = _app()
    # all three reconciliation methods must report coherent — that's the property
    # being demonstrated, not an incidental detail.
    assert sum("coherent" in m.value for m in at.markdown) == 3


def test_regular_series_backtest_runs():
    at = _app()
    button = next(b for b in at.button if b.label == "Run backtest")
    button.click().run(timeout=15)
    assert not at.exception


def test_intermittent_series_backtest_runs():
    at = _app()
    at.radio[0].set_value("Intermittent demand (Croston's case)").run(timeout=15)
    assert not at.exception
    button = next(b for b in at.button if b.label == "Run backtest")
    button.click().run(timeout=15)
    assert not at.exception
