from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(
    not (ROOT / "reports" / "dryrun" / "predictions.parquet").exists(),
    reason="run `python -m costrouter evaluate --dry-run` first",
)
def test_dashboard_renders_without_errors(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("COSTROUTER_ROOT", str(ROOT))
    monkeypatch.setenv("COSTROUTER_PROFILE", "dryrun")
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) == 5
    at.slider[0].set_value(0.9).run()
    assert not at.exception
