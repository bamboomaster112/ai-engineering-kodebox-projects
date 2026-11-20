from __future__ import annotations

import os

import pytest

from aidesk import db
from aidesk.config import reset_settings
from aidesk.llm.gateway import reset_gateway

LIVE = os.environ.get("AIDESK_LIVE") == "1"


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch, request):
    """Every test gets its own data folder. Offline tests use the fake model; live tests keep KodeKey."""
    monkeypatch.setenv("AIDESK_DATA_DIR", str(tmp_path / "var"))
    if "live" not in request.keywords:
        monkeypatch.setenv("AIDESK_PROVIDER", "fake")
        monkeypatch.setenv("AIDESK_SEARCH", "fixture")
    db.forget_initialised()
    reset_settings()
    reset_gateway()
    yield
    reset_settings()
    reset_gateway()
    db.forget_initialised()
