from __future__ import annotations

import pytest

from aidesk import db
from aidesk.config import reset_settings
from aidesk.llm.gateway import reset_gateway


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch, request):
    """Offline tests: fake model, fixture web, and a fresh data folder per test.

    Live and e2e tests keep the real settings and the shared var/ folder, so the response cache
    makes re-runs nearly free.
    """
    offline = not ({"live", "e2e"} & set(request.keywords))
    if offline:
        monkeypatch.setenv("AIDESK_DATA_DIR", str(tmp_path / "var"))
        monkeypatch.setenv("AIDESK_PROVIDER", "fake")
        monkeypatch.setenv("AIDESK_SEARCH", "fixture")
        monkeypatch.delenv("AIDESK_A2A_URL", raising=False)
        db.forget_initialised()
        reset_settings()
        reset_gateway()
    yield
    if offline:
        reset_settings()
        reset_gateway()
        db.forget_initialised()
