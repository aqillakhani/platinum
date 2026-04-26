"""Tests for scripts/score_server/server.py.

Mocks the scorer dependency via app.dependency_overrides so torch + open_clip
are never imported in local CI.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from score_server.server import app, get_scorer


def _fake_scorer(_image_bytes: bytes) -> float:
    return 6.42


# Override the production scorer for the duration of all tests in this module.
# The override is applied at import time; pytest's test isolation handles cleanup
# implicitly because we're not modifying any persistent state outside this module.
app.dependency_overrides[get_scorer] = lambda: _fake_scorer
client = TestClient(app)


def test_health_returns_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "model" in body
