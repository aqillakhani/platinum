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


def _make_png_bytes(color: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    import io as _io

    from PIL import Image

    img = Image.new("RGB", (32, 32), color=color)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_score_happy_path_returns_float() -> None:
    png = _make_png_bytes()
    resp = client.post(
        "/score",
        files={"image": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] == 6.42  # _fake_scorer's sentinel
    assert isinstance(body["score"], float)
