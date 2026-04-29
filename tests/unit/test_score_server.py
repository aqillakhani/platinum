"""Tests for scripts/score_server/server.py.

Mocks the scorer dependencies via app.dependency_overrides so torch +
open_clip are never imported in local CI.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from score_server.server import app, get_clip_sim_scorer, get_scorer


def _fake_scorer(_image_bytes: bytes) -> float:
    return 6.42


def _fake_clip_sim_scorer(_image_bytes: bytes, text: str) -> float:
    # Deterministic stub: returns 0.42 for the canonical test text and
    # 0.10 for everything else. Lets tests assert on text routing.
    if text == "a candle in a dark hallway":
        return 0.42
    return 0.10


# Override the production scorers for all tests in this module.
app.dependency_overrides[get_scorer] = lambda: _fake_scorer
app.dependency_overrides[get_clip_sim_scorer] = lambda: _fake_clip_sim_scorer
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


def test_score_400_on_non_image_bytes() -> None:
    resp = client.post(
        "/score",
        files={"image": ("garbage.png", b"this is not a PNG", "image/png")},
    )
    assert resp.status_code == 400
    assert "could not decode" in resp.json()["detail"].lower()


def test_score_422_on_missing_image_field() -> None:
    # No "image" form field at all -> FastAPI/pydantic validation 422
    resp = client.post("/score", files={})
    assert resp.status_code == 422


def test_score_400_on_empty_image() -> None:
    resp = client.post(
        "/score",
        files={"image": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_clip_sim_happy_path_returns_similarity() -> None:
    """S7.1.A3.1: /clip-sim accepts image + text, returns float similarity."""
    png = _make_png_bytes()
    resp = client.post(
        "/clip-sim",
        files={"image": ("test.png", png, "image/png")},
        data={"text": "a candle in a dark hallway"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["similarity"] == 0.42  # _fake_clip_sim_scorer routed by text
    assert isinstance(body["similarity"], float)


def test_clip_sim_text_routes_correctly() -> None:
    """The text body field is forwarded to the scorer (not silently ignored)."""
    png = _make_png_bytes()
    resp = client.post(
        "/clip-sim",
        files={"image": ("test.png", png, "image/png")},
        data={"text": "an astronaut riding a horse"},
    )
    assert resp.status_code == 200
    assert resp.json()["similarity"] == 0.10  # not the routed text


def test_clip_sim_400_on_non_image_bytes() -> None:
    resp = client.post(
        "/clip-sim",
        files={"image": ("garbage.png", b"this is not a PNG", "image/png")},
        data={"text": "anything"},
    )
    assert resp.status_code == 400
    assert "could not decode" in resp.json()["detail"].lower()


def test_clip_sim_400_on_empty_image() -> None:
    resp = client.post(
        "/clip-sim",
        files={"image": ("empty.png", b"", "image/png")},
        data={"text": "anything"},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_clip_sim_422_on_missing_text() -> None:
    """text is a required form field."""
    png = _make_png_bytes()
    resp = client.post(
        "/clip-sim",
        files={"image": ("test.png", png, "image/png")},
        # no text
    )
    assert resp.status_code == 422
