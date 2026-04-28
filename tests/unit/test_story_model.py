"""Round-trip + new-field tests for the Scene dataclass.

S7 adds review_feedback (str | None), regen_count (int), and reject_count
(int) to Scene.
"""
from __future__ import annotations

from platinum.models.story import Scene


def _build_minimal_scene(**overrides) -> Scene:
    base = {
        "id": "scene_001",
        "index": 1,
        "narration_text": "x",
    }
    base.update(overrides)
    return Scene(**base)


def test_scene_review_feedback_defaults_to_none() -> None:
    scene = _build_minimal_scene()
    assert scene.review_feedback is None


def test_scene_regen_count_defaults_to_zero() -> None:
    scene = _build_minimal_scene()
    assert scene.regen_count == 0


def test_scene_round_trip_preserves_review_feedback() -> None:
    scene = _build_minimal_scene(review_feedback="needs more candle light")
    rt = Scene.from_dict(scene.to_dict())
    assert rt.review_feedback == "needs more candle light"


def test_scene_round_trip_preserves_regen_count() -> None:
    scene = _build_minimal_scene(regen_count=3)
    rt = Scene.from_dict(scene.to_dict())
    assert rt.regen_count == 3


def test_scene_from_dict_backfills_missing_review_feedback() -> None:
    """Old story.json files without these fields must still load."""
    raw = {"id": "s1", "index": 1, "narration_text": "x"}
    scene = Scene.from_dict(raw)
    assert scene.review_feedback is None
    assert scene.regen_count == 0
    assert scene.reject_count == 0


def test_scene_reject_count_defaults_to_zero() -> None:
    scene = _build_minimal_scene()
    assert scene.reject_count == 0


def test_scene_round_trip_preserves_reject_count() -> None:
    scene = _build_minimal_scene(reject_count=2)
    rt = Scene.from_dict(scene.to_dict())
    assert rt.reject_count == 2
