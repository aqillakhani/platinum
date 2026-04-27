"""Tests for scripts/reset_scene_keyframes.py."""
from __future__ import annotations

import json


def test_reset_clears_keyframe_state_for_specified_scenes(tmp_path):
    """Targeted scenes get keyframe_path=None + cleared candidates/scores."""
    from reset_scene_keyframes import reset_scenes

    story_id = "test_story"
    story_dir = tmp_path / story_id
    story_dir.mkdir()
    story_json = {
        "id": story_id,
        "scenes": [
            {"index": 0, "keyframe_path": "/foo/0.png",
             "keyframe_candidates": ["/foo/c0.png", "/foo/c1.png"],
             "keyframe_scores": [6.5, 7.0],
             "validation": {"keyframe_anatomy": [True, True],
                            "keyframe_selected_via_fallback": False}},
            {"index": 1, "keyframe_path": "/foo/1.png",
             "keyframe_candidates": ["/foo/c2.png"], "keyframe_scores": [8.0],
             "validation": {"keyframe_anatomy": [True]}},
        ],
    }
    (story_dir / "story.json").write_text(json.dumps(story_json))

    rc = reset_scenes(story_id=story_id, scenes=[0],
                      story_dir=tmp_path, delete_files=False)
    assert rc == 0

    out = json.loads((story_dir / "story.json").read_text())
    assert out["scenes"][0]["keyframe_path"] is None
    assert out["scenes"][0]["keyframe_candidates"] == []
    assert out["scenes"][0]["keyframe_scores"] == []
    assert "keyframe_anatomy" not in out["scenes"][0]["validation"]
    assert "keyframe_selected_via_fallback" not in out["scenes"][0]["validation"]
    # Scene 1 untouched
    assert out["scenes"][1]["keyframe_path"] == "/foo/1.png"


def test_reset_with_missing_story_returns_nonzero(tmp_path):
    from reset_scene_keyframes import reset_scenes
    rc = reset_scenes(story_id="nonexistent", scenes=[0],
                      story_dir=tmp_path, delete_files=False)
    assert rc == 2


def test_reset_with_delete_files_removes_pngs(tmp_path):
    """--delete-files clears candidate_*.png from scene subdir."""
    from reset_scene_keyframes import reset_scenes

    story_id = "test_story"
    story_dir = tmp_path / story_id
    story_dir.mkdir()
    keyframes_dir = story_dir / "keyframes" / "scene_001"
    keyframes_dir.mkdir(parents=True)
    (keyframes_dir / "candidate_0.png").write_bytes(b"fake")
    (keyframes_dir / "candidate_1.png").write_bytes(b"fake")
    (keyframes_dir / "metadata.json").write_text("{}")        # not a candidate

    story_json = {
        "id": story_id,
        "scenes": [
            {"index": 1, "keyframe_path": "/x/c.png",
             "keyframe_candidates": [], "keyframe_scores": [],
             "validation": {}},
        ],
    }
    (story_dir / "story.json").write_text(json.dumps(story_json))

    rc = reset_scenes(story_id=story_id, scenes=[1],
                      story_dir=tmp_path, delete_files=True)
    assert rc == 0
    assert not (keyframes_dir / "candidate_0.png").exists()
    assert not (keyframes_dir / "candidate_1.png").exists()
    assert (keyframes_dir / "metadata.json").exists()       # NOT touched
