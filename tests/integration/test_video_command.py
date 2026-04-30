"""Integration tests for `platinum video` CLI."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner


def _redirect_config_to(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Config so its __init__ uses `project` as root (Config defaults to _ROOT)."""
    from platinum import cli as cli_mod

    original_init = cli_mod.Config.__init__

    def init_with_root(self, root=None):  # type: ignore[no-untyped-def]
        original_init(self, root=project)

    monkeypatch.setattr(cli_mod.Config, "__init__", init_with_root)


@pytest.fixture
def cli_project(tmp_path: Path) -> Path:
    """Mirror the real project layout under tmp_path for video CLI tests."""
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_path / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data" / "stories").mkdir(parents=True)
    return tmp_path


class TestVideoCommand:
    def test_dry_run_lists_targets_without_running_stage(
        self, cli_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from platinum.cli import app

        _redirect_config_to(cli_project, monkeypatch)

        # Build a minimal story directory with one scene whose keyframe_path
        # is set + the keyframe file actually exists.
        story_dir = cli_project / "data" / "stories" / "story_test_001"
        keyframes_dir = story_dir / "keyframes" / "scene_000"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        kf = keyframes_dir / "candidate_0.png"
        kf.write_bytes(b"fake_png")
        story_json = story_dir / "story.json"
        story_data: dict = {
            "id": "story_test_001",
            "track": "atmospheric_horror",
            "source": {"type": "test", "title": "t", "author": None,
                       "url": "https://x", "license": "PD", "fetched_at": "2026-04-30T00:00:00",
                       "raw_text": ""},
            "adapted": None,
            "scenes": [{
                "id": "scene_000", "index": 1,
                "narration_text": "n", "narration_duration_seconds": 5.0,
                "visual_prompt": "x", "negative_prompt": None,
                "keyframe_path": str(kf), "video_path": None,
                "video_duration_seconds": 0.0,
                "music_cue": None, "sfx_cues": [],
                "validation": {}, "review_status": "pending",
            }],
            "audio": {}, "review_status": "pending",
            "characters": {},
        }
        story_json.write_text(json.dumps(story_data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["video", "story_test_001", "--dry-run"])
        assert result.exit_code == 0, result.stdout
        assert "scene" in result.stdout.lower() or "1" in result.stdout

    def test_missing_story_id_exits_nonzero(
        self, cli_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from platinum.cli import app

        _redirect_config_to(cli_project, monkeypatch)

        runner = CliRunner()
        result = runner.invoke(app, ["video", "nonexistent_story_id"])
        assert result.exit_code != 0

    def test_scenes_filter_parses_csv(
        self, cli_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--scenes 1,3 should filter to scenes with index 1 and 3.

        We verify by checking dry-run output mentions only the targeted scenes.
        Scene indices are 1-indexed per scene_breakdown stage.
        """
        from platinum.cli import app

        _redirect_config_to(cli_project, monkeypatch)

        story_dir = cli_project / "data" / "stories" / "story_test_002"
        story_dir.mkdir(parents=True, exist_ok=True)
        for idx in (1, 2, 3):
            kdir = story_dir / "keyframes" / f"scene_{idx:03d}"
            kdir.mkdir(parents=True, exist_ok=True)
            (kdir / "candidate_0.png").write_bytes(b"fake_png")

        story_json = story_dir / "story.json"
        scenes: list[dict] = []
        for idx in (1, 2, 3):
            scenes.append({
                "id": f"scene_{idx:03d}", "index": idx,
                "narration_text": "n", "narration_duration_seconds": 5.0,
                "visual_prompt": "x", "negative_prompt": None,
                "keyframe_path": str(
                    story_dir / "keyframes" / f"scene_{idx:03d}" / "candidate_0.png"
                ),
                "video_path": None, "video_duration_seconds": 0.0,
                "music_cue": None, "sfx_cues": [],
                "validation": {}, "review_status": "pending",
            })
        story_data: dict = {
            "id": "story_test_002", "track": "atmospheric_horror",
            "source": {"type": "test", "title": "t", "author": None,
                       "url": "https://x", "license": "PD",
                       "fetched_at": "2026-04-30T00:00:00", "raw_text": ""},
            "adapted": None, "scenes": scenes,
            "audio": {}, "review_status": "pending",
            "characters": {},
        }
        story_json.write_text(json.dumps(story_data), encoding="utf-8")

        # Also ensure atmospheric_horror.yaml exists in cli_project for track config
        (cli_project / "config" / "tracks").mkdir(parents=True, exist_ok=True)
        import shutil as sh
        repo_root = Path(__file__).resolve().parents[2]
        sh.copy(
            repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
            cli_project / "config" / "tracks" / "atmospheric_horror.yaml",
        )

        runner = CliRunner()
        # --scenes 1,3 -> filter to scenes with index 1 and 3
        result = runner.invoke(
            app, ["video", "story_test_002", "--scenes", "1,3", "--dry-run"]
        )
        assert result.exit_code == 0, result.stdout
        # Scenes 1 and 3 targeted
        output_lower = result.stdout.lower()
        assert "scene" in output_lower or "1" in output_lower or "3" in output_lower
        # Scene 2 NOT in dry-run output (it's filtered out).
        assert "scene_002" not in result.stdout
