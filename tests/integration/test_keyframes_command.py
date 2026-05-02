"""Integration tests for `platinum keyframes` CLI command."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from platinum.cli import app


@pytest.fixture
def cli_project(tmp_path: Path) -> Iterator[Path]:
    """Mirror the real project layout under tmp_path for keyframes CLI tests."""
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_path / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    shutil.copytree(
        repo_root / "config" / "workflows",
        tmp_path / "config" / "workflows",
        dirs_exist_ok=True,
    )
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data" / "stories").mkdir(parents=True)
    yield tmp_path


def _minimal_bible(scene_indices: list[int]):  # type: ignore[no-untyped-def]
    """Return a StoryBible covering the given scene indices.

    Used by keyframes integration tests so the S8.B-prepended StoryBibleStage
    sees ``story.bible`` already populated and skips via ``is_complete``. The
    content is intentionally trivial — these tests aren't about bible quality.
    """
    from platinum.models.story_bible import BibleScene, StoryBible

    return StoryBible(
        world_genre_atmosphere="test atmosphere",
        character_continuity={},
        environment_continuity={},
        scenes=[
            BibleScene(
                index=i,
                narrative_beat=f"beat {i}",
                hero_shot="medium shot",
                visible_characters=[],
                gaze_map={},
                props_visible=[],
                blocking="centered",
                light_source="single candle",
                color_anchors=[],
                brightness_floor="low",
            )
            for i in scene_indices
        ],
    )


def _seed_adapted_story(project: Path, story_id: str, n_scenes: int = 3) -> Path:
    """Build a Story with N scenes, all visual_prompts populated, visual_prompts COMPLETE."""
    from platinum.models.story import (
        Scene,
        Source,
        StageRun,
        StageStatus,
        Story,
    )

    indices = list(range(n_scenes))
    s = Story(
        id=story_id,
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url=f"https://example/{story_id}",
            title=f"Title {story_id}",
            author="A",
            raw_text="raw...",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
        scenes=[
            Scene(
                id=f"scene_{i:03d}",
                index=i,
                narration_text=f"Narration {i}.",
                visual_prompt=f"prompt {i}",
                negative_prompt="bright daylight",
            )
            for i in indices
        ],
        stages=[
            StageRun(
                stage="visual_prompts",
                status=StageStatus.COMPLETE,
                started_at=datetime(2026, 4, 25),
                completed_at=datetime(2026, 4, 25),
            )
        ],
        bible=_minimal_bible(indices),
    )
    story_dir = project / "data" / "stories" / story_id
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")
    return story_dir / "story.json"


def _redirect_config_to(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Config so its __init__ uses `project` as root (Config defaults to _ROOT)."""
    from platinum import cli as cli_mod

    original_init = cli_mod.Config.__init__

    def init_with_root(self, root=None):  # type: ignore[no-untyped-def]
        original_init(self, root=project)

    monkeypatch.setattr(cli_mod.Config, "__init__", init_with_root)


def test_keyframes_dry_run_prints_plan_and_exits_zero(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(cli_project)
    monkeypatch.setenv("PLATINUM_COMFYUI_HOST", "http://test:8188")
    monkeypatch.setenv("PLATINUM_AESTHETICS_HOST", "http://test:8189")
    _redirect_config_to(cli_project, monkeypatch)

    _seed_adapted_story(cli_project, "TEST_STORY", n_scenes=3)

    runner = CliRunner()
    result = runner.invoke(
        app, ["keyframes", "TEST_STORY", "--scenes", "0,2", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "0" in result.output
    assert "2" in result.output
    assert "test:8188" in result.output or "test:8189" in result.output


def test_keyframes_invalid_scenes_raises_bad_parameter(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--scenes 0,abc,2 -> non-zero exit with 'comma-separated integers' or 'Invalid'."""
    monkeypatch.chdir(cli_project)
    _redirect_config_to(cli_project, monkeypatch)
    _seed_adapted_story(cli_project, "X", n_scenes=3)

    runner = CliRunner()
    result = runner.invoke(app, ["keyframes", "X", "--scenes", "0,abc,2"])
    assert result.exit_code != 0
    output_lower = result.output.lower()
    assert "comma-separated integers" in output_lower or "invalid" in output_lower


def test_keyframes_out_of_range_scenes_raises_bad_parameter(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--scenes 0,99 against a 3-scene story -> non-zero exit mentioning the bad index."""
    monkeypatch.chdir(cli_project)
    _redirect_config_to(cli_project, monkeypatch)
    _seed_adapted_story(cli_project, "Y", n_scenes=3)

    runner = CliRunner()
    result = runner.invoke(app, ["keyframes", "Y", "--scenes", "0,99"])
    assert result.exit_code != 0
    assert "99" in result.output


def test_keyframes_story_without_visual_prompts_exits_1(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Story with no visual_prompts COMPLETE StageRun -> exit 1 with red message."""
    from platinum.models.story import Scene, Source, Story

    monkeypatch.chdir(cli_project)
    _redirect_config_to(cli_project, monkeypatch)

    s = Story(
        id="NO_VP",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="https://example/NO_VP",
            title="NO VP",
            author="A",
            raw_text="raw...",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
        scenes=[Scene(id="scene_000", index=0, narration_text="x", visual_prompt="x")],
        stages=[],  # NO visual_prompts stage
    )
    story_dir = cli_project / "data" / "stories" / "NO_VP"
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")

    runner = CliRunner()
    result = runner.invoke(app, ["keyframes", "NO_VP"])
    assert result.exit_code == 1
    assert "no completed visual_prompts" in result.output.lower()


def test_keyframes_missing_story_exits_1(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nonexistent story id -> exit 1 with 'not found' message."""
    monkeypatch.chdir(cli_project)
    _redirect_config_to(cli_project, monkeypatch)

    runner = CliRunner()
    result = runner.invoke(app, ["keyframes", "DOES_NOT_EXIST"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_keyframes_runs_charref_when_unresolved_then_halts_for_pick(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S7.1 / C2: when story.characters lacks picks, `platinum keyframes`
    runs CharacterReferenceStage to materialise candidate ref portraits,
    then halts cleanly (exit 0) with a pointer at the review UI.

    The previous behavior was to exit 1 immediately without generating
    candidates -- which was a deadlock: the gate told the user to pick
    refs from candidates that no command had ever produced. C2 replaces
    that early-exit gate with a two-phase orchestrator flow.
    """
    from platinum.models.story import (
        Scene,
        Source,
        StageRun,
        StageStatus,
        Story,
    )
    from platinum.pipeline.character_references import CharacterReferenceStage
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    monkeypatch.chdir(cli_project)
    _redirect_config_to(cli_project, monkeypatch)

    story_id = "STORY_WITH_CHARS"
    s = Story(
        id=story_id,
        track="atmospheric_horror",
        source=Source(
            type="gutenberg", url="x", title="t", author="a",
            raw_text="r", fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
        scenes=[
            Scene(
                id="scene_001", index=1,
                narration_text="Fortunato laughed at Montresor.",
                visual_prompt="x", negative_prompt="y",
                character_refs=["Fortunato", "Montresor"],
            ),
        ],
        stages=[
            StageRun(
                stage="visual_prompts",
                status=StageStatus.COMPLETE,
                started_at=datetime(2026, 4, 25),
                completed_at=datetime(2026, 4, 25),
            )
        ],
    )
    # NB: s.characters is empty -- the user has not picked refs yet.
    story_dir = cli_project / "data" / "stories" / story_id
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")

    # Stub the three stages so we can verify the two-phase flow without
    # spinning up a real Comfy + Flux. CharRef "runs" (records the call)
    # but doesn't write picks -- so the post-run is_complete check still
    # returns False and the command halts cleanly.
    char_called: list[bool] = []
    pose_called: list[bool] = []
    keyframe_called: list[bool] = []

    async def fake_charref_run(self, story, ctx):  # type: ignore[no-untyped-def]
        char_called.append(True)
        return {"characters_discovered": ["Fortunato", "Montresor"]}

    async def fake_pose_run(self, story, ctx):  # type: ignore[no-untyped-def]
        pose_called.append(True)
        return {"prepared_scenes": []}

    async def fake_keyframe_run(self, story, ctx):  # type: ignore[no-untyped-def]
        keyframe_called.append(True)
        return {}

    monkeypatch.setattr(CharacterReferenceStage, "run", fake_charref_run)
    monkeypatch.setattr(PoseDepthMapStage, "run", fake_pose_run)
    monkeypatch.setattr(KeyframeGeneratorStage, "run", fake_keyframe_run)

    runner = CliRunner()
    result = runner.invoke(app, ["keyframes", story_id])

    # Phase 1 ran CharacterReferenceStage; phases 2/3 did NOT run because
    # picks are still missing.
    assert char_called == [True], result.output
    assert pose_called == []
    assert keyframe_called == []
    # Clean halt -- not a hard error.
    assert result.exit_code == 0, result.output
    # Output points at the review UI and names the missing characters.
    output_lower = result.output.lower()
    assert "review characters" in output_lower or "platinum review" in output_lower
    assert "Fortunato" in result.output or "Montresor" in result.output


def test_keyframes_skips_charref_runs_pose_and_keyframe_when_picks_present(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S7.1 / C2: when story.characters has all picks present on disk,
    `platinum keyframes` skips Phase 1 (CharacterReferenceStage already
    complete) and runs PoseDepthMapStage + KeyframeGeneratorStage."""
    from platinum.models.story import (
        Scene,
        Source,
        StageRun,
        StageStatus,
        Story,
    )
    from platinum.pipeline.character_references import CharacterReferenceStage
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    monkeypatch.chdir(cli_project)
    monkeypatch.setenv("PLATINUM_COMFYUI_HOST", "http://test:8188")
    monkeypatch.setenv("PLATINUM_AESTHETICS_HOST", "http://test:8189")
    _redirect_config_to(cli_project, monkeypatch)

    story_id = "STORY_PICKED"
    story_dir = cli_project / "data" / "stories" / story_id
    refs_dir = story_dir / "references"
    (refs_dir / "Fortunato").mkdir(parents=True)
    fortunato_pick = refs_dir / "Fortunato" / "candidate_0.png"
    fortunato_pick.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

    s = Story(
        id=story_id,
        track="atmospheric_horror",
        source=Source(
            type="gutenberg", url="x", title="t", author="a",
            raw_text="r", fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
        scenes=[
            Scene(
                id="scene_001", index=1,
                narration_text="Fortunato laughed.",
                visual_prompt="x", negative_prompt="y",
                character_refs=["Fortunato"],
            ),
        ],
        stages=[
            StageRun(
                stage="visual_prompts",
                status=StageStatus.COMPLETE,
                started_at=datetime(2026, 4, 25),
                completed_at=datetime(2026, 4, 25),
            )
        ],
        characters={"Fortunato": str(fortunato_pick)},
        bible=_minimal_bible([1]),
    )
    s.save(story_dir / "story.json")

    char_called: list[bool] = []
    pose_called: list[bool] = []
    keyframe_called: list[bool] = []

    async def fake_charref_run(self, story, ctx):  # type: ignore[no-untyped-def]
        char_called.append(True)
        return {}

    async def fake_pose_run(self, story, ctx):  # type: ignore[no-untyped-def]
        pose_called.append(True)
        return {}

    async def fake_keyframe_run(self, story, ctx):  # type: ignore[no-untyped-def]
        keyframe_called.append(True)
        return {}

    # Force the orchestrator to actually invoke run() on each stage by
    # pinning is_complete to False -- otherwise PoseDepth + Keyframe both
    # trivially short-circuit on a minimal seed (no composition_notes,
    # nothing to render). The point of this test is the orchestrator's
    # phase-2 stage list, not the individual stages' resume logic.
    monkeypatch.setattr(
        PoseDepthMapStage, "is_complete", lambda self, story: False
    )
    monkeypatch.setattr(
        KeyframeGeneratorStage, "is_complete", lambda self, story: False
    )
    monkeypatch.setattr(CharacterReferenceStage, "run", fake_charref_run)
    monkeypatch.setattr(PoseDepthMapStage, "run", fake_pose_run)
    monkeypatch.setattr(KeyframeGeneratorStage, "run", fake_keyframe_run)

    runner = CliRunner()
    result = runner.invoke(app, ["keyframes", story_id])

    assert result.exit_code == 0, result.output
    assert char_called == []  # is_complete=True -> orchestrator skips Phase 1
    assert pose_called == [True]
    assert keyframe_called == [True]


def test_keyframes_skips_character_check_when_no_refs(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S7.1.B4.6: stories without recurring characters (e.g. documentary
    tracks) skip the character-refs check entirely. The dry-run path
    confirms preconditions all pass."""
    monkeypatch.chdir(cli_project)
    monkeypatch.setenv("PLATINUM_COMFYUI_HOST", "http://test:8188")
    monkeypatch.setenv("PLATINUM_AESTHETICS_HOST", "http://test:8189")
    _redirect_config_to(cli_project, monkeypatch)

    # _seed_adapted_story builds scenes WITHOUT character_refs.
    _seed_adapted_story(cli_project, "STORY_NO_CHARS", n_scenes=2)

    runner = CliRunner()
    result = runner.invoke(
        app, ["keyframes", "STORY_NO_CHARS", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
