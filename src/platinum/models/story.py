"""Canonical Story data model + JSON I/O.

Story is the source of truth — persisted as per-story ``story.json`` under
``data/stories/<id>/``. SQLite (see ``models/db.py``) is a derived queryable
index projected from these documents. Atomic write (tmp + ``os.replace``)
keeps the JSON coherent even if the process dies mid-stage.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REGENERATE = "regenerate"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _path_to_str(p: Optional[Path]) -> Optional[str]:
    if p is None:
        return None
    # POSIX-style separators keep the JSON portable across platforms.
    return str(p).replace("\\", "/")


def _str_to_path(s: Any) -> Optional[Path]:
    if s is None:
        return None
    return Path(s)


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _str_to_dt(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Source:
    type: str              # "gutenberg" | "wikisource" | "reddit" | "standard_ebooks"
    url: str
    title: str
    author: Optional[str]
    raw_text: str
    fetched_at: datetime
    license: str           # "PD-US" | "CC-BY-4.0" | ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "url": self.url,
            "title": self.title,
            "author": self.author,
            "raw_text": self.raw_text,
            "fetched_at": _dt_to_str(self.fetched_at),
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Source":
        fetched = _str_to_dt(d["fetched_at"])
        assert fetched is not None, "Source.fetched_at is required"
        return cls(
            type=d["type"],
            url=d["url"],
            title=d["title"],
            author=d.get("author"),
            raw_text=d["raw_text"],
            fetched_at=fetched,
            license=d["license"],
        )


@dataclass
class Adapted:
    title: str
    synopsis: str
    narration_script: str
    estimated_duration_seconds: float
    tone_notes: str
    arc: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "synopsis": self.synopsis,
            "narration_script": self.narration_script,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "tone_notes": self.tone_notes,
            "arc": dict(self.arc),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Adapted":
        return cls(
            title=d["title"],
            synopsis=d["synopsis"],
            narration_script=d["narration_script"],
            estimated_duration_seconds=d["estimated_duration_seconds"],
            tone_notes=d["tone_notes"],
            arc=dict(d.get("arc", {})),
        )


@dataclass
class Scene:
    id: str                                 # "scene_001"
    index: int
    narration_text: str
    narration_duration_seconds: float = 0.0
    narration_audio_path: Optional[Path] = None
    visual_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    ip_adapter_reference: Optional[Path] = None
    controlnet_depth: Optional[Path] = None
    keyframe_candidates: list[Path] = field(default_factory=list)
    keyframe_scores: list[float] = field(default_factory=list)
    keyframe_path: Optional[Path] = None
    video_path: Optional[Path] = None
    video_upscaled_path: Optional[Path] = None
    video_graded_path: Optional[Path] = None
    video_duration_seconds: float = 0.0
    music_cue: Optional[str] = None
    sfx_cues: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    review_status: ReviewStatus = ReviewStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "narration_text": self.narration_text,
            "narration_duration_seconds": self.narration_duration_seconds,
            "narration_audio_path": _path_to_str(self.narration_audio_path),
            "visual_prompt": self.visual_prompt,
            "negative_prompt": self.negative_prompt,
            "ip_adapter_reference": _path_to_str(self.ip_adapter_reference),
            "controlnet_depth": _path_to_str(self.controlnet_depth),
            "keyframe_candidates": [_path_to_str(p) for p in self.keyframe_candidates],
            "keyframe_scores": list(self.keyframe_scores),
            "keyframe_path": _path_to_str(self.keyframe_path),
            "video_path": _path_to_str(self.video_path),
            "video_upscaled_path": _path_to_str(self.video_upscaled_path),
            "video_graded_path": _path_to_str(self.video_graded_path),
            "video_duration_seconds": self.video_duration_seconds,
            "music_cue": self.music_cue,
            "sfx_cues": list(self.sfx_cues),
            "validation": dict(self.validation),
            "review_status": self.review_status.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scene":
        return cls(
            id=d["id"],
            index=d["index"],
            narration_text=d["narration_text"],
            narration_duration_seconds=d.get("narration_duration_seconds", 0.0),
            narration_audio_path=_str_to_path(d.get("narration_audio_path")),
            visual_prompt=d.get("visual_prompt"),
            negative_prompt=d.get("negative_prompt"),
            ip_adapter_reference=_str_to_path(d.get("ip_adapter_reference")),
            controlnet_depth=_str_to_path(d.get("controlnet_depth")),
            keyframe_candidates=[Path(p) for p in d.get("keyframe_candidates", []) if p],
            keyframe_scores=list(d.get("keyframe_scores", [])),
            keyframe_path=_str_to_path(d.get("keyframe_path")),
            video_path=_str_to_path(d.get("video_path")),
            video_upscaled_path=_str_to_path(d.get("video_upscaled_path")),
            video_graded_path=_str_to_path(d.get("video_graded_path")),
            video_duration_seconds=d.get("video_duration_seconds", 0.0),
            music_cue=d.get("music_cue"),
            sfx_cues=list(d.get("sfx_cues", [])),
            validation=dict(d.get("validation", {})),
            review_status=ReviewStatus(d.get("review_status", ReviewStatus.PENDING.value)),
        )


@dataclass
class StageRun:
    """One execution of one pipeline stage. Append-log: multiple StageRuns
    for the same stage indicate retries."""

    stage: str
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "started_at": _dt_to_str(self.started_at),
            "completed_at": _dt_to_str(self.completed_at),
            "error": self.error,
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageRun":
        return cls(
            stage=d["stage"],
            status=StageStatus(d["status"]),
            started_at=_str_to_dt(d.get("started_at")),
            completed_at=_str_to_dt(d.get("completed_at")),
            error=d.get("error"),
            artifacts=dict(d.get("artifacts", {})),
        )


@dataclass
class Story:
    id: str                                      # "story_2026_04_24_001"
    track: str                                   # "atmospheric_horror"
    source: Source
    adapted: Optional[Adapted] = None
    scenes: list[Scene] = field(default_factory=list)
    audio: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    publish: dict[str, Any] = field(default_factory=dict)
    review_gates: dict[str, Any] = field(default_factory=dict)
    stages: list[StageRun] = field(default_factory=list)

    # --- Serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "track": self.track,
            "source": self.source.to_dict(),
            "adapted": self.adapted.to_dict() if self.adapted else None,
            "scenes": [s.to_dict() for s in self.scenes],
            "audio": dict(self.audio),
            "video": dict(self.video),
            "publish": dict(self.publish),
            "review_gates": dict(self.review_gates),
            "stages": [r.to_dict() for r in self.stages],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Story":
        return cls(
            id=d["id"],
            track=d["track"],
            source=Source.from_dict(d["source"]),
            adapted=Adapted.from_dict(d["adapted"]) if d.get("adapted") else None,
            scenes=[Scene.from_dict(s) for s in d.get("scenes", [])],
            audio=dict(d.get("audio", {})),
            video=dict(d.get("video", {})),
            publish=dict(d.get("publish", {})),
            review_gates=dict(d.get("review_gates", {})),
            stages=[StageRun.from_dict(r) for r in d.get("stages", [])],
        )

    # --- File I/O --------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomic write via tempfile + ``os.replace``.

        Writes to a sibling tmp file in the same directory, then atomically
        renames. If the process dies before ``os.replace``, the target file
        is untouched; after, the new contents are fully visible.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> "Story":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # --- Convenience -----------------------------------------------------

    def latest_stage_run(self, stage_name: str) -> Optional[StageRun]:
        """Most recent StageRun for the named stage (or None)."""
        for run in reversed(self.stages):
            if run.stage == stage_name:
                return run
        return None
