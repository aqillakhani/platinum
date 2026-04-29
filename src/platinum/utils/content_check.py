"""Content fidelity checker -- Claude vision rates how well a rendered image
depicts the prompt's specific content (S7.1.A4).

Three layers mirror utils/aesthetics.py:
- ContentChecker Protocol -- the contract.
- FakeContentChecker -- deterministic stub for tests.
- ClaudeContentChecker -- live Anthropic vision call.

Used by keyframe_generator after the LAION aesthetic gate to filter out
candidates that look great but don't depict the requested narrative beat
(the "moody-but-wrong-content" failure mode that drove S7 Phase 2's
4/16 approval rate).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from platinum.utils.claude import Recorder
from platinum.utils.claude import call as claude_call


@dataclass(frozen=True, slots=True)
class ContentCheckResult:
    """Per-image content-fidelity score from a vision model.

    score: 1-10 integer rating of how specifically the image depicts the
        prompt. 10 = depicts everything; 5 = atmospheric only; 1 = nothing
        matches.
    missing: list of named elements the prompt asked for that the image
        does not depict (e.g. ["chains", "fog", "trowel"]).
    rationale: short prose justification from the model.
    raw_response: JSON-serialized tool_input for audit trail.
    """

    score: int
    missing: list[str]
    rationale: str
    raw_response: str


@runtime_checkable
class ContentChecker(Protocol):
    """Score image content fidelity against a text prompt.

    The keyword-only signature keeps callsites self-documenting at the
    keyframe_generator integration site, matching the visual_prompts
    pipeline's Claude-call style.
    """

    async def check(
        self, *, prompt: str, image_path: Path
    ) -> ContentCheckResult: ...


class FakeContentChecker:
    """Deterministic ContentChecker for tests.

    Configurable per-path scores + missing element lists; falls back to
    `default_score` (8 = neutral pass against a typical >=6 threshold)
    for unmapped paths.

    `call_count` tracks invocations so tests can assert the gate skipped
    candidates that failed earlier checks (used by A4.5's "content_gate=off
    skips check" test).
    """

    def __init__(
        self,
        *,
        scores: dict[Path, int] | None = None,
        missing_by_path: dict[Path, list[str]] | None = None,
        default_score: int = 8,
    ) -> None:
        self._scores: dict[Path, int] = dict(scores or {})
        self._missing: dict[Path, list[str]] = dict(missing_by_path or {})
        self._default = int(default_score)
        self.call_count = 0

    async def check(
        self, *, prompt: str, image_path: Path
    ) -> ContentCheckResult:
        self.call_count += 1
        score = self._scores.get(image_path, self._default)
        missing = list(self._missing.get(image_path, []))
        return ContentCheckResult(
            score=score,
            missing=missing,
            rationale=f"fake check for {image_path.name}",
            raw_response='{"score": ' + str(score) + "}",
        )


CONTENT_CHECK_TOOL: dict[str, Any] = {
    "name": "submit_content_check",
    "description": "Score how well the image depicts the prompt.",
    "input_schema": {
        "type": "object",
        "required": ["score", "missing", "rationale"],
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": (
                    "1-10 rating of how specifically the image depicts the prompt. "
                    "10 = depicts everything; 5 = mostly atmospheric only; "
                    "1 = nothing matches."
                ),
            },
            "missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific named elements from the prompt that are NOT depicted. "
                    "Empty list when the image hits everything."
                ),
            },
            "rationale": {
                "type": "string",
                "description": "1-2 sentences justifying the score.",
            },
        },
    },
}


CONTENT_CHECK_SYSTEM = (
    "You are a content fidelity reviewer for an AI image-generation pipeline. "
    "Given an image and the prompt that produced it, rate 1-10 how specifically "
    "the image depicts the prompt's named subjects, actions, spatial layout, "
    "and atmospheric details. 10 = depicts every named element specifically; "
    "5 = mostly atmospheric only, generic match; 1 = nothing matches. "
    "List specific elements MISSING from the image (named subjects, actions, "
    "or props the prompt asked for that aren't visible). "
    "Submit your evaluation via the submit_content_check tool."
)


class ClaudeContentChecker:
    """Live Anthropic-vision implementation of ContentChecker.

    Defaults to Haiku 4.5 -- a vision-capable, low-cost model well-suited
    to the per-candidate gating workload (3 candidates per scene * 16 scenes
    = ~48 calls per Cask render at ~$0.005 each = ~$0.24/render).

    Uses the project Claude wrapper (utils/claude.call) so all calls
    participate in the existing fixture recorder, retry, and cost-tracking
    infrastructure.
    """

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5",
        recorder: Recorder | None = None,
        db_path: Path | None = None,
        story_id: str = "content_check",
    ) -> None:
        self._model = model
        self._recorder = recorder
        self._db_path = db_path or Path("data/platinum.db")
        self._story_id = story_id

    async def check(
        self, *, prompt: str, image_path: Path
    ) -> ContentCheckResult:
        with image_path.open("rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")
        media_type = (
            "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"}
            else "image/png"
        )
        system = [{"type": "text", "text": CONTENT_CHECK_SYSTEM}]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": f"Prompt: {prompt}"},
                ],
            }
        ]
        result = await claude_call(
            model=self._model,
            system=system,
            messages=messages,
            tool=CONTENT_CHECK_TOOL,
            story_id=self._story_id,
            stage="content_check",
            db_path=self._db_path,
            recorder=self._recorder,
        )
        ti = result.tool_input
        return ContentCheckResult(
            score=int(ti["score"]),
            missing=list(ti.get("missing", [])),
            rationale=str(ti.get("rationale", "")),
            raw_response=json.dumps(ti),
        )
