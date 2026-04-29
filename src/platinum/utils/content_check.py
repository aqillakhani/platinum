"""Content fidelity checker -- Claude vision rates how well a rendered image
depicts the prompt's specific content (S7.1.A4).

Three layers mirror utils/aesthetics.py:
- ContentChecker Protocol -- the contract.
- FakeContentChecker -- deterministic stub for tests (lands in A4.2).
- ClaudeContentChecker -- live Anthropic vision call (lands in A4.3).

Used by keyframe_generator after the LAION aesthetic gate to filter out
candidates that look great but don't depict the requested narrative beat
(the "moody-but-wrong-content" failure mode that drove S7 Phase 2's
4/16 approval rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


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
