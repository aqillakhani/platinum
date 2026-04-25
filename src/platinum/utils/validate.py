"""Quality-gate primitives.

Five sync `check_*` functions returning a uniform `CheckResult`. Used by
Sessions 6/8/13 to reject AI-generated assets before human review.

`passed=True` always means the asset cleared the gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a quality check.

    `metric` is the measured value; `threshold` is what it was compared
    against; `reason` is a human-readable summary suitable for log lines
    and review-UI display.
    """

    passed: bool
    metric: float
    threshold: float
    reason: str
