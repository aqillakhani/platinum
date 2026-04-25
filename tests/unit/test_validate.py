"""Tests for utils/validate.py quality-gate primitives."""

from __future__ import annotations

import dataclasses

import pytest

from platinum.utils.validate import CheckResult


def test_check_result_is_frozen_and_carries_fields() -> None:
    r = CheckResult(passed=True, metric=0.5, threshold=0.4, reason="ok")
    assert r.passed and r.metric == 0.5 and r.threshold == 0.4 and r.reason == "ok"
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.passed = False  # type: ignore[misc]
