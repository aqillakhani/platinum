"""Verify _seeds_for_scene is backwards-compatible at regen_count=0
and produces distinct seeds at regen_count>=1.

S7 §4.3: seed = scene_index * 1000 + regen_count * 100 + offset.
"""
from __future__ import annotations

from platinum.pipeline.keyframe_generator import _seeds_for_scene


def test_seeds_unchanged_with_default_regen_count_zero() -> None:
    """Default behavior identical to pre-S7 (no regen_count parameter)."""
    # scene_index=1, n=3, regen_count default
    assert _seeds_for_scene(1, 3) == (1000, 1001, 1002)
    assert _seeds_for_scene(7, 3) == (7000, 7001, 7002)


def test_seeds_differ_with_regen_count_one() -> None:
    """regen_count=1 shifts seeds by 100 per regen."""
    base = _seeds_for_scene(1, 3, regen_count=0)
    once = _seeds_for_scene(1, 3, regen_count=1)
    twice = _seeds_for_scene(1, 3, regen_count=2)
    assert base == (1000, 1001, 1002)
    assert once == (1100, 1101, 1102)
    assert twice == (1200, 1201, 1202)
    # All three sets disjoint
    assert set(base) & set(once) == set()
    assert set(once) & set(twice) == set()


def test_seeds_deterministic_per_regen_count() -> None:
    """Same (scene_index, n, regen_count) always produces same seeds."""
    a = _seeds_for_scene(5, 4, regen_count=2)
    b = _seeds_for_scene(5, 4, regen_count=2)
    assert a == b
    assert a == (5200, 5201, 5202, 5203)
