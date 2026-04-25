"""Tests for utils/comfyui.py -- ComfyClient Protocol + FakeComfyClient."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _signature(workflow: dict) -> str:
    """Mirror the FakeComfyClient response-keying scheme."""
    import hashlib

    canonical = json.dumps(workflow, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _trivial_workflow(seed: int = 0) -> dict:
    return {"6": {"class_type": "KSampler", "inputs": {"seed": seed}}}


async def test_fake_comfy_client_satisfies_protocol() -> None:
    from platinum.utils.comfyui import ComfyClient, FakeComfyClient

    client = FakeComfyClient(responses={})
    assert isinstance(client, ComfyClient)


async def test_fake_comfy_client_copies_fixture_to_output_path(tmp_path: Path) -> None:
    from platinum.utils.comfyui import FakeComfyClient

    fixtures_root = Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"
    src = fixtures_root / "candidate_0.png"
    wf = _trivial_workflow(seed=1)
    sig = _signature(wf)
    client = FakeComfyClient(responses={sig: [src]})
    out = tmp_path / "scene_001" / "candidate_0.png"
    returned = await client.generate_image(workflow=wf, output_path=out)
    assert returned == out
    assert out.exists()
    assert out.read_bytes() == src.read_bytes()


async def test_fake_comfy_client_rotates_through_responses(tmp_path: Path) -> None:
    from platinum.utils.comfyui import FakeComfyClient

    fixtures_root = Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"
    a = fixtures_root / "candidate_0.png"
    b = fixtures_root / "candidate_1.png"
    wf = _trivial_workflow(seed=2)
    sig = _signature(wf)
    client = FakeComfyClient(responses={sig: [a, b]})
    out_a = tmp_path / "a.png"
    out_b = tmp_path / "b.png"
    await client.generate_image(workflow=wf, output_path=out_a)
    await client.generate_image(workflow=wf, output_path=out_b)
    assert out_a.read_bytes() == a.read_bytes()
    assert out_b.read_bytes() == b.read_bytes()


async def test_fake_comfy_client_reuses_last_when_exhausted(tmp_path: Path) -> None:
    from platinum.utils.comfyui import FakeComfyClient

    fixtures_root = Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"
    a = fixtures_root / "candidate_0.png"
    wf = _trivial_workflow(seed=3)
    sig = _signature(wf)
    client = FakeComfyClient(responses={sig: [a]})
    out_1 = tmp_path / "1.png"
    out_2 = tmp_path / "2.png"
    await client.generate_image(workflow=wf, output_path=out_1)
    await client.generate_image(workflow=wf, output_path=out_2)
    assert out_1.read_bytes() == a.read_bytes()
    assert out_2.read_bytes() == a.read_bytes()


async def test_fake_comfy_client_raises_when_workflow_unknown(tmp_path: Path) -> None:
    from platinum.utils.comfyui import FakeComfyClient

    client = FakeComfyClient(responses={})  # no responses configured
    wf = _trivial_workflow(seed=99)
    out = tmp_path / "x.png"
    with pytest.raises(KeyError):
        await client.generate_image(workflow=wf, output_path=out)


async def test_fake_comfy_client_records_calls(tmp_path: Path) -> None:
    """FakeComfyClient.calls is a public list of {workflow_signature, output_path} dicts."""
    from platinum.utils.comfyui import FakeComfyClient

    fixtures_root = Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"
    a = fixtures_root / "candidate_0.png"
    wf = _trivial_workflow(seed=4)
    sig = _signature(wf)
    client = FakeComfyClient(responses={sig: [a]})
    out = tmp_path / "out.png"
    await client.generate_image(workflow=wf, output_path=out)
    assert len(client.calls) == 1
    assert client.calls[0]["workflow_signature"] == sig
    assert client.calls[0]["output_path"] == out


async def test_fake_comfy_client_health_check_returns_true() -> None:
    from platinum.utils.comfyui import FakeComfyClient

    client = FakeComfyClient(responses={})
    assert await client.health_check() is True


async def test_fake_comfy_client_upload_image_returns_basename(tmp_path: Path) -> None:
    from platinum.utils.comfyui import FakeComfyClient

    img = tmp_path / "my_face.png"
    img.write_bytes(b"x")
    client = FakeComfyClient(responses={})
    name = await client.upload_image(img)
    assert name == "my_face.png"
