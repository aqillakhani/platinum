"""Tests for utils/comfyui.py -- ComfyClient Protocol + FakeComfyClient."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
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


def _build_mock_handler(handlers: dict[tuple[str, str], httpx.Response]):
    """Build a MockTransport handler that dispatches by (method, path-prefix)."""

    def handler(request: httpx.Request) -> httpx.Response:
        for (method, prefix), response in handlers.items():
            if request.method == method and request.url.path.startswith(prefix):
                return response
        return httpx.Response(404, json={"error": f"unmatched {request.method} {request.url}"})

    return handler


async def test_http_comfy_client_health_check_200_returns_true() -> None:
    from platinum.utils.comfyui import HttpComfyClient

    handler = _build_mock_handler(
        {("GET", "/system_stats"): httpx.Response(200, json={"system": {"os": "linux"}})}
    )
    transport = httpx.MockTransport(handler)
    client = HttpComfyClient(host="http://stub:8188", transport=transport)
    assert await client.health_check() is True


async def test_http_comfy_client_health_check_500_returns_false() -> None:
    from platinum.utils.comfyui import HttpComfyClient

    handler = _build_mock_handler(
        {("GET", "/system_stats"): httpx.Response(500, json={"err": "boom"})}
    )
    transport = httpx.MockTransport(handler)
    client = HttpComfyClient(host="http://stub:8188", transport=transport)
    assert await client.health_check() is False


async def test_http_comfy_client_generate_image_happy_path(tmp_path: Path) -> None:
    """POST /prompt -> poll /history/<id> -> GET /view -> write bytes to output_path."""
    from platinum.utils.comfyui import HttpComfyClient

    expected_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # plausible PNG header + padding
    history_payload = {
        "abc123": {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "8": {
                    "images": [
                        {"filename": "flux_dev_00001_.png", "subfolder": "", "type": "output"}
                    ]
                }
            },
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/prompt":
            body = json.loads(request.content)
            assert "prompt" in body and "client_id" in body
            return httpx.Response(200, json={"prompt_id": "abc123"})
        if request.method == "GET" and request.url.path == "/history/abc123":
            return httpx.Response(200, json=history_payload)
        if request.method == "GET" and request.url.path == "/view":
            assert request.url.params["filename"] == "flux_dev_00001_.png"
            return httpx.Response(200, content=expected_bytes)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = HttpComfyClient(host="http://stub:8188", transport=transport, poll_interval=0.0)
    out = tmp_path / "scene_001" / "candidate_0.png"
    result = await client.generate_image(
        workflow={"6": {"class_type": "KSampler", "inputs": {"seed": 1}}},
        output_path=out,
    )
    assert result == out
    assert out.read_bytes() == expected_bytes


async def test_http_comfy_client_generate_image_polls_until_complete(tmp_path: Path) -> None:
    """Two polling responses: first incomplete, second complete."""
    from platinum.utils.comfyui import HttpComfyClient

    expected_bytes = b"\x89PNG\r\n\x1a\n_done"
    poll_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if request.method == "GET" and request.url.path == "/history/p1":
            poll_count["n"] += 1
            if poll_count["n"] < 2:
                return httpx.Response(200, json={})  # not yet in history
            return httpx.Response(
                200,
                json={
                    "p1": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "8": {
                                "images": [
                                    {"filename": "x.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        if request.method == "GET" and request.url.path == "/view":
            return httpx.Response(200, content=expected_bytes)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = HttpComfyClient(host="http://stub:8188", transport=transport, poll_interval=0.0)
    out = tmp_path / "x.png"
    await client.generate_image(workflow={}, output_path=out)
    assert poll_count["n"] >= 2
    assert out.read_bytes() == expected_bytes


async def test_http_comfy_client_generate_image_raises_on_error_status(tmp_path: Path) -> None:
    from platinum.utils.comfyui import HttpComfyClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "errp"})
        if request.method == "GET" and request.url.path == "/history/errp":
            return httpx.Response(
                200,
                json={
                    "errp": {
                        "status": {"completed": False, "status_str": "error"},
                        "outputs": {},
                    }
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = HttpComfyClient(host="http://stub:8188", transport=transport, poll_interval=0.0)
    out = tmp_path / "x.png"
    with pytest.raises(RuntimeError) as exc:
        await client.generate_image(workflow={}, output_path=out)
    assert "error" in str(exc.value).lower()


async def test_http_comfy_client_strips_meta_from_prompt_payload(tmp_path: Path) -> None:
    """`_meta` is platinum's role-mapping block; ComfyUI rejects it as a malformed
    node ("missing class_type"). Verify _submit strips it before posting."""
    from platinum.utils.comfyui import HttpComfyClient

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/prompt":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"prompt_id": "p_meta"})
        if request.method == "GET" and request.url.path == "/history/p_meta":
            return httpx.Response(
                200,
                json={
                    "p_meta": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "8": {
                                "images": [
                                    {"filename": "x.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        if request.method == "GET" and request.url.path == "/view":
            return httpx.Response(200, content=b"\x89PNG\x00")
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = HttpComfyClient(host="http://stub:8188", transport=transport, poll_interval=0.0)
    workflow = {
        "_meta": {"role": {"positive_prompt": "6", "seed": "8"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
        "8": {"class_type": "KSampler", "inputs": {"seed": 1}},
    }
    await client.generate_image(workflow=workflow, output_path=tmp_path / "out.png")

    prompt = captured["body"]["prompt"]
    assert "_meta" not in prompt, f"_meta leaked into ComfyUI /prompt body: {list(prompt)}"
    assert "6" in prompt and "8" in prompt, "real nodes must still be present"


async def test_http_comfy_client_upload_image_form_shape(tmp_path: Path) -> None:
    from platinum.utils.comfyui import HttpComfyClient

    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG_face")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/upload/image":
            assert b"ref.png" in request.content
            return httpx.Response(200, json={"name": "uploaded_ref.png"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = HttpComfyClient(host="http://stub:8188", transport=transport)
    name = await client.upload_image(img)
    assert name == "uploaded_ref.png"
