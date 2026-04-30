"""Tests for scripts/preflight_check.py."""
from __future__ import annotations


def test_check_hf_token_happy_path(monkeypatch):
    """Valid HF token returns (True, descriptive ok message)."""
    import httpx
    from preflight_check import _check_hf_token

    # Patch httpx.Client to return 206 Partial Content for the Range request.
    class _MockResponse:
        status_code = 206
        text = ""
        content = b"x" * 1024

    class _MockClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, headers):
            assert headers.get("Range") == "bytes=0-1023"
            assert "Bearer" in headers.get("Authorization", "")
            return _MockResponse()

    monkeypatch.setattr(httpx, "Client", _MockClient)
    ok, msg = _check_hf_token("hf_real_token")
    assert ok is True
    assert "OK" in msg


def test_check_hf_token_unauthorized(monkeypatch):
    """403 from HF: returns (False, error message including 403)."""
    import httpx
    from preflight_check import _check_hf_token

    class _MockResponse:
        status_code = 403
        text = "Forbidden"
        content = b""

    class _MockClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, headers): return _MockResponse()

    monkeypatch.setattr(httpx, "Client", _MockClient)
    ok, msg = _check_hf_token("hf_bad_token")
    assert ok is False
    assert "403" in msg


def test_check_workflow_json_valid(tmp_path):
    """Valid workflow with all required roles passes."""
    import json

    from preflight_check import _check_workflow_json

    wf = {
        "_meta": {"role": {
            "positive_prompt": "3", "negative_prompt": "4", "empty_latent": "5",
            "sampler": "6", "save_image": "8",
            "model_sampling_flux": "10", "flux_guidance": "11",
        }},
        "3": {}, "4": {}, "5": {}, "6": {}, "8": {}, "10": {}, "11": {},
    }
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(wf))
    ok, msg = _check_workflow_json(p)
    assert ok is True
    assert "OK" in msg


def test_check_workflow_json_missing_role(tmp_path):
    """Missing required role: returns (False, error listing missing role)."""
    import json

    from preflight_check import _check_workflow_json

    wf = {
        "_meta": {"role": {"positive_prompt": "3"}},  # everything else missing
        "3": {},
    }
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(wf))
    ok, msg = _check_workflow_json(p)
    assert ok is False
    assert "missing" in msg.lower()


def test_check_comfyui_alive(monkeypatch):
    import httpx
    from preflight_check import _check_comfyui_alive

    class _MockResponse:
        status_code = 200
        def json(self): return {"devices": [{"name": "RTX A6000"}]}

    class _MockClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): return _MockResponse()

    monkeypatch.setattr(httpx, "Client", _MockClient)
    ok, msg = _check_comfyui_alive("http://test:8188")
    assert ok is True
    assert "RTX A6000" in msg


def test_check_comfyui_503(monkeypatch):
    import httpx
    from preflight_check import _check_comfyui_alive

    class _MockResponse:
        status_code = 503
        def json(self): return {}

    class _MockClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): return _MockResponse()

    monkeypatch.setattr(httpx, "Client", _MockClient)
    ok, msg = _check_comfyui_alive("http://test:8188")
    assert ok is False
    assert "503" in msg


def test_check_score_server_alive(monkeypatch):
    import httpx
    from preflight_check import _check_score_server_alive

    class _MockResponse:
        status_code = 200

    class _MockClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): return _MockResponse()

    monkeypatch.setattr(httpx, "Client", _MockClient)
    ok, msg = _check_score_server_alive("http://test:8189")
    assert ok is True


def test_check_score_server_503(monkeypatch):
    import httpx
    from preflight_check import _check_score_server_alive

    class _MockResponse:
        status_code = 503

    class _MockClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): return _MockResponse()

    monkeypatch.setattr(httpx, "Client", _MockClient)
    ok, msg = _check_score_server_alive("http://test:8189")
    assert ok is False


def test_workflow_signature_stable(tmp_path):
    """sha256 of canonical-form JSON is stable for equivalent input."""
    import json

    from preflight_check import _workflow_signature

    wf1 = {"a": 1, "b": [1, 2, 3]}
    wf2 = {"b": [1, 2, 3], "a": 1}                    # different key order
    p1 = tmp_path / "wf1.json"
    p2 = tmp_path / "wf2.json"
    p1.write_text(json.dumps(wf1))
    p2.write_text(json.dumps(wf2))
    sig1 = _workflow_signature(p1)
    sig2 = _workflow_signature(p2)
    assert sig1 == sig2
    assert len(sig1) == 12                            # short hash


class TestWanPreflightChecks:
    def test_check_wan_workflow_valid(self, tmp_path) -> None:
        """Wan workflow JSON has all required _meta.role entries."""
        from preflight_check import _check_wan_workflow_json

        good = tmp_path / "wan_good.json"
        good.write_text('{"_meta":{"role":{"image_in":"10","prompt":"20","seed":"30","video_out":"60"}},"10":{},"20":{},"30":{},"60":{}}')
        ok, msg = _check_wan_workflow_json(good)
        assert ok, msg

        bad = tmp_path / "wan_bad.json"
        bad.write_text('{"_meta":{"role":{"image_in":"10"}},"10":{}}')
        ok, msg = _check_wan_workflow_json(bad)
        assert not ok
        assert "missing roles" in msg

    def test_check_wan_weights_present(self, tmp_path) -> None:
        import os

        from preflight_check import _check_wan_weights

        # Empty dir -> fail.
        ok, msg = _check_wan_weights(tmp_path)
        assert not ok
        assert "HIGH" in msg or "LOW" in msg or "missing" in msg.lower()

        # All 4 files present, just above the (loose) preflight min size.
        # Production weights are ~28/28/1.4/11 GB but the preflight threshold
        # is only 100MB (any-file-at-all sanity); tests float at 110MB.
        # NOTE: os.truncate on Windows allocates real bytes (NTFS sparse needs
        # FSCTL_SET_SPARSE), so test file sizes here are also disk usage.
        (tmp_path / "diffusion_models").mkdir()
        high_path = tmp_path / "diffusion_models" / "Wan2_2-I2V-A14B-HIGH_bf16.safetensors"
        high_path.touch()
        os.truncate(high_path, 110_000_000)

        low_path = tmp_path / "diffusion_models" / "Wan2_2-I2V-A14B-LOW_bf16.safetensors"
        low_path.touch()
        os.truncate(low_path, 110_000_000)

        (tmp_path / "vae").mkdir()
        vae_path = tmp_path / "vae" / "Wan2_2_VAE_bf16.safetensors"
        vae_path.touch()
        os.truncate(vae_path, 110_000_000)

        (tmp_path / "text_encoders").mkdir()
        umt5_path = tmp_path / "text_encoders" / "umt5_xxl.pth"
        umt5_path.touch()
        os.truncate(umt5_path, 110_000_000)

        ok, msg = _check_wan_weights(tmp_path)
        assert ok, msg


def test_main_wan_mode_routes_to_wan_checks(tmp_path, monkeypatch, capsys):
    """When --workflow points at a Wan-shaped workflow, main() runs the Wan
    checks (workflow JSON, weights, extension) -- not the Flux checks.
    Regression for an S8 cumulative-review gap where _check_wan_* were
    defined but never wired into main()."""
    import json
    import os
    import sys

    import httpx
    import preflight_check as pc

    # 1. Build a Wan-shaped workflow JSON.
    wan_wf = tmp_path / "wan22.json"
    wan_wf.write_text(json.dumps({
        "_meta": {"role": {
            "image_in": "10", "prompt": "20", "seed": "30", "video_out": "60",
        }},
        "10": {}, "20": {}, "30": {}, "60": {},
    }))

    # 2. Build a fake Wan models dir with files just above the 100MB preflight
    #    floor (Kijai single-file repackages + Wan-AI UMT5; real weights are
    #    much larger but tests don't need to allocate real-size bytes).
    (tmp_path / "diffusion_models").mkdir()
    p1 = tmp_path / "diffusion_models" / "Wan2_2-I2V-A14B-HIGH_bf16.safetensors"
    p1.touch()
    os.truncate(p1, 110_000_000)
    p2 = tmp_path / "diffusion_models" / "Wan2_2-I2V-A14B-LOW_bf16.safetensors"
    p2.touch()
    os.truncate(p2, 110_000_000)
    (tmp_path / "vae").mkdir()
    p3 = tmp_path / "vae" / "Wan2_2_VAE_bf16.safetensors"
    p3.touch()
    os.truncate(p3, 110_000_000)
    (tmp_path / "text_encoders").mkdir()
    p4 = tmp_path / "text_encoders" / "umt5_xxl.pth"
    p4.touch()
    os.truncate(p4, 110_000_000)

    # 3. Stub all network checks (HF range GET, ComfyUI /system_stats,
    #    score-server /health) to pass.
    class _Ok:
        status_code = 200
        text = ""
        content = b"x" * 1024
        def json(self):
            return {"devices": [{"name": "RTX A6000"}]}

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def get(self, *a, **k):
            return _Ok()

    monkeypatch.setattr(httpx, "Client", _Client)

    # 4. Stub the extension-on-disk check (the real one looks at a vast.ai-only
    #    hardcoded path).
    monkeypatch.setattr(
        pc, "_check_wan_extension_importable",
        lambda: (True, "WanVideoWrapper present (test-stub)"),
    )

    # 5. Required env vars + sys.argv.
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("PLATINUM_COMFYUI_HOST", "http://localhost:8188")
    monkeypatch.setenv("PLATINUM_AESTHETICS_HOST", "http://localhost:8189")
    monkeypatch.setattr(sys, "argv", [
        "preflight_check.py",
        "--workflow", str(wan_wf),
        "--wan-models-dir", str(tmp_path),
    ])

    rc = pc.main()
    captured = capsys.readouterr()
    out = captured.out
    assert rc == 0, out
    # Wan-specific checks must have run.
    assert "Wan workflow JSON" in out
    assert "Wan weights" in out
    assert "Wan extension" in out
    # Flux-only check label must not have run (would be wrong shape for Wan).
    assert "[OK ] Workflow JSON" not in out
    assert "[FAIL]" not in out


def test_main_flux_mode_unchanged(tmp_path, monkeypatch, capsys):
    """Default mode (Flux workflow path) still runs the original 4 checks
    only. Regression to ensure the Wan branch doesn't leak into Flux mode."""
    import json
    import sys

    import httpx
    import preflight_check as pc

    flux_wf = tmp_path / "flux.json"
    flux_wf.write_text(json.dumps({
        "_meta": {"role": {
            "positive_prompt": "3", "negative_prompt": "4", "empty_latent": "5",
            "sampler": "6", "save_image": "8",
            "model_sampling_flux": "10", "flux_guidance": "11",
        }},
        "3": {}, "4": {}, "5": {}, "6": {}, "8": {}, "10": {}, "11": {},
    }))

    class _Ok:
        status_code = 200
        text = ""
        content = b"x" * 1024
        def json(self):
            return {"devices": [{"name": "RTX A6000"}]}

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def get(self, *a, **k):
            return _Ok()

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("PLATINUM_COMFYUI_HOST", "http://localhost:8188")
    monkeypatch.setenv("PLATINUM_AESTHETICS_HOST", "http://localhost:8189")
    monkeypatch.setattr(sys, "argv", [
        "preflight_check.py",
        "--workflow", str(flux_wf),
    ])

    rc = pc.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Workflow JSON" in out
    assert "Wan workflow JSON" not in out
    assert "Wan weights" not in out
    assert "Wan extension" not in out
