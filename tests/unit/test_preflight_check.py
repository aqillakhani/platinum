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
