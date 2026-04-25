# Session 6 -- Keyframe generator + ComfyUI client + workflow injection (implementation plan)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Ship `src/platinum/utils/comfyui.py` (`ComfyClient` Protocol + `FakeComfyClient` + `HttpComfyClient`), `src/platinum/utils/workflow.py` (`load_workflow` + `inject`), `config/workflows/flux_dev_keyframe.json` (hand-authored Flux.1 Dev workflow tagged via `_meta.role`), `src/platinum/pipeline/keyframe_generator.py` (`KeyframeReport` + pure `generate_for_scene` + `KeyframeGeneratorStage`), and additions to `src/platinum/utils/aesthetics.py` (`MappedFakeScorer` + `RemoteAestheticScorer` stub). All offline, no live GPU.

**Architecture:** Pure-core / impure-shell at the Stage boundary. `ComfyClient` Protocol with `FakeComfyClient` for tests and `HttpComfyClient` for production (vendored shape from gold + httpx `MockTransport` testability). Workflow JSON is static and tagged by node-role; `inject` is a deepcopy-and-mutate pure function. Sequential 3-candidate generation per scene with seeds derived from `scene.index`; LAION score + hand anatomy gate; selection rule = highest-score-among-eligible, ties to lowest index, fallback to candidate 0 if none eligible. Per-scene `story.save(...)` checkpoint inside the Stage so a crash mid-story leaves earlier scenes intact.

**Tech stack:** Python 3.14, `httpx>=0.27`, `Pillow>=10`, `opencv-python>=4.10`, `mediapipe>=0.10`, `numpy>=1.26`, `pytest`+`pytest-asyncio` (auto mode already on). **Zero new top-level dependencies.**

**Design doc:** `docs/plans/2026-04-25-session-6-keyframe-generator-design.md` -- read first if any task feels under-specified.

---

## Pre-flight context (read before Task 1)

### Existing files you will integrate with

- `src/platinum/utils/aesthetics.py` -- already exports `AestheticScorer` Protocol + `FakeAestheticScorer`. Append `MappedFakeScorer` and `RemoteAestheticScorer` here in Task 1.
- `src/platinum/utils/validate.py` -- `check_hand_anomalies(image_path, *, mp_hands_factory=None)` is already shipped; the keyframe generator calls it.
- `src/platinum/pipeline/stage.py` -- the `Stage` ABC. **Important:** `run(self, story, ctx) -> dict[str, Any]` returns an artifacts dict and mutates `story` in place. Default `is_complete()` reads the StageRun log; **do not override it** unless we need to.
- `src/platinum/pipeline/story_adapter.py` -- the canonical pure-core / impure-shell example. `KeyframeGeneratorStage` mirrors `StoryAdapterStage`'s shape.
- `src/platinum/pipeline/context.py` -- `PipelineContext` exposes `config`, `db_path`, and `story_path(story)`.
- `src/platinum/models/story.py` -- `Scene` already has `keyframe_candidates: list[Path]`, `keyframe_scores: list[float]`, `keyframe_path: Path | None`, `validation: dict[str, Any]`. **No model changes needed this session.**
- `src/platinum/config.py` -- `Config.track(track_id)` returns the track dict; `quality_gates` and `visual` blocks already populated for all five tracks.
- `tests/conftest.py` -- exposes `tmp_project`, `config`, `context`, `source`, `story` fixtures. `story` returns a Story with one scene by default; integration tests build their own Story with multiple scenes.
- `tests/_fixtures.py` -- already has `FixtureRecorder`, `make_test_video`, `make_test_video_with_motion`, `make_test_audio`, `make_silent_audio`. Append helpers below the existing functions.
- `tests/unit/test_validate.py:183-204` -- the `_make_fake_hands_factory` helper. Hoist this to `tests/_fixtures.py` in Task 3.

### Conventions established in earlier sessions

- ASCII only in any string that flows to a Windows console. No smart quotes, em dashes, or fancy arrows.
- Pure-core / impure-shell: utilities export pure-ish functions; injectable seams use late binding (`runner=None`, `factory=None`, resolved at call time).
- DI seam in Stages: `ctx.config.settings.get("test", {}).get("comfy_client")`. Production path defaults to live impl; tests inject Fake. Mirrors Session 4's `claude_recorder` pattern.
- `pyproject.toml` already has `[tool.pytest.ini_options] asyncio_mode = "auto"`. Plain `def test_x():` for sync tests; `async def test_x():` for async.
- Dataclasses default to `frozen=True, slots=True` for value types.
- ruff config in `pyproject.toml`: line-length 100, target py311, rules E/F/W/I/B/UP. Run `ruff check src tests` before each commit.
- `data/stories/<id>/` directories are gitignored; per-story workspace is regeneratable.

### What you are NOT building this session

- Real ComfyUI HTTP traffic exercised against a live server -- Session 6.1 smoke run.
- Real LAION-Aesthetics implementation (SSH+script over to vast.ai) -- Session 6.1.
- IP-Adapter FaceID or ControlNet Depth wiring -- Session 6.2 or piggyback on Session 8.
- vast.ai provisioning -- Session 6.1.
- Per-track `image_model: flux_pro_api` support -- the flag is parsed but `flux_pro_api` raises `NotImplementedError`.
- Concurrent candidate generation per scene -- sequential is fine; one GPU serialises Flux calls anyway.

### Library notes that will save debugging time

- **httpx `MockTransport`** -- pass a request handler `def handler(request: httpx.Request) -> httpx.Response`; assert request URL/method/body in the handler; return canned responses. Tests build the client with `transport=httpx.MockTransport(handler)`.
- **ComfyUI HTTP wire shape** (verified against gold's client):
  - `POST /prompt` body `{"prompt": <workflow_dict>, "client_id": "<uuid>"}` -> `{"prompt_id": "..."}`
  - `GET /history/{prompt_id}` -> `{prompt_id: {"status": {"completed": bool, "status_str": str}, "outputs": {<node_id>: {"images": [{"filename": str, "subfolder": str, "type": str}, ...]}}}}`
  - `GET /view?filename=...&subfolder=...&type=output` -> raw image bytes (200) or 404
  - `POST /upload/image` multipart form with `image` file part -> `{"name": "<server_filename>"}`
  - `GET /system_stats` -> 200 JSON if healthy
- **Pillow tiny PNG** -- `from PIL import Image; Image.new("RGB", (64, 64), color=(128, 128, 128)).save(path)` writes a ~250 byte PNG. Checkerboard with `numpy` then `Image.fromarray(arr)`.
- **`hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()`** is the canonical workflow-signature hash. Use it for `FakeComfyClient` response keying.
- **Late binding for the `Hands` factory:** `mp_hands_factory: Callable[[], Any] | None`. The test factory builds and returns a MagicMock; `check_hand_anomalies` calls `factory()` once and uses the returned object.
- **`ctx.config.config_dir`** points to `<repo_root>/config/`. To find the workflows dir, do `ctx.config.config_dir / "workflows"`.
- **`asyncio.gather`** is NOT used in this session. Sequential candidate generation is the Spec choice.

---

## Task 1: `MappedFakeScorer` + `RemoteAestheticScorer` stub in `utils/aesthetics.py`

**Files:**
- Modify: `src/platinum/utils/aesthetics.py`
- Modify: `tests/unit/test_aesthetics.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_aesthetics.py`:

```python
import pytest


async def test_mapped_fake_scorer_returns_mapped_value(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import MappedFakeScorer
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    img_a.write_bytes(b"")
    img_b.write_bytes(b"")
    scorer = MappedFakeScorer(scores_by_path={img_a: 7.5, img_b: 3.0}, default=0.0)
    assert await scorer.score(img_a) == 7.5
    assert await scorer.score(img_b) == 3.0


async def test_mapped_fake_scorer_returns_default_for_unmapped(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import MappedFakeScorer
    unknown = tmp_path / "unknown.png"
    unknown.write_bytes(b"")
    scorer = MappedFakeScorer(scores_by_path={}, default=4.2)
    assert await scorer.score(unknown) == 4.2


def test_remote_aesthetic_scorer_init_raises_with_session_pointer() -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer
    with pytest.raises(NotImplementedError) as exc:
        RemoteAestheticScorer(host="example.com", ssh_user="root", ssh_key_path=None)
    assert "Session 6.1" in str(exc.value)
```

### Step 2: Run, see fail

```
pytest tests/unit/test_aesthetics.py -v
```
Expected: 3 ImportError / AttributeError -- `MappedFakeScorer` and `RemoteAestheticScorer` not defined.

### Step 3: Implement

Append to `src/platinum/utils/aesthetics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


# ... existing AestheticScorer Protocol and FakeAestheticScorer above ...


@dataclass(frozen=True, slots=True)
class MappedFakeScorer:
    """Test scorer: returns score per image_path; falls back to default for unmapped paths.

    Used by tests that need different scores for different candidates within the same
    test (e.g., to drive selection logic). The plain FakeAestheticScorer with a single
    fixed_score cannot do this.
    """

    scores_by_path: dict[Path, float] = field(default_factory=dict)
    default: float = 0.0

    async def score(self, image_path: Path) -> float:
        return self.scores_by_path.get(image_path, self.default)


class RemoteAestheticScorer:
    """LAION-Aesthetics v2 scorer that runs on the vast.ai box via SSH+script.

    Implementation deferred to Session 6.1. Construction is wired (so the
    production code path in KeyframeGeneratorStage can reference this class
    by name) but raises NotImplementedError so tests / dry-runs fail loudly
    with a clear pointer rather than silently returning zeros.
    """

    def __init__(
        self,
        *,
        host: str,
        ssh_user: str = "root",
        ssh_key_path: Path | None = None,
    ) -> None:
        raise NotImplementedError(
            "Session 6.1: implement SSH+script LAION scorer. "
            "Until then, inject FakeAestheticScorer / MappedFakeScorer in tests."
        )

    async def score(self, image_path: Path) -> float:
        raise NotImplementedError
```

### Step 4: Run, see pass

```
pytest tests/unit/test_aesthetics.py -v
```
Expected: 6 PASSED total (3 existing + 3 new).

### Step 5: Commit

```
git add src/platinum/utils/aesthetics.py tests/unit/test_aesthetics.py
git commit -m "feat(aesthetics): MappedFakeScorer + RemoteAestheticScorer stub (S6.1 deferral)"
```

---

## Task 2: Synthetic PNG fixture helper + commit three fixture PNGs

**Files:**
- Modify: `tests/_fixtures.py` -- append `make_synthetic_png` helper.
- Create: `tests/fixtures/keyframes/candidate_0.png`, `candidate_1.png`, `candidate_2.png`.
- Create: `tests/unit/test_fixture_helpers.py` (extend if exists).

### Step 1: Write the failing test

Append to `tests/unit/test_fixture_helpers.py`:

```python
def test_make_synthetic_png_writes_solid_grey(tmp_path: Path) -> None:
    from PIL import Image

    from tests._fixtures import make_synthetic_png

    out = tmp_path / "grey.png"
    make_synthetic_png(out, kind="grey", value=128, size=(64, 64))
    assert out.exists()
    img = Image.open(out)
    assert img.size == (64, 64)
    assert img.mode == "RGB"
    px = img.getpixel((10, 10))
    assert px == (128, 128, 128)


def test_make_synthetic_png_writes_checkerboard(tmp_path: Path) -> None:
    from PIL import Image

    from tests._fixtures import make_synthetic_png

    out = tmp_path / "checker.png"
    make_synthetic_png(out, kind="checkerboard", size=(64, 64))
    assert out.exists()
    img = Image.open(out)
    assert img.size == (64, 64)
    # Distinct pixel values present (not single-colour)
    pixels = {img.getpixel((x, y)) for x in (0, 16, 32, 48) for y in (0, 16, 32, 48)}
    assert len(pixels) >= 2
```

### Step 2: Run, see fail

```
pytest tests/unit/test_fixture_helpers.py -k synthetic_png -v
```
Expected: ImportError -- `make_synthetic_png` not defined.

### Step 3: Implement

Append to `tests/_fixtures.py`:

```python
from typing import Literal

from PIL import Image


def make_synthetic_png(
    path: Path,
    *,
    kind: Literal["grey", "checkerboard"] = "grey",
    value: int = 128,
    size: tuple[int, int] = (64, 64),
    block: int = 8,
) -> None:
    """Write a small synthetic PNG to `path`.

    kind="grey": solid RGB at (value, value, value).
    kind="checkerboard": black/white blocks of `block` pixels.
    """
    w, h = size
    if kind == "grey":
        Image.new("RGB", size, color=(value, value, value)).save(path)
        return
    if kind == "checkerboard":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, block):
            for x in range(0, w, block):
                if ((x // block) + (y // block)) % 2 == 0:
                    arr[y : y + block, x : x + block] = 255
        Image.fromarray(arr, "RGB").save(path)
        return
    raise ValueError(f"unknown kind: {kind!r}")
```

### Step 4: Run, see pass

```
pytest tests/unit/test_fixture_helpers.py -v
```
Expected: 6 PASSED (4 existing from S5 + 2 new).

### Step 5: Generate the three permanent fixture PNGs

```
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'tests')
from _fixtures import make_synthetic_png
out = Path('tests/fixtures/keyframes')
out.mkdir(parents=True, exist_ok=True)
make_synthetic_png(out / 'candidate_0.png', kind='grey', value=128)
make_synthetic_png(out / 'candidate_1.png', kind='checkerboard')
make_synthetic_png(out / 'candidate_2.png', kind='grey', value=64)
print('wrote 3 fixtures')
"
```

Verify with `ls -la tests/fixtures/keyframes/`. Each PNG should be ~250-1000 bytes.

### Step 6: Commit

```
git add tests/_fixtures.py tests/unit/test_fixture_helpers.py tests/fixtures/keyframes/
git commit -m "test(fixtures): make_synthetic_png helper + 3 keyframe candidate PNGs"
```

---

## Task 3: Hoist `_make_fake_hands_factory` into `tests/_fixtures.py`

**Files:**
- Modify: `tests/_fixtures.py` -- add the helper.
- Modify: `tests/unit/test_validate.py` -- import from `tests._fixtures` instead of defining locally.

### Step 1: Confirm tests currently pass

```
pytest tests/unit/test_validate.py -k hand -v
```
Expected: 4 PASSED.

### Step 2: Move the helper

Cut the `_make_fake_hands_factory` function from `tests/unit/test_validate.py` (currently around line 183-204). Paste it into `tests/_fixtures.py` below `make_synthetic_png`, renamed (drop the leading underscore -- this is a public test helper now):

```python
def make_fake_hands_factory(landmarks_per_hand: list[int] | None) -> Callable[[], Any]:
    """Return a factory that yields a Hands-stub returning configured results.

    Used by tests that exercise check_hand_anomalies (and the keyframe
    generator's anatomy gate) without mediapipe. Pass None to simulate
    "no hands detected"; pass [21, 21] for two valid hands; pass [21, 22]
    to simulate an anomaly.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    def factory() -> Any:
        instance = MagicMock()
        if landmarks_per_hand is None:
            result = SimpleNamespace(multi_hand_landmarks=None)
        else:
            hands = []
            for n in landmarks_per_hand:
                lm = SimpleNamespace(
                    landmark=[SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(n)]
                )
                hands.append(lm)
            result = SimpleNamespace(multi_hand_landmarks=hands)
        instance.process.return_value = result
        instance.close.return_value = None
        return instance

    return factory
```

In `tests/unit/test_validate.py`, replace the local helper definition with:

```python
from tests._fixtures import make_fake_hands_factory
```

And update each call site: `_make_fake_hands_factory(...)` -> `make_fake_hands_factory(...)`.

### Step 3: Run

```
pytest tests/unit/test_validate.py -k hand -v
```
Expected: 4 PASSED, no behaviour change.

```
pytest -q
```
Expected: full suite green; ~190 tests.

### Step 4: Commit

```
git add tests/_fixtures.py tests/unit/test_validate.py
git commit -m "refactor(tests): hoist make_fake_hands_factory into shared _fixtures.py"
```

---

## Task 4: `utils/workflow.py` -- `inject` (pure mutation) + `load_workflow`

**Files:**
- Create: `src/platinum/utils/workflow.py`
- Create: `tests/unit/test_workflow.py`

### Step 1: Write the failing tests

Create `tests/unit/test_workflow.py`:

```python
"""Tests for utils/workflow.py."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest


def _minimal_workflow() -> dict:
    return {
        "_meta": {
            "title": "minimal",
            "role": {
                "positive_prompt": "3",
                "negative_prompt": "4",
                "empty_latent": "5",
                "sampler": "6",
                "save_image": "8",
            },
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "OLD_POS", "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "OLD_NEG", "clip": ["2", 0]}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0, "steps": 20, "cfg": 3.5,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "OLD", "images": ["7", 0]}},
    }


def test_inject_swaps_tagged_node_inputs() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    out = inject(
        wf, prompt="cinematic dread", negative_prompt="cartoon",
        seed=4242, width=1024, height=1024, output_prefix="scene_001",
    )
    assert out["3"]["inputs"]["text"] == "cinematic dread"
    assert out["4"]["inputs"]["text"] == "cartoon"
    assert out["5"]["inputs"]["width"] == 1024
    assert out["5"]["inputs"]["height"] == 1024
    assert out["6"]["inputs"]["seed"] == 4242
    assert out["8"]["inputs"]["filename_prefix"] == "scene_001"


def test_inject_does_not_mutate_input() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    snapshot = copy.deepcopy(wf)
    _ = inject(
        wf, prompt="x", negative_prompt="y", seed=1,
        width=512, height=512, output_prefix="z",
    )
    assert wf == snapshot


def test_inject_leaves_untagged_nodes_alone() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    wf["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}}
    wf["2"] = {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors"}}
    out = inject(
        wf, prompt="a", negative_prompt="b", seed=7,
        width=1024, height=1024, output_prefix="p",
    )
    assert out["1"] == wf["1"]
    assert out["2"] == wf["2"]


def test_inject_raises_keyerror_on_missing_role() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    del wf["_meta"]["role"]["sampler"]
    with pytest.raises(KeyError) as exc:
        inject(
            wf, prompt="x", negative_prompt="y", seed=1,
            width=512, height=512, output_prefix="z",
        )
    assert "sampler" in str(exc.value)


def test_load_workflow_reads_named_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_workflow takes (config_dir, name) -- config_dir injected for testability."""
    from platinum.utils.workflow import load_workflow

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "demo.json").write_text(
        '{"_meta": {"role": {}}, "1": {"class_type": "X", "inputs": {}}}',
        encoding="utf-8",
    )
    out = load_workflow("demo", config_dir=tmp_path)
    assert out["1"]["class_type"] == "X"


def test_load_workflow_raises_on_missing_file(tmp_path: Path) -> None:
    from platinum.utils.workflow import load_workflow

    (tmp_path / "workflows").mkdir()
    with pytest.raises(FileNotFoundError):
        load_workflow("nope", config_dir=tmp_path)
```

### Step 2: Run, see fail

```
pytest tests/unit/test_workflow.py -v
```
Expected: ImportError -- module not present.

### Step 3: Implement

Create `src/platinum/utils/workflow.py`:

```python
"""ComfyUI workflow JSON loader + injector.

Workflows are static JSON files under config/workflows/. Each file has a
top-level `_meta.role` block mapping role names ("positive_prompt",
"sampler", ...) to node IDs, so `inject` can mutate the right node-input
fields without depending on node-id numbering staying stable.

`inject` is a pure function: deepcopy the input, mutate the copy, return.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

REQUIRED_ROLES = (
    "positive_prompt",
    "negative_prompt",
    "empty_latent",
    "sampler",
    "save_image",
)


def load_workflow(name: str, *, config_dir: Path) -> dict[str, Any]:
    """Load `<config_dir>/workflows/<name>.json`.

    Raises FileNotFoundError if the named file is missing.
    """
    path = Path(config_dir) / "workflows" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_role(workflow: dict[str, Any], role: str) -> str:
    roles = workflow.get("_meta", {}).get("role", {})
    if role not in roles:
        raise KeyError(f"workflow _meta.role missing required role: {role!r}")
    return roles[role]


def inject(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    output_prefix: str = "flux_dev",
) -> dict[str, Any]:
    """Return a new workflow dict with the variable fields swapped in.

    Required _meta.role entries: positive_prompt, negative_prompt,
    empty_latent, sampler, save_image. Raises KeyError if any are missing.
    """
    out = copy.deepcopy(workflow)
    pos_id = _resolve_role(out, "positive_prompt")
    neg_id = _resolve_role(out, "negative_prompt")
    latent_id = _resolve_role(out, "empty_latent")
    sampler_id = _resolve_role(out, "sampler")
    save_id = _resolve_role(out, "save_image")
    out[pos_id]["inputs"]["text"] = prompt
    out[neg_id]["inputs"]["text"] = negative_prompt
    out[latent_id]["inputs"]["width"] = width
    out[latent_id]["inputs"]["height"] = height
    out[sampler_id]["inputs"]["seed"] = seed
    out[save_id]["inputs"]["filename_prefix"] = output_prefix
    return out
```

### Step 4: Run, see pass

```
pytest tests/unit/test_workflow.py -v
```
Expected: 6 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/workflow.py tests/unit/test_workflow.py
git commit -m "feat(workflow): load_workflow + inject (deepcopy + tagged-node mutation)"
```

---

## Task 5: `config/workflows/flux_dev_keyframe.json` + integration test against `load_workflow`

**Files:**
- Create: `config/workflows/flux_dev_keyframe.json`
- Modify: `tests/unit/test_workflow.py`

### Step 1: Write the failing test

Append to `tests/unit/test_workflow.py`:

```python
def test_flux_dev_keyframe_workflow_loads_and_injects() -> None:
    """The shipped flux_dev_keyframe.json must round-trip through inject without errors."""
    from platinum.utils.workflow import inject, load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    # All required roles must be present and resolve to actual node ids.
    roles = wf["_meta"]["role"]
    assert {"positive_prompt", "negative_prompt", "empty_latent", "sampler", "save_image"} <= set(
        roles.keys()
    )
    for role_name, node_id in roles.items():
        assert node_id in wf, f"role {role_name!r} points at missing node {node_id!r}"
    out = inject(
        wf,
        prompt="a candle in a dark hallway",
        negative_prompt="bright daylight, neon",
        seed=12345,
        width=1024, height=1024,
        output_prefix="scene_001_candidate_0",
    )
    pos_id = roles["positive_prompt"]
    neg_id = roles["negative_prompt"]
    latent_id = roles["empty_latent"]
    sampler_id = roles["sampler"]
    save_id = roles["save_image"]
    assert out[pos_id]["inputs"]["text"] == "a candle in a dark hallway"
    assert out[neg_id]["inputs"]["text"] == "bright daylight, neon"
    assert out[latent_id]["inputs"]["width"] == 1024
    assert out[latent_id]["inputs"]["height"] == 1024
    assert out[sampler_id]["inputs"]["seed"] == 12345
    assert out[save_id]["inputs"]["filename_prefix"] == "scene_001_candidate_0"
```

### Step 2: Run, see fail

```
pytest tests/unit/test_workflow.py -k flux_dev -v
```
Expected: FileNotFoundError -- workflow file does not exist.

### Step 3: Create the workflow JSON

Create `config/workflows/flux_dev_keyframe.json`:

```json
{
  "_meta": {
    "title": "Flux.1 Dev keyframe baseline",
    "session": 6,
    "role": {
      "positive_prompt": "3",
      "negative_prompt": "4",
      "empty_latent": "5",
      "sampler": "6",
      "save_image": "8"
    }
  },
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "flux1-dev.safetensors",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "DualCLIPLoader",
    "inputs": {
      "clip_name1": "t5xxl_fp16.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux"
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "PLACEHOLDER_POSITIVE",
      "clip": ["2", 0]
    }
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "PLACEHOLDER_NEGATIVE",
      "clip": ["2", 0]
    }
  },
  "5": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    }
  },
  "6": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 0,
      "steps": 20,
      "cfg": 3.5,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0,
      "model": ["1", 0],
      "positive": ["3", 0],
      "negative": ["4", 0],
      "latent_image": ["5", 0]
    }
  },
  "7": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["6", 0],
      "vae": ["9", 0]
    }
  },
  "8": {
    "class_type": "SaveImage",
    "inputs": {
      "filename_prefix": "flux_dev",
      "images": ["7", 0]
    }
  },
  "9": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "ae.safetensors"
    }
  }
}
```

### Step 4: Run, see pass

```
pytest tests/unit/test_workflow.py -v
```
Expected: 7 PASSED.

### Step 5: Commit

```
git add config/workflows/flux_dev_keyframe.json tests/unit/test_workflow.py
git commit -m "feat(workflows): flux_dev_keyframe.json (Flux.1 Dev baseline, _meta.role-tagged)"
```

---

## Task 6: `ComfyClient` Protocol + `FakeComfyClient`

**Files:**
- Create: `src/platinum/utils/comfyui.py`
- Create: `tests/unit/test_comfyui.py`

### Step 1: Write the failing tests

Create `tests/unit/test_comfyui.py`:

```python
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
    """FakeComfyClient.calls is a public list of (workflow_signature, output_path) tuples."""
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
```

### Step 2: Run, see fail

```
pytest tests/unit/test_comfyui.py -v
```
Expected: ImportError -- module not present.

### Step 3: Implement (Protocol + FakeComfyClient only; HttpComfyClient lands in Task 7)

Create `src/platinum/utils/comfyui.py`:

```python
"""ComfyUI client.

Production: HttpComfyClient -- async httpx wrapper around ComfyUI's REST API.
Tests:      FakeComfyClient  -- deterministic, copies prebaked fixture PNGs to
            the requested output_path.

`generate_image` takes an *already-injected* workflow dict. Workflow JSON
schema knowledge lives in utils/workflow.py; this module only handles
transport (or fake transport).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


def workflow_signature(workflow: dict[str, Any]) -> str:
    """SHA256 of canonical JSON; used by FakeComfyClient response keying."""
    canonical = json.dumps(workflow, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@runtime_checkable
class ComfyClient(Protocol):
    """Talk to a ComfyUI server. Workflow injection is the caller's job."""

    async def generate_image(
        self,
        *,
        workflow: dict[str, Any],
        output_path: Path,
    ) -> Path: ...

    async def upload_image(self, image_path: Path) -> str: ...

    async def health_check(self) -> bool: ...


@dataclass
class FakeComfyClient:
    """Deterministic ComfyClient for tests.

    `responses` maps a workflow signature to a list of fixture PNG paths.
    Each generate_image call rotates through the list; once exhausted, the
    last entry is reused. Unknown signatures raise KeyError.

    `calls` records every generate_image call as a dict with
    {"workflow_signature": str, "output_path": Path} so tests can assert
    on call count, ordering, and the workflow each candidate received.
    """

    responses: dict[str, list[Path]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursors: dict[str, int] = field(default_factory=dict)

    async def generate_image(
        self,
        *,
        workflow: dict[str, Any],
        output_path: Path,
    ) -> Path:
        sig = workflow_signature(workflow)
        if sig not in self.responses:
            raise KeyError(
                f"FakeComfyClient has no response configured for workflow signature {sig[:12]}..."
            )
        sources = self.responses[sig]
        if not sources:
            raise ValueError(f"FakeComfyClient.responses[{sig[:12]}...] is empty")
        cursor = self._cursors.get(sig, 0)
        src = sources[min(cursor, len(sources) - 1)]
        self._cursors[sig] = cursor + 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, output_path)
        self.calls.append({"workflow_signature": sig, "output_path": output_path})
        return output_path

    async def upload_image(self, image_path: Path) -> str:
        return Path(image_path).name

    async def health_check(self) -> bool:
        return True
```

### Step 4: Run, see pass

```
pytest tests/unit/test_comfyui.py -v
```
Expected: 8 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/comfyui.py tests/unit/test_comfyui.py
git commit -m "feat(comfyui): ComfyClient Protocol + FakeComfyClient (rotating fixture map)"
```

---

## Task 7: `HttpComfyClient` (httpx + MockTransport tests)

**Files:**
- Modify: `src/platinum/utils/comfyui.py` -- append `HttpComfyClient`.
- Modify: `tests/unit/test_comfyui.py` -- append HTTP-shape tests.

### Step 1: Write the failing tests

Append to `tests/unit/test_comfyui.py`:

```python
import httpx


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
                "8": {"images": [{"filename": "flux_dev_00001_.png", "subfolder": "", "type": "output"}]}
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
```

### Step 2: Run, see fail

```
pytest tests/unit/test_comfyui.py -k Http -v
```
Expected: AttributeError -- `HttpComfyClient` not defined.

### Step 3: Implement

Append to `src/platinum/utils/comfyui.py`:

```python
import asyncio
import logging
import uuid

import httpx

logger = logging.getLogger(__name__)


class HttpComfyClient:
    """Async httpx-based ComfyUI client.

    `transport` plumbs through to httpx.AsyncClient; tests pass MockTransport
    for unit-level wire-shape verification without a Fake.

    `poll_interval` defaults to 2.0s for production; tests pass 0.0 to get
    instant polling.
    """

    def __init__(
        self,
        host: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_polls: int = 600,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.transport = transport
        self.max_polls = max_polls

    def _client(self, *, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.host,
            timeout=timeout or self.timeout,
            transport=self.transport,
        )

    async def health_check(self) -> bool:
        try:
            async with self._client(timeout=10.0) as client:
                resp = await client.get("/system_stats")
                return resp.status_code == 200
        except Exception:
            return False

    async def upload_image(self, image_path: Path) -> str:
        path = Path(image_path)
        async with self._client(timeout=60.0) as client:
            with path.open("rb") as fh:
                resp = await client.post(
                    "/upload/image",
                    files={"image": (path.name, fh, "image/png")},
                    data={"overwrite": "true"},
                )
                resp.raise_for_status()
                payload = resp.json()
                return payload.get("name", path.name)

    async def generate_image(
        self,
        *,
        workflow: dict[str, Any],
        output_path: Path,
    ) -> Path:
        prompt_id = await self._submit(workflow)
        result = await self._poll(prompt_id)
        await self._download(result, output_path)
        return output_path

    async def _submit(self, workflow: dict[str, Any]) -> str:
        async with self._client(timeout=30.0) as client:
            payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
            resp = await client.post("/prompt", json=payload)
            resp.raise_for_status()
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"No prompt_id in /prompt response: {data!r}")
            logger.info("submitted ComfyUI workflow, prompt_id=%s", prompt_id)
            return prompt_id

    async def _poll(self, prompt_id: str) -> dict[str, Any]:
        async with self._client(timeout=30.0) as client:
            for attempt in range(self.max_polls):
                resp = await client.get(f"/history/{prompt_id}")
                if resp.status_code == 200:
                    body = resp.json()
                    if prompt_id in body:
                        result = body[prompt_id]
                        status = result.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI workflow error: {result!r}")
                        if status.get("completed"):
                            return result
                if self.poll_interval > 0:
                    await asyncio.sleep(self.poll_interval)
            raise RuntimeError(f"Timed out polling /history/{prompt_id} after {self.max_polls} attempts")

    async def _download(self, result: dict[str, Any], output_path: Path) -> None:
        outputs = result.get("outputs", {})
        for _node_id, node_output in outputs.items():
            files = node_output.get("images") or node_output.get("gifs") or node_output.get("videos") or []
            if not files:
                continue
            file_info = files[0]
            params = {
                "filename": file_info.get("filename", ""),
                "subfolder": file_info.get("subfolder", ""),
                "type": file_info.get("type", "output"),
            }
            async with self._client() as client:
                resp = await client.get("/view", params=params)
                resp.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(resp.content)
                logger.info("downloaded ComfyUI output to %s (%d bytes)", output_path, len(resp.content))
                return
        raise RuntimeError(f"No output files found in ComfyUI result. Keys: {list(outputs.keys())}")
```

### Step 4: Run, see pass

```
pytest tests/unit/test_comfyui.py -v
```
Expected: 14 PASSED (8 Fake + 6 Http).

### Step 5: Commit

```
git add src/platinum/utils/comfyui.py tests/unit/test_comfyui.py
git commit -m "feat(comfyui): HttpComfyClient with httpx MockTransport-friendly transport seam"
```

---

## Task 8: `KeyframeReport` + `KeyframeGenerationError` scaffold

**Files:**
- Create: `src/platinum/pipeline/keyframe_generator.py`
- Create: `tests/unit/test_keyframe_generator.py`

### Step 1: Write the failing test

Create `tests/unit/test_keyframe_generator.py`:

```python
"""Tests for pipeline/keyframe_generator.py."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_keyframe_report_is_frozen_and_carries_fields() -> None:
    from platinum.pipeline.keyframe_generator import KeyframeReport

    r = KeyframeReport(
        scene_index=0,
        candidates=[Path("a"), Path("b"), Path("c")],
        scores=[7.0, 5.0, 3.0],
        anatomy_passed=[True, True, False],
        selected_index=0,
        selected_via_fallback=False,
    )
    assert r.scene_index == 0
    assert r.selected_index == 0
    assert not r.selected_via_fallback
    with pytest.raises(Exception):
        r.scene_index = 1  # type: ignore[misc]


def test_keyframe_generation_error_carries_per_candidate_exceptions() -> None:
    from platinum.pipeline.keyframe_generator import KeyframeGenerationError

    excs: list[BaseException] = [RuntimeError("a"), ValueError("b"), TimeoutError("c")]
    err = KeyframeGenerationError(scene_index=4, exceptions=excs)
    assert err.scene_index == 4
    assert len(err.exceptions) == 3
    assert "scene 4" in str(err) or "scene_index=4" in str(err)
```

### Step 2: Run, see fail

```
pytest tests/unit/test_keyframe_generator.py -v
```
Expected: ImportError -- module not present.

### Step 3: Implement

Create `src/platinum/pipeline/keyframe_generator.py`:

```python
"""Keyframe generator pipeline -- per-scene Flux candidate generation + selection.

For each Scene:
  1. Generate N candidates (default 3) via ComfyClient at deterministic seeds.
  2. Score each via AestheticScorer; check_hand_anomalies for anatomy gate.
  3. Select the highest-scoring candidate that passes both gates;
     fall back to candidate 0 if none qualify.

Pure functions (`generate_for_scene`, `generate`) take all dependencies as
args; the impure `KeyframeGeneratorStage` pulls dependencies from `ctx`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KeyframeReport:
    """Per-scene generation report. Persisted onto Scene fields by the Stage."""

    scene_index: int
    candidates: list[Path]
    scores: list[float]
    anatomy_passed: list[bool]
    selected_index: int
    selected_via_fallback: bool


class KeyframeGenerationError(RuntimeError):
    """Raised when ALL candidates for a scene threw during generation.

    Carries the per-candidate exception list so the Stage can record a
    useful error string in the StageRun log.
    """

    def __init__(self, *, scene_index: int, exceptions: Sequence[BaseException]) -> None:
        self.scene_index = scene_index
        self.exceptions = list(exceptions)
        joined = "; ".join(f"{type(e).__name__}: {e}" for e in self.exceptions)
        super().__init__(f"all candidates failed for scene_index={scene_index}: {joined}")
```

### Step 4: Run, see pass

```
pytest tests/unit/test_keyframe_generator.py -v
```
Expected: 2 PASSED.

### Step 5: Commit

```
git add src/platinum/pipeline/keyframe_generator.py tests/unit/test_keyframe_generator.py
git commit -m "feat(keyframe_generator): KeyframeReport + KeyframeGenerationError scaffold"
```

---

## Task 9: `generate_for_scene` -- happy path + selection rules

**Files:**
- Modify: `src/platinum/pipeline/keyframe_generator.py`
- Modify: `tests/unit/test_keyframe_generator.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_keyframe_generator.py`:

```python
from platinum.models.story import Scene


def _scene(idx: int = 0, *, visual_prompt: str = "a candle") -> Scene:
    return Scene(
        id=f"scene_{idx:03d}",
        index=idx,
        narration_text="Once upon a time...",
        narration_duration_seconds=5.0,
        visual_prompt=visual_prompt,
        negative_prompt="bright daylight",
    )


_TRACK_VISUAL = {
    "aesthetic": "cinematic dark, candlelight",
    "negative_prompt": "bright daylight, neon",
}
_GATES = {"aesthetic_min_score": 6.0}


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"


def _build_fake_comfy_with_three_candidates() -> "FakeComfyClient":
    """Configure a FakeComfyClient that returns the 3 fixture PNGs in order,
    regardless of the workflow signature seen (test does not care which seed
    maps to which image, only that each candidate gets a distinct PNG)."""
    from platinum.utils.comfyui import FakeComfyClient

    fixtures = _fixture_dir()
    a = fixtures / "candidate_0.png"
    b = fixtures / "candidate_1.png"
    c = fixtures / "candidate_2.png"
    # Use a Sentinel signature that the implementation uses for ALL workflows in this test:
    # we'll subclass FakeComfyClient or use a wildcard. Simpler: have the implementation
    # rotate through a fallback list for unknown signatures.
    # ALTERNATIVE: build a real workflow per seed, hash them, and put each into responses.
    # This is what production-shaped tests should do, so let's do that:
    from platinum.utils.comfyui import workflow_signature
    from platinum.utils.workflow import inject

    repo_root = Path(__file__).resolve().parents[2]
    from platinum.utils.workflow import load_workflow

    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    seeds = (0, 1, 2)
    responses: dict[str, list[Path]] = {}
    for seed, fixture in zip(seeds, (a, b, c), strict=True):
        wf = inject(
            wf_template,
            prompt="cinematic dark, candlelight a candle",
            negative_prompt="bright daylight, neon",
            seed=seed,
            width=1024, height=1024,
            output_prefix="scene_000",
        )
        responses[workflow_signature(wf)] = [fixture]
    return FakeComfyClient(responses=responses)


async def test_generate_for_scene_happy_path_picks_highest_scoring(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    fixtures = _fixture_dir()
    # Score candidate_1 highest, candidate_0 second, candidate_2 third.
    score_map = {
        output_dir / "candidate_0.png": 6.5,
        output_dir / "candidate_1.png": 8.0,
        output_dir / "candidate_2.png": 7.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.scene_index == 0
    assert len(report.candidates) == 3
    assert report.scores == [6.5, 8.0, 7.0]
    assert report.anatomy_passed == [True, True, True]
    assert report.selected_index == 1   # candidate_1 had highest score
    assert not report.selected_via_fallback


async def test_generate_for_scene_ties_selected_lowest_index(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 7.5,
        output_dir / "candidate_1.png": 7.5,
        output_dir / "candidate_2.png": 6.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.selected_index == 0   # lowest-index tie wins


async def test_generate_for_scene_below_threshold_falls_back(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 4.0,
        output_dir / "candidate_1.png": 5.0,
        output_dir / "candidate_2.png": 5.5,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.selected_index == 0
    assert report.selected_via_fallback is True


async def test_generate_for_scene_anatomy_rejects_high_score_candidate(tmp_path: Path) -> None:
    """High-score candidate fails anatomy; second-highest wins."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 6.0,
        output_dir / "candidate_1.png": 9.0,   # highest, but anatomy will reject
        output_dir / "candidate_2.png": 7.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = _build_fake_comfy_with_three_candidates()

    # A factory that returns "two valid hands" for cand 0 and 2 but "anomaly" for cand 1.
    # Implement by making the factory itself stateful via a counter.
    call_idx = {"n": 0}

    def factory():
        from tests._fixtures import make_fake_hands_factory

        n = call_idx["n"]
        call_idx["n"] += 1
        if n == 1:
            return make_fake_hands_factory([21, 22])()  # anomaly
        return make_fake_hands_factory(None)()  # no hands -> passes

    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        seeds=(0, 1, 2),
        mp_hands_factory=factory,
    )
    assert report.anatomy_passed == [True, False, True]
    assert report.selected_index == 2   # cand 2 is the highest passing


async def test_generate_for_scene_missing_visual_prompt_raises(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0, visual_prompt=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError) as exc:
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates=_GATES,
            comfy=FakeComfyClient(responses={}),
            scorer=FakeAestheticScorer(fixed_score=8.0),
            output_dir=tmp_path / "scene_000",
            seeds=(0, 1, 2),
            mp_hands_factory=make_fake_hands_factory(None),
        )
    assert "visual_prompt" in str(exc.value)
```

### Step 2: Run, see fail

```
pytest tests/unit/test_keyframe_generator.py -v
```
Expected: AttributeError -- `generate_for_scene` not defined.

### Step 3: Implement

Append to `src/platinum/pipeline/keyframe_generator.py`:

```python
from collections.abc import Callable, Sequence
from typing import Any

from platinum.models.story import Scene
from platinum.utils.aesthetics import AestheticScorer
from platinum.utils.comfyui import ComfyClient
from platinum.utils.validate import check_hand_anomalies
from platinum.utils.workflow import inject, load_workflow


def _seeds_for_scene(scene_index: int, n: int) -> tuple[int, ...]:
    """Deterministic seeds: scene_index*1000 + offset."""
    return tuple(scene_index * 1000 + i for i in range(n))


async def generate_for_scene(
    scene: Scene,
    *,
    track_visual: dict[str, Any],
    quality_gates: dict[str, Any],
    comfy: ComfyClient,
    scorer: AestheticScorer,
    output_dir: Path,
    workflow_template: dict[str, Any] | None = None,
    config_dir: Path | None = None,
    n_candidates: int = 3,
    seeds: Sequence[int] | None = None,
    width: int = 1024,
    height: int = 1024,
    mp_hands_factory: Callable[[], Any] | None = None,
) -> KeyframeReport:
    """Generate N candidates, score + anatomy-check each, return a KeyframeReport.

    `workflow_template` is loaded from `config_dir/workflows/flux_dev_keyframe.json`
    if None. `seeds` defaults to `_seeds_for_scene(scene.index, n_candidates)`.
    """
    if not scene.visual_prompt:
        raise ValueError(
            f"scene {scene.index} has no visual_prompt; run `platinum adapt` first"
        )
    if workflow_template is None:
        if config_dir is None:
            raise ValueError("either workflow_template or config_dir is required")
        workflow_template = load_workflow("flux_dev_keyframe", config_dir=config_dir)
    use_seeds = tuple(seeds) if seeds is not None else _seeds_for_scene(scene.index, n_candidates)
    if len(use_seeds) != n_candidates:
        raise ValueError(f"seeds length {len(use_seeds)} != n_candidates {n_candidates}")

    output_dir.mkdir(parents=True, exist_ok=True)

    aesthetic_text = " ".join(
        s for s in (track_visual.get("aesthetic"), scene.visual_prompt) if s
    )
    negative_text = scene.negative_prompt or track_visual.get("negative_prompt", "")

    candidate_paths: list[Path] = []
    candidate_exceptions: list[BaseException | None] = []
    for i, seed in enumerate(use_seeds):
        wf = inject(
            workflow_template,
            prompt=aesthetic_text,
            negative_prompt=negative_text,
            seed=seed,
            width=width,
            height=height,
            output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
        )
        path = output_dir / f"candidate_{i}.png"
        try:
            await comfy.generate_image(workflow=wf, output_path=path)
            candidate_paths.append(path)
            candidate_exceptions.append(None)
        except Exception as exc:  # noqa: BLE001 -- per-candidate isolation is intentional
            logger.warning(
                "scene %d candidate %d generate_image failed: %r", scene.index, i, exc
            )
            candidate_paths.append(path)
            candidate_exceptions.append(exc)

    if all(e is not None for e in candidate_exceptions):
        raise KeyframeGenerationError(
            scene_index=scene.index,
            exceptions=[e for e in candidate_exceptions if e is not None],
        )

    scores: list[float] = []
    anatomy_passed: list[bool] = []
    for path, exc in zip(candidate_paths, candidate_exceptions, strict=True):
        if exc is not None or not path.exists():
            scores.append(0.0)
            anatomy_passed.append(False)
            continue
        try:
            score = await scorer.score(path)
        except Exception as scorer_exc:  # noqa: BLE001
            logger.warning("scorer failed for %s: %r", path, scorer_exc)
            score = 0.0
        if not _is_finite(score):
            score = 0.0
        scores.append(float(score))
        result = check_hand_anomalies(path, mp_hands_factory=mp_hands_factory)
        anatomy_passed.append(result.passed)

    threshold = float(quality_gates.get("aesthetic_min_score", 0.0))
    eligible = [
        i for i, (s, a) in enumerate(zip(scores, anatomy_passed, strict=True))
        if s >= threshold and a
    ]
    if eligible:
        # highest score; ties -> lowest index
        selected_index = max(eligible, key=lambda i: (scores[i], -i))
        # max() with key (score, -i) selects highest score; among ties on score,
        # the smaller -i (i.e. larger i) wins. We want LOWEST index on ties,
        # so flip:
        max_score = max(scores[i] for i in eligible)
        selected_index = next(i for i in eligible if scores[i] == max_score)
        selected_via_fallback = False
    else:
        selected_index = 0
        selected_via_fallback = True

    return KeyframeReport(
        scene_index=scene.index,
        candidates=candidate_paths,
        scores=scores,
        anatomy_passed=anatomy_passed,
        selected_index=selected_index,
        selected_via_fallback=selected_via_fallback,
    )


def _is_finite(x: float) -> bool:
    import math as _math

    try:
        return _math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
```

### Step 4: Run, see pass

```
pytest tests/unit/test_keyframe_generator.py -v
```
Expected: 7 PASSED.

### Step 5: Commit

```
git add src/platinum/pipeline/keyframe_generator.py tests/unit/test_keyframe_generator.py
git commit -m "feat(keyframe_generator): generate_for_scene happy path + selection rules"
```

---

## Task 10: `generate_for_scene` -- failure isolation + all-fail error

**Files:**
- Modify: `tests/unit/test_keyframe_generator.py`
- (Implementation already in place from Task 9; this task only verifies behaviour with new tests.)

### Step 1: Write the failing tests

Append to `tests/unit/test_keyframe_generator.py`:

```python
async def test_generate_for_scene_isolates_per_candidate_exception(tmp_path: Path) -> None:
    """One candidate throws; other two succeed; selection picks among survivors."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    fixtures = _fixture_dir()
    output_dir = tmp_path / "scene_000"

    # Configure responses for seeds 0 and 2 only; seed 1 will raise KeyError.
    responses: dict[str, list[Path]] = {}
    for seed, fixture in [(0, fixtures / "candidate_0.png"), (2, fixtures / "candidate_2.png")]:
        wf = inject(
            wf_template,
            prompt="cinematic dark, candlelight a candle",
            negative_prompt="bright daylight, neon",
            seed=seed, width=1024, height=1024,
            output_prefix=f"scene_000_candidate_{seed}",
        )
        responses[workflow_signature(wf)] = [fixture]
    comfy = FakeComfyClient(responses=responses)

    scene = _scene(idx=0)
    score_map = {
        output_dir / "candidate_0.png": 6.5,
        output_dir / "candidate_2.png": 8.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.scores[1] == 0.0          # the failed candidate
    assert report.anatomy_passed[1] is False
    assert report.selected_index == 2       # candidate_2 has highest passing score


async def test_generate_for_scene_all_fail_raises_keyframe_generation_error(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        generate_for_scene,
    )
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    comfy = FakeComfyClient(responses={})  # no responses -> KeyError on every call
    with pytest.raises(KeyframeGenerationError) as exc:
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates=_GATES,
            comfy=comfy,
            scorer=FakeAestheticScorer(fixed_score=8.0),
            output_dir=tmp_path / "scene_000",
            seeds=(0, 1, 2),
            mp_hands_factory=make_fake_hands_factory(None),
        )
    assert exc.value.scene_index == 0
    assert len(exc.value.exceptions) == 3


async def test_generate_for_scene_seeds_default_to_index_offset(tmp_path: Path) -> None:
    """When seeds=None, _seeds_for_scene(scene.index, n) is used."""
    from platinum.pipeline.keyframe_generator import _seeds_for_scene

    assert _seeds_for_scene(0, 3) == (0, 1, 2)
    assert _seeds_for_scene(7, 3) == (7000, 7001, 7002)
    assert _seeds_for_scene(12, 4) == (12000, 12001, 12002, 12003)
```

### Step 2: Run, see pass (impl from Task 9 already supports this)

```
pytest tests/unit/test_keyframe_generator.py -v
```
Expected: 10 PASSED.

### Step 3: Commit

```
git add tests/unit/test_keyframe_generator.py
git commit -m "test(keyframe_generator): per-candidate isolation + all-fail + seed defaults"
```

---

## Task 11: `generate` -- story-level loop

**Files:**
- Modify: `src/platinum/pipeline/keyframe_generator.py`
- Modify: `tests/unit/test_keyframe_generator.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_keyframe_generator.py`:

```python
def _build_story_with_n_scenes(n: int) -> "Story":
    from datetime import datetime

    from platinum.models.story import Scene, Source, Story

    scenes = [
        Scene(
            id=f"scene_{i:03d}",
            index=i,
            narration_text=f"Scene {i} narration.",
            narration_duration_seconds=5.0,
            visual_prompt=f"a candle in dark hallway scene {i}",
            negative_prompt="bright daylight",
        )
        for i in range(n)
    ]
    return Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="http://example.test",
            title="Test Story",
            author="Test",
            raw_text="x",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
        scenes=scenes,
    )


async def test_generate_iterates_all_scenes(tmp_path: Path) -> None:
    """generate() returns one KeyframeReport per scene."""
    from platinum.pipeline.keyframe_generator import generate
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    fixtures = _fixture_dir()

    n_scenes = 2
    story = _build_story_with_n_scenes(n_scenes)

    responses: dict[str, list[Path]] = {}
    for scene in story.scenes:
        for i, seed in enumerate((scene.index * 1000, scene.index * 1000 + 1, scene.index * 1000 + 2)):
            wf = inject(
                wf_template,
                prompt=f"cinematic dark, candlelight a candle in dark hallway scene {scene.index}",
                negative_prompt="bright daylight",
                seed=seed, width=1024, height=1024,
                output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
            )
            responses[workflow_signature(wf)] = [fixtures / f"candidate_{i}.png"]
    comfy = FakeComfyClient(responses=responses)

    # Stub Config.track via a minimal duck-typed object.
    from types import SimpleNamespace

    track_dict = {
        "visual": _TRACK_VISUAL,
        "quality_gates": _GATES,
    }

    class _Cfg:
        config_dir = repo_root / "config"

        def track(self, _id: str) -> dict:
            return track_dict

    reports = await generate(
        story,
        config=_Cfg(),  # type: ignore[arg-type]
        comfy=comfy,
        scorer=FakeAestheticScorer(fixed_score=7.5),
        output_root=tmp_path,
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert len(reports) == n_scenes
    assert reports[0].scene_index == 0
    assert reports[1].scene_index == 1
    assert all(r.selected_via_fallback is False for r in reports)
```

### Step 2: Run, see fail

```
pytest tests/unit/test_keyframe_generator.py -k generate_iterates -v
```
Expected: AttributeError -- `generate` not defined.

### Step 3: Implement

Append to `src/platinum/pipeline/keyframe_generator.py`:

```python
from platinum.models.story import Story


async def generate(
    story: Story,
    *,
    config: Any,                    # platinum.config.Config (duck-typed for tests)
    comfy: ComfyClient,
    scorer: AestheticScorer,
    output_root: Path,
    mp_hands_factory: Callable[[], Any] | None = None,
) -> list[KeyframeReport]:
    """Run keyframe generation for every scene whose keyframe_path is None.

    Mutates each scene in-place: keyframe_candidates, keyframe_scores,
    keyframe_path, validation["keyframe_anatomy"],
    validation["keyframe_selected_via_fallback"].
    """
    track_cfg = config.track(story.track)
    track_visual = dict(track_cfg.get("visual", {}))
    quality_gates = dict(track_cfg.get("quality_gates", {}))
    workflow_template = load_workflow("flux_dev_keyframe", config_dir=config.config_dir)

    reports: list[KeyframeReport] = []
    for scene in story.scenes:
        if scene.keyframe_path is not None:
            logger.info(
                "scene %d already has keyframe_path=%s; skipping (resume)",
                scene.index, scene.keyframe_path,
            )
            continue
        scene_dir = output_root / f"scene_{scene.index:03d}"
        report = await generate_for_scene(
            scene,
            track_visual=track_visual,
            quality_gates=quality_gates,
            comfy=comfy,
            scorer=scorer,
            output_dir=scene_dir,
            workflow_template=workflow_template,
            mp_hands_factory=mp_hands_factory,
        )
        scene.keyframe_candidates = list(report.candidates)
        scene.keyframe_scores = list(report.scores)
        scene.keyframe_path = report.candidates[report.selected_index]
        scene.validation["keyframe_anatomy"] = list(report.anatomy_passed)
        scene.validation["keyframe_selected_via_fallback"] = report.selected_via_fallback
        reports.append(report)
        logger.info(
            "scene %d selected candidate %d (score=%.2f, fallback=%s)",
            scene.index, report.selected_index,
            report.scores[report.selected_index], report.selected_via_fallback,
        )
    return reports
```

### Step 4: Run, see pass

```
pytest tests/unit/test_keyframe_generator.py -v
```
Expected: 11 PASSED.

### Step 5: Commit

```
git add src/platinum/pipeline/keyframe_generator.py tests/unit/test_keyframe_generator.py
git commit -m "feat(keyframe_generator): generate (story-level loop with mid-run resume)"
```

---

## Task 12: `KeyframeGeneratorStage` -- impure shell + integration test

**Files:**
- Modify: `src/platinum/pipeline/keyframe_generator.py`
- Create: `tests/integration/test_keyframe_generator_stage.py`

### Step 1: Write the failing test

Create `tests/integration/test_keyframe_generator_stage.py`:

```python
"""Integration tests for KeyframeGeneratorStage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from platinum.models.story import Scene, Source, Story
from platinum.pipeline.context import PipelineContext


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"


def _build_story(n: int = 2) -> Story:
    scenes = [
        Scene(
            id=f"scene_{i:03d}",
            index=i,
            narration_text=f"Narration {i}.",
            narration_duration_seconds=5.0,
            visual_prompt=f"candle scene {i}",
            negative_prompt="bright daylight",
        )
        for i in range(n)
    ]
    return Story(
        id="story_int_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="http://example.test",
            title="Stage Test Story",
            author="Anon",
            raw_text="x",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
        scenes=scenes,
    )


def _build_responses_for_story(story: Story, repo_root: Path) -> dict[str, list[Path]]:
    from platinum.utils.comfyui import workflow_signature
    from platinum.utils.workflow import inject, load_workflow

    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    fixtures = _fixture_dir()
    out: dict[str, list[Path]] = {}
    for scene in story.scenes:
        for i, seed in enumerate(
            (scene.index * 1000, scene.index * 1000 + 1, scene.index * 1000 + 2)
        ):
            wf = inject(
                wf_template,
                prompt=f"cinematic dark, candlelight candle scene {scene.index}",
                negative_prompt="bright daylight",
                seed=seed, width=1024, height=1024,
                output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
            )
            out[workflow_signature(wf)] = [fixtures / f"candidate_{i}.png"]
    return out


async def test_keyframe_stage_runs_end_to_end(tmp_project, config) -> None:  # noqa: ANN001
    """Stage.run mutates each scene, persists keyframe_path, returns artifacts dict."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    repo_root = Path(__file__).resolve().parents[2]
    story = _build_story(n=2)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    scorer = FakeAestheticScorer(fixed_score=8.0)

    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": scorer,
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, db_path=tmp_project / "data" / "platinum.db")

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    assert all(s.keyframe_path is not None for s in story.scenes)
    assert all(len(s.keyframe_candidates) == 3 for s in story.scenes)
    assert all(len(s.keyframe_scores) == 3 for s in story.scenes)
    assert artifacts["scenes_total"] == 2
    assert artifacts["scenes_succeeded"] == 2
    assert artifacts["scenes_via_fallback"] == 0


async def test_keyframe_stage_persists_story_json_after_each_scene(tmp_project, config) -> None:  # noqa: ANN001
    """Per-scene checkpoint: story.json on disk reflects each scene as it lands."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    repo_root = Path(__file__).resolve().parents[2]
    story = _build_story(n=2)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story.save(story_dir / "story.json")

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, db_path=tmp_project / "data" / "platinum.db")

    stage = KeyframeGeneratorStage()
    await stage.run(story, ctx)

    # Reload from disk; assert scenes round-trip with keyframe_path set.
    reloaded = Story.load(story_dir / "story.json")
    assert all(s.keyframe_path is not None for s in reloaded.scenes)


async def test_keyframe_stage_resumes_when_scene_already_has_keyframe(tmp_project, config) -> None:  # noqa: ANN001
    """Pre-set scene 0's keyframe_path; Stage should skip it."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    repo_root = Path(__file__).resolve().parents[2]
    story = _build_story(n=2)
    pre_existing = tmp_project / "data" / "stories" / story.id / "keyframes" / "scene_000" / "candidate_0.png"
    pre_existing.parent.mkdir(parents=True, exist_ok=True)
    pre_existing.write_bytes((_fixture_dir() / "candidate_0.png").read_bytes())
    story.scenes[0].keyframe_path = pre_existing

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, db_path=tmp_project / "data" / "platinum.db")

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    assert story.scenes[0].keyframe_path == pre_existing  # unchanged
    assert story.scenes[1].keyframe_path is not None       # newly generated
    assert artifacts["scenes_total"] == 2
    assert artifacts["scenes_succeeded"] == 1               # only scene 1 ran


async def test_keyframe_stage_records_failure_in_artifacts(tmp_project, config) -> None:  # noqa: ANN001
    """All-fail scene -> artifacts['scenes_failed'] increments and Stage raises."""
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        KeyframeGeneratorStage,
    )
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    story = _build_story(n=1)
    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses={}),  # all candidates KeyError
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, db_path=tmp_project / "data" / "platinum.db")
    stage = KeyframeGeneratorStage()
    with pytest.raises(KeyframeGenerationError):
        await stage.run(story, ctx)
```

### Step 2: Run, see fail

```
pytest tests/integration/test_keyframe_generator_stage.py -v
```
Expected: AttributeError -- `KeyframeGeneratorStage` not defined.

### Step 3: Implement

Append to `src/platinum/pipeline/keyframe_generator.py`:

```python
from typing import ClassVar

from platinum.pipeline.stage import Stage


class KeyframeGeneratorStage(Stage):
    """Per-story keyframe generation stage. See module docstring."""

    name: ClassVar[str] = "keyframe_generator"

    async def run(self, story: Story, ctx: Any) -> dict[str, Any]:
        from platinum.utils.aesthetics import RemoteAestheticScorer
        from platinum.utils.comfyui import HttpComfyClient

        test_overrides = ctx.config.settings.get("test", {})
        comfy: ComfyClient = test_overrides.get("comfy_client") or HttpComfyClient(
            host=ctx.config.settings.get("comfyui", {}).get(
                "host", "http://localhost:8188"
            ),
        )
        scorer: AestheticScorer = test_overrides.get("aesthetic_scorer") or RemoteAestheticScorer(
            host=ctx.config.settings.get("aesthetics", {}).get("host", "")
        )
        mp_hands_factory = test_overrides.get("mp_hands_factory")

        story_dir = ctx.story_path(story).parent if hasattr(ctx, "story_path") else None
        output_root = (
            (story_dir / "keyframes")
            if story_dir is not None
            else Path("data/stories") / story.id / "keyframes"
        )

        scenes_total = len(story.scenes)
        scenes_succeeded_before = sum(1 for s in story.scenes if s.keyframe_path is not None)
        try:
            reports = await generate(
                story,
                config=ctx.config,
                comfy=comfy,
                scorer=scorer,
                output_root=output_root,
                mp_hands_factory=mp_hands_factory,
            )
        except KeyframeGenerationError:
            # Save what we have so far before raising.
            try:
                story.save(ctx.story_path(story))
            except Exception:  # noqa: BLE001 -- save best-effort on the failure path
                logger.exception("failed to save story.json after KeyframeGenerationError")
            raise

        # Atomic per-story save (the per-scene save inside `generate` is
        # left as a future enhancement; for now we save once at end of Stage).
        try:
            story.save(ctx.story_path(story))
        except Exception:
            logger.exception("failed to save story.json after stage completion")

        scenes_via_fallback = sum(1 for r in reports if r.selected_via_fallback)
        scenes_succeeded_now = sum(1 for s in story.scenes if s.keyframe_path is not None)
        return {
            "scenes_total": scenes_total,
            "scenes_succeeded": scenes_succeeded_now,
            "scenes_succeeded_before": scenes_succeeded_before,
            "scenes_via_fallback": scenes_via_fallback,
            "scenes_failed": scenes_total - scenes_succeeded_now,
            "reports": [
                {
                    "scene_index": r.scene_index,
                    "selected_index": r.selected_index,
                    "selected_score": r.scores[r.selected_index],
                    "selected_via_fallback": r.selected_via_fallback,
                }
                for r in reports
            ],
        }
```

### Step 4: Run, see pass

```
pytest tests/integration/test_keyframe_generator_stage.py -v
```
Expected: 4 PASSED.

```
pytest -q
```
Expected: ~227 PASSED total.

### Step 5: Commit

```
git add src/platinum/pipeline/keyframe_generator.py tests/integration/test_keyframe_generator_stage.py
git commit -m "feat(keyframe_generator): KeyframeGeneratorStage with DI seam + per-scene resume"
```

---

## Task 13: Quality sweep + smoke run + memory update

**Files:**
- Verify: full repo
- Modify: `C:/Users/claws/.claude/projects/C--Users-claws-OneDrive-Desktop-platinum/memory/project_platinum.md` -- append a Session 6 review section.

### Step 1: Run the full quality bar

```
pytest -q
```
Expected: ~227 tests, 0 failures, 0 skips, run time under 25s.

```
ruff check src tests
```
Expected: All checks passed.

```
mypy src
```
Expected: Success: no issues found in N source files (or accept previously-deferred items not introduced this session).

If new mypy errors appear in `comfyui.py` (httpx stubs), `keyframe_generator.py`, or `workflow.py`, narrow them with `# type: ignore[<exact-code>]` comments and document in the commit. Do NOT add a blanket `# type: ignore`.

### Step 2: Offline smoke run

Generate a 3-scene Story end-to-end through the Stage:

```
python - <<'PY'
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

from platinum.config import Config
from platinum.models.story import Scene, Source, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
from platinum.utils.aesthetics import MappedFakeScorer
from platinum.utils.comfyui import FakeComfyClient, workflow_signature
from platinum.utils.workflow import inject, load_workflow
from tests._fixtures import make_fake_hands_factory


async def main():
    repo_root = Path(__file__).resolve().parent
    cfg = Config(root=repo_root)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        story = Story(
            id="story_smoke_001",
            track="atmospheric_horror",
            source=Source(
                type="gutenberg",
                url="http://example",
                title="Smoke",
                author="x",
                raw_text="y",
                fetched_at=datetime(2026, 4, 25),
                license="PD-US",
            ),
            scenes=[
                Scene(
                    id=f"scene_{i:03d}",
                    index=i,
                    narration_text=f"S{i}",
                    narration_duration_seconds=5.0,
                    visual_prompt=f"candle scene {i}",
                    negative_prompt="bright daylight",
                )
                for i in range(3)
            ],
        )
        story_dir = d / "data" / "stories" / story.id
        story_dir.mkdir(parents=True, exist_ok=True)
        story.save(story_dir / "story.json")

        # Build responses for all 9 candidate generations.
        wf_template = load_workflow("flux_dev_keyframe", config_dir=cfg.config_dir)
        fixtures = repo_root / "tests" / "fixtures" / "keyframes"
        responses = {}
        for scene in story.scenes:
            for i, seed in enumerate((scene.index * 1000, scene.index * 1000 + 1, scene.index * 1000 + 2)):
                wf = inject(
                    wf_template,
                    prompt=f"cinematic dark, candlelight candle scene {scene.index}",
                    negative_prompt="bright daylight",
                    seed=seed,
                    width=1024, height=1024,
                    output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
                )
                responses[workflow_signature(wf)] = [fixtures / f"candidate_{i}.png"]

        # Score map: candidate_1 highest in scene 0; all-low in scene 1 (forces fallback);
        # candidate_2 highest in scene 2.
        score_map = {}
        for scene in story.scenes:
            scene_dir = story_dir / "keyframes" / f"scene_{scene.index:03d}"
            if scene.index == 0:
                score_map[scene_dir / "candidate_0.png"] = 6.5
                score_map[scene_dir / "candidate_1.png"] = 8.5
                score_map[scene_dir / "candidate_2.png"] = 7.0
            elif scene.index == 1:
                score_map[scene_dir / "candidate_0.png"] = 3.0
                score_map[scene_dir / "candidate_1.png"] = 4.5
                score_map[scene_dir / "candidate_2.png"] = 4.0
            else:
                score_map[scene_dir / "candidate_0.png"] = 6.0
                score_map[scene_dir / "candidate_1.png"] = 7.0
                score_map[scene_dir / "candidate_2.png"] = 9.0

        # Override Config.story_path for this smoke (it normally derives from project root).
        # Rather than reaching into Config, we mock PipelineContext.story_path through ctx.
        cfg.settings["test"] = {
            "comfy_client": FakeComfyClient(responses=responses),
            "aesthetic_scorer": MappedFakeScorer(scores_by_path=score_map, default=0.0),
            "mp_hands_factory": make_fake_hands_factory(None),
        }

        class _Ctx:
            config = cfg
            db_path = d / "data" / "platinum.db"

            def story_path(self, s):
                return story_dir / "story.json"

        ctx = _Ctx()
        stage = KeyframeGeneratorStage()
        artifacts = await stage.run(story, ctx)
        print("artifacts:", artifacts)
        for s in story.scenes:
            print(
                f"scene {s.index}: keyframe_path={s.keyframe_path.name}, "
                f"scores={s.keyframe_scores}, "
                f"fallback={s.validation.get('keyframe_selected_via_fallback')}"
            )


asyncio.run(main())
PY
```

Expected output:

```
artifacts: {'scenes_total': 3, 'scenes_succeeded': 3, ..., 'scenes_via_fallback': 1, 'scenes_failed': 0, ...}
scene 0: keyframe_path=candidate_1.png, scores=[6.5, 8.5, 7.0], fallback=False
scene 1: keyframe_path=candidate_0.png, scores=[3.0, 4.5, 4.0], fallback=True
scene 2: keyframe_path=candidate_2.png, scores=[6.0, 7.0, 9.0], fallback=False
```

### Step 3: Append Session 6 review to project memory

Open `C:/Users/claws/.claude/projects/C--Users-claws-OneDrive-Desktop-platinum/memory/project_platinum.md` and append a `## Review (Session 6 complete -- 2026-04-25)` section summarising:
- Test count delta (~37 new; ~227 total).
- Files added: `utils/comfyui.py`, `utils/workflow.py`, `pipeline/keyframe_generator.py`, `config/workflows/flux_dev_keyframe.json`, three `tests/fixtures/keyframes/candidate_N.png`, four new test files.
- Files modified: `utils/aesthetics.py` (+`MappedFakeScorer` + `RemoteAestheticScorer` stub), `tests/_fixtures.py` (+`make_synthetic_png` + hoisted `make_fake_hands_factory`), `tests/unit/test_validate.py` (uses hoisted helper), `tests/unit/test_aesthetics.py` (+3 tests).
- Decisions locked in (offline-first scope, baseline Flux only, `RemoteAestheticScorer` stubbed for S6.1, deterministic seeds per scene.index, sequential candidate generation, ties-to-lowest-index selection, per-scene mid-run resume via `keyframe_path is not None`).
- Lessons reinforced (late binding for testability, Protocol+Fake pattern, pure-core/impure-shell at the Stage boundary, MockTransport for HTTP-shape verification without a Fake, no-new-deps when existing libs cover the seam).
- Status update: Session 6 complete; Session 6.1 (vast.ai provisioning + real LAION + first live keyframe smoke) up next.

### Step 4: Final commit

```
git add -A
git commit -m "$(cat <<'EOF'
docs: Session 6 complete -- keyframe generator + ComfyUI client (offline-first)

ComfyClient Protocol with FakeComfyClient (rotating fixture map, response
keyed by sha256 of canonical workflow JSON) + HttpComfyClient (httpx
MockTransport-friendly transport seam, no live ComfyUI exercised).
utils/workflow.py loads + injects static workflow JSON via _meta.role
tags. flux_dev_keyframe.json baseline workflow shipped. KeyframeGenerator
Stage runs 3 candidates per scene at deterministic seeds, scores via
AestheticScorer, anatomy-checks via check_hand_anomalies, picks highest
passing or falls back to candidate 0. Per-scene mid-run resume via
keyframe_path is not None. RemoteAestheticScorer stubbed for Session 6.1.

Tests: ~227 total (~37 new), 0 fail, 0 skip. ruff + mypy clean.
EOF
)"
```

### Step 5: Push not required

`git push` is the user's call; the plan does not push automatically.

---

## Done criteria

- [ ] Tasks 1-13 each have a green commit landed on `main` (or the active worktree branch).
- [ ] `pytest -q` shows ~227 tests passing, 0 failures, 0 skips.
- [ ] `ruff check src tests` clean.
- [ ] `mypy src` clean (no new errors introduced this session; pre-existing items deferred per Session 5).
- [ ] `src/platinum/utils/comfyui.py`, `src/platinum/utils/workflow.py`, `src/platinum/pipeline/keyframe_generator.py` exist with the documented surface.
- [ ] `config/workflows/flux_dev_keyframe.json` exists with all five required `_meta.role` tags resolving to actual node ids.
- [ ] `tests/fixtures/keyframes/candidate_{0,1,2}.png` are committed.
- [ ] Project memory updated with Session 6 review.
- [ ] Session 6.1 (vast.ai + live keyframe smoke) can pick up by replacing `FakeComfyClient` with a configured `HttpComfyClient` and implementing `RemoteAestheticScorer`, with zero refactor of Session 6 code.
