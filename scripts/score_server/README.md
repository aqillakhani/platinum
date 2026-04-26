# score_server

LAION-Aesthetics v2 score server (CLIP ViT-L/14 + 4-layer MLP head).
Runs on the vast.ai box alongside ComfyUI. Listens on port 8189.

## Endpoints

- `GET /health` -> `{"ok": true, "model": "ViT-L-14 + LAION MLP"}`
- `POST /score` (multipart `image=@file.png`) -> `{"score": <float>}`

## Runtime

Installed by `scripts/vast_setup.sh`. MLP weights downloaded to
`/workspace/models/aesthetic/sac+logos+ava1-l14-linearMSE.pth` (~6MB).
Model loaded once at process start; uvicorn keeps it alive across requests.

## Local testing

Tests in `tests/unit/test_score_server.py` use FastAPI `TestClient` with a
mocked scorer via `app.dependency_overrides[get_scorer]`. No torch import
needed in local CI; `model.py` is only imported lazily inside the
production `get_scorer` factory.

## Live setup

See: `docs/runbooks/vast-ai-keyframe-smoke.md` for the rent → provision →
launch → wire → smoke → tear-down runbook.
