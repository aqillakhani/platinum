"""LAION-Aesthetics v2 score server.

Three-layer split for testability:
- server.py    -- FastAPI app, handlers, Depends-injected scorer factory.
- model.py     -- torch + open_clip imports + model singletons (box-only;
                  imported lazily inside get_scorer so local pytest never
                  triggers torch import).

Tests override `get_scorer` and `get_clip_sim_scorer` via
`app.dependency_overrides[...]` to inject callables that return fixed
values, avoiding the GPU stack.

Endpoints:
- GET  /health    -> {ok: True, model: "ViT-L-14 + LAION MLP"}
- POST /score     (multipart image) -> {score: float}
- POST /clip-sim  (multipart image + text form field) -> {similarity: float}
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="platinum score_server", version="0.1.0")


# Type alias for the scorer dependency: bytes -> float.
Scorer = Callable[[bytes], float]
# Type alias for the CLIP similarity dep: (bytes, text) -> cosine similarity.
ClipSimScorer = Callable[[bytes, str], float]


def get_scorer() -> Scorer:
    """Returns the production scorer (LAION CLIP+MLP).

    Tests override this via `app.dependency_overrides[get_scorer]`. The lazy
    import keeps torch + open_clip out of local CI -- model.py is only loaded
    when the production path runs.
    """
    from score_server.model import score_image_bytes  # lazy import: torch + open_clip
    return score_image_bytes


def get_clip_sim_scorer() -> ClipSimScorer:
    """Returns the production CLIP image-text similarity callable.

    Reuses the LAION ViT-L-14 backbone already loaded for the aesthetic
    scorer; the function returns cosine similarity in [-1, 1]. Lazy import
    keeps torch out of local CI.
    """
    from score_server.model import clip_similarity_image_text  # lazy
    return clip_similarity_image_text


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness/readiness probe for the score server."""
    return {"ok": True, "model": "ViT-L-14 + LAION MLP"}


@app.post("/score")
async def score(
    image: UploadFile = File(...),  # noqa: B008 -- FastAPI requires File() in defaults
    scorer: Scorer = Depends(get_scorer),  # noqa: B008 -- FastAPI requires Depends() in defaults
) -> dict[str, float]:
    """Score an uploaded image on the LAION-Aesthetics v2 0-10 scale."""
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image upload")
    try:
        Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode image: {exc}"
        ) from exc
    score_value = float(scorer(raw))
    if not math.isfinite(score_value):
        raise HTTPException(status_code=500, detail="model returned non-finite score")
    return {"score": score_value}


@app.post("/clip-sim")
async def clip_sim(
    image: UploadFile = File(...),  # noqa: B008 -- FastAPI requires File() in defaults
    text: str = Form(...),  # noqa: B008 -- FastAPI requires Form() in defaults
    scorer: ClipSimScorer = Depends(get_clip_sim_scorer),  # noqa: B008
) -> dict[str, float]:
    """CLIP image-text cosine similarity in [-1, 1]; primary content gate input.

    Used by the keyframe pipeline to reject candidates whose rendered image
    drifts too far from the visual_prompt's semantic content (S7.1 retro:
    Cask Phase 2 saw 8/16 scenes match render quality but miss prompt
    intent).
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image upload")
    try:
        Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode image: {exc}"
        ) from exc
    similarity = float(scorer(raw, text))
    if not math.isfinite(similarity):
        raise HTTPException(
            status_code=500, detail="model returned non-finite similarity"
        )
    return {"similarity": similarity}
