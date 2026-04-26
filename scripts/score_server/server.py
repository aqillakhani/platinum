"""LAION-Aesthetics v2 score server.

Three-layer split for testability:
- server.py    -- FastAPI app, handlers, Depends-injected scorer factory.
- model.py     -- torch + open_clip imports + model singletons (box-only;
                  imported lazily inside get_scorer so local pytest never
                  triggers torch import).

Tests override `get_scorer` via `app.dependency_overrides[get_scorer]` to
inject a callable that returns a fixed score, avoiding the GPU stack.
See tests/unit/test_score_server.py.

Endpoints:
- GET /health  -> {ok: True, model: "ViT-L-14 + LAION MLP"}
- POST /score (multipart image) -> {score: float}
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

app = FastAPI(title="platinum score_server", version="0.1.0")


# Type alias for the scorer dependency: bytes -> float.
Scorer = Callable[[bytes], float]


def get_scorer() -> Scorer:
    """Returns the production scorer (LAION CLIP+MLP).

    Tests override this via `app.dependency_overrides[get_scorer]`. The lazy
    import keeps torch + open_clip out of local CI -- model.py is only loaded
    when the production path runs.
    """
    from score_server.model import score_image_bytes  # lazy import: torch + open_clip
    return score_image_bytes


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness/readiness probe for the score server."""
    return {"ok": True, "model": "ViT-L-14 + LAION MLP"}
