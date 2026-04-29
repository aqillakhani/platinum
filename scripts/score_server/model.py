"""LAION-Aesthetics v2 model singletons -- runs only on the vast.ai box.

Imports torch + open_clip; do NOT import this module in local CI.
server.py uses lazy `from score_server.model import ...` inside the
`get_scorer` dependency factory so local pytest can override the dep
without ever touching torch.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import open_clip  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLP_WEIGHTS = Path(
    os.environ.get(
        "LAION_MLP_PATH",
        "/workspace/models/aesthetic/sac+logos+ava1-l14-linearMSE.pth",
    )
)


class AestheticMLP(nn.Module):
    """4-layer MLP head for LAION-Aesthetics v2 (ViT-L/14 features -> score).

    Vendored from christophschuhmann/improved-aesthetic-predictor; weights
    file `sac+logos+ava1-l14-linearMSE.pth` is pre-trained on SAC + LOGOS +
    AVA1 datasets with linear MSE loss. Input: 768-dim CLIP image embedding.
    Output: scalar 0-10 aesthetic score.
    """

    def __init__(self, input_size: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# --- Module-level singletons; loaded once on first import --------------------
print(f"[score_server.model] loading CLIP ViT-L-14 on {DEVICE}...")  # noqa: T201
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai", device=DEVICE
)
clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

print(f"[score_server.model] loading MLP weights from {MLP_WEIGHTS}...")  # noqa: T201
_mlp = AestheticMLP(input_size=768)
_mlp.load_state_dict(torch.load(MLP_WEIGHTS, map_location=DEVICE))
_mlp.to(DEVICE).eval()
print("[score_server.model] ready.")  # noqa: T201


def score_image_bytes(image_bytes: bytes) -> float:
    """Score raw image bytes on the LAION-Aesthetics v2 0-10 scale."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        feats = clip_model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float(_mlp(feats).cpu().item())


def clip_similarity_image_text(image_bytes: bytes, text: str) -> float:
    """Cosine similarity between image and text in CLIP ViT-L-14 space (-1..1).

    Reuses the already-loaded LAION CLIP backbone (deviates from the S7.1 plan
    which named ViT-B/32 -- avoiding a second model load saves ~600 MB GPU
    memory). Text is encoded with the matching tokenizer.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        img_feats = clip_model.encode_image(img_tensor)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_tokens = clip_tokenizer([text]).to(DEVICE)
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return float((img_feats @ text_feats.T).cpu().item())
