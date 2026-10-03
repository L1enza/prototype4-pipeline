"""Jersey-number legibility gate: only crops a classifier judges legible reach a reader.

Number readers invent numbers on illegible crops (gemma3:4b, Tesseract and
qwen2.5vl:7b were measured doing so), and the same invented number repeated across
frames passes the two-frame agreement rule. A legibility classifier screens crops
first. The model is the hockey-trained ResNet34 legibility classifier of Koshkina &
Elder, "A General Framework for Jersey Number Recognition in Sports Video" (CVPR 2024
workshops, github.com/mkoshkina/jersey-number-pipeline). Its weights are licensed
CC BY-NC 3.0 (non-commercial) and are downloaded separately, never committed.

On the 130 manually labelled nll_test4 crops (scripts/eval_legibility_classifier.py)
it passed 0 of 94 unreadable crops and 13 of 20 clear ones at 0.5.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

DEFAULT_WEIGHTS = ".cache/models/legibility_resnet34_hockey.pth"
DEFAULT_THRESHOLD = 0.5
HOCKEY_WEIGHTS_DRIVE_ID = "1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp"
# Pinned so a changed file at the shared link is refused rather than silently used.
HOCKEY_WEIGHTS_SHA256 = "98de82bc56fe2943bf9db961325fbd08a4508532182b3c27f8d4f4e3786bc786"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(path: Path, allow_download: bool = False) -> Path:
    """Return the verified weights path, downloading them first only when allowed."""
    path = Path(path)
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Legibility weights not found at {path}. Re-run with --allow-download-weights "
                "(85 MB, once; CC BY-NC 3.0, research use only) or pass --no-legibility-gate."
            )
        import gdown

        path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=HOCKEY_WEIGHTS_DRIVE_ID, output=str(path), quiet=False)
    actual = sha256_of(path)
    if actual != HOCKEY_WEIGHTS_SHA256:
        raise ValueError(f"Legibility weights at {path} have SHA-256 {actual}, expected {HOCKEY_WEIGHTS_SHA256}")
    return path


def load_model(weights: Path):
    """LegibilityClassifier34 from the paper: ResNet34 with a single sigmoid output."""
    import torch
    from torch import nn
    from torchvision import models

    class LegibilityClassifier34(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_ft = models.resnet34(weights=None)
            self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)

        def forward(self, x):
            return torch.sigmoid(self.model_ft(x))

    model = LegibilityClassifier34()
    # weights_only refuses pickled code; the published file is a plain state dict.
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


def score_images(model, images: Iterable, batch_size: int = 16) -> list[float]:
    """Legibility probability per image (a path or a PIL image), with the paper's test transforms."""
    import torch
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    items = list(images)
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = [
                transform((item if isinstance(item, Image.Image) else Image.open(item)).convert("RGB"))
                for item in items[start:start + batch_size]
            ]
            scores.extend(float(value) for value in model(torch.stack(batch)).flatten())
    return scores
