"""Appearance feature helpers for team-level track assignment.

The module is intentionally lightweight and safe: pretrained torchvision weights
are used only when they are already cached or when the caller explicitly allows
downloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


RESNET18_FILENAME = "resnet18-f37072fd.pth"


@dataclass
class AppearanceBackend:
    requested_backend: str
    active_backend: str
    model: Any = None
    transform: Any = None
    torch_module: Any = None
    metadata: Dict[str, Any] = None

    def embed_rgb(self, crop_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or self.transform is None or self.torch_module is None:
            return None
        if crop_rgb.size == 0:
            return None
        image = Image.fromarray(crop_rgb.astype(np.uint8), mode="RGB")
        tensor = self.transform(image).unsqueeze(0)
        with self.torch_module.inference_mode():
            vector = self.model(tensor).flatten().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return None
        return vector / norm


def _candidate_cache_paths() -> list[Path]:
    paths = []
    home = Path.home()
    paths.append(home / ".cache" / "torch" / "hub" / "checkpoints" / RESNET18_FILENAME)
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        paths.append(Path(torch_home) / "hub" / "checkpoints" / RESNET18_FILENAME)
        paths.append(Path(torch_home) / "checkpoints" / RESNET18_FILENAME)
    return paths


def cached_resnet18_weights_path() -> Optional[Path]:
    for path in _candidate_cache_paths():
        if path.exists():
            return path
    return None


def load_appearance_backend(backend: str = "auto", allow_download_weights: bool = False) -> AppearanceBackend:
    """Load a pretrained appearance embedding backend if it is safe to do so.

    Returns a backend object even when unavailable. In that case ``active_backend``
    is ``"none"`` and metadata explains why color-only mode is being used.
    """
    requested = backend
    metadata: Dict[str, Any] = {
        "requested_backend": requested,
        "allow_download_weights": bool(allow_download_weights),
        "weights_downloaded": False,
        "weights_cached_before_load": False,
        "cache_candidates": [str(path) for path in _candidate_cache_paths()],
    }
    if backend == "none":
        metadata.update({"active_backend": "none", "status": "disabled_by_config"})
        return AppearanceBackend(requested, "none", metadata=metadata)
    if backend not in ("auto", "torchvision_resnet18"):
        metadata.update({"active_backend": "none", "status": "unknown_backend"})
        return AppearanceBackend(requested, "none", metadata=metadata)

    cached = cached_resnet18_weights_path()
    metadata["weights_cached_before_load"] = cached is not None
    metadata["cached_weights_path"] = str(cached) if cached else None
    if cached is None and not allow_download_weights:
        metadata.update({
            "active_backend": "none",
            "status": "weights_not_cached_download_not_allowed",
            "evidence_mode": "color_only",
        })
        return AppearanceBackend(requested, "none", metadata=metadata)

    try:
        import torch
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:  # pragma: no cover - depends on local optional deps.
        metadata.update({
            "active_backend": "none",
            "status": "import_failed",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "evidence_mode": "color_only",
        })
        return AppearanceBackend(requested, "none", metadata=metadata)

    try:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        feature_model = nn.Sequential(*(list(model.children())[:-1]))
        feature_model.eval()
        for param in feature_model.parameters():
            param.requires_grad_(False)
        transform = weights.transforms()
        metadata.update({
            "active_backend": "torchvision_resnet18",
            "status": "loaded",
            "weights_enum": str(weights),
            "weights_downloaded": cached is None and allow_download_weights,
            "embedding_dim": 512,
            "evidence_mode": "appearance_plus_color",
        })
        return AppearanceBackend(
            requested_backend=requested,
            active_backend="torchvision_resnet18",
            model=feature_model,
            transform=transform,
            torch_module=torch,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - depends on local optional deps.
        metadata.update({
            "active_backend": "none",
            "status": "load_failed",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "evidence_mode": "color_only",
        })
        return AppearanceBackend(requested, "none", metadata=metadata)
