"""General utilities shared by the BNSER-WavLM training experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def set_seed(seed: int) -> None:
    """Set the random seeds used by the reported training protocol."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = total - trainable
    return total, trainable, frozen


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON artifact, creating its parent directory when necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def resolve_data_root(explicit_path: str | None = None) -> Path:
    """Resolve the local BNSER dataset root without embedding a private path."""
    value = explicit_path or os.environ.get("BNSER_DATA_DIR")

    if not value:
        raise RuntimeError(
            "Set BNSER_DATA_DIR to the directory containing Train/, Val/, and Test/."
        )

    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BNSER dataset directory not found: {root}")

    return root
