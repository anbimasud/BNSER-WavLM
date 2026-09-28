"""WavLM-Large construction and selective layer freezing."""

from __future__ import annotations

import torch
from transformers import WavLMForSequenceClassification

from .utils import EMOTIONS, count_parameters


def build_model(
    model_name: str,
    num_labels: int,
    freeze_first_n_layers: int = 0,
) -> WavLMForSequenceClassification:
    """Load WavLM-Large and optionally freeze its first N Transformer layers."""
    model = WavLMForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    encoder_layers = model.wavlm.encoder.layers
    total_layers = len(encoder_layers)

    if not 0 <= freeze_first_n_layers <= total_layers:
        raise ValueError(
            f"freeze_first_n_layers must be between 0 and {total_layers}; "
            f"received {freeze_first_n_layers}."
        )

    for layer in encoder_layers[:freeze_first_n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    return model


def make_loss(
    class_weights: dict[str, float],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Create the configured cross-entropy loss and its class-weight tensor."""
    weights = torch.tensor(
        [class_weights[emotion] for emotion in EMOTIONS],
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    return criterion, weights


def model_parameter_summary(model) -> dict[str, int | float]:
    """Return parameter counts and the trainable-parameter reduction."""
    total, trainable, frozen = count_parameters(model)
    reduction = 100.0 * (1.0 - trainable / total) if total else 0.0

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_parameter_reduction_percent": reduction,
    }
