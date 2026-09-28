"""Evaluation utilities for the fixed BNSER test protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from .utils import EMOTIONS


def evaluate_loss(
    model,
    loader,
    criterion,
    device: torch.device,
    aggregation: str = "batch_mean",
) -> float:
    """Calculate validation loss using the selected historical aggregation rule."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = batch["input_values"].to(device)
            logits = model(input_values=inputs).logits
            loss = criterion(logits, labels)

            if aggregation == "sample_weighted":
                total_loss += loss.item() * labels.size(0)
                total_examples += labels.size(0)
            elif aggregation == "batch_mean":
                total_loss += loss.item()
                total_batches += 1
            else:
                raise ValueError(
                    "Unknown validation loss aggregation: "
                    f"{aggregation}"
                )

    if aggregation == "sample_weighted":
        return total_loss / max(total_examples, 1)

    return total_loss / max(total_batches, 1)


def predict(model, loader, device: torch.device) -> pd.DataFrame:
    """Collect test predictions and class probabilities without augmentation."""
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = batch["input_values"].to(device)
            logits = model(input_values=inputs).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)

            for index, prediction in enumerate(predictions):
                row = {
                    "sample_id": batch["sample_id"][index],
                    "speaker": batch["speaker"][index],
                    "path": batch["path"][index],
                    "y_true": int(labels[index].item()),
                    "y_true_label": batch["emotion"][index],
                    "y_pred": int(prediction),
                    "y_pred_label": EMOTIONS[int(prediction)],
                    "confidence": float(probabilities[index, prediction]),
                }

                row.update(
                    {
                        f"prob_{emotion}": float(probabilities[index, class_index])
                        for class_index, emotion in enumerate(EMOTIONS)
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, float | None]:
    """Calculate the main manuscript metrics from test predictions."""
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    probabilities = predictions[
        [f"prob_{emotion}" for emotion in EMOTIONS]
    ].to_numpy()

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
    }

    try:
        metrics["macro_auc_ovr"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        metrics["macro_auc_ovr"] = None

    return metrics


def save_evaluation_outputs(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, float | None]:
    """Save predictions, class-wise report, and machine-readable metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = summarize_predictions(predictions)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    report = classification_report(
        predictions["y_true"],
        predictions["y_pred"],
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        output_dir / "classification_report.csv"
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
