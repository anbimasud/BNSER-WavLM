"""Shared training loop for the three BNSER-WavLM configurations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .evaluation import evaluate_loss


def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    accumulation_steps: int,
    output_dir: str | Path,
    validation_loss_aggregation: str,
) -> tuple[dict, list[dict]]:
    """Train with validation-loss early stopping and save the best checkpoint.

    The final Model-3 reference run used four-step gradient accumulation with a
    physical batch size of four. As in the original implementation, an incomplete
    final accumulation group is not stepped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    early_stop_counter = 0
    best_epoch = None
    history: list[dict] = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        for batch_index, batch in enumerate(train_loader):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_values=inputs).logits
            raw_loss = criterion(logits, labels)
            loss = raw_loss / accumulation_steps
            loss.backward()

            if (batch_index + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += raw_loss.item()
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            batch_count += 1

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
            aggregation=validation_loss_aggregation,
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": total_train_loss / max(batch_count, 1),
            "val_loss": validation_loss,
            "train_accuracy": train_correct / max(train_total, 1),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={epoch_record['train_loss']:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"train_acc={epoch_record['train_accuracy']:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            early_stop_counter = 0
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt",
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    torch.save(
        model.state_dict(),
        output_dir / "last_model.pt",
    )
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_trained": len(history),
    }
    return summary, history
