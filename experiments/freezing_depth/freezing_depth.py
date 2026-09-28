from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

# Allow execution as:
#     python experiments/freezing_depth/freezing_depth.py
# while keeping the shared training implementation in one place.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import f1_score

from experiments.training.common.augmentation import ControlledAugmentation
from experiments.training.common.data import (
    create_dataloaders,
    create_feature_extractor,
    load_predefined_splits,
)
from experiments.training.common.evaluation import predict, save_evaluation_outputs
from experiments.training.common.model import build_model, model_parameter_summary
from experiments.training.common.training import train_model
from experiments.training.common.utils import EMOTIONS, save_json, set_seed


# ---------------------------------------------------------------------------
# Manuscript-defined experiment settings
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/wavlm-large"
NUM_LABELS = 6
SEED = 42
SAMPLE_RATE = 16_000
DURATION_SECONDS = 4.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SECONDS)
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
MAX_EPOCHS = 50
PATIENCE = 5
AUGMENTATION_PROBABILITY = 0.35
NOISE_PROBABILITY = 0.40
PITCH_PROBABILITY = 0.30
TIME_STRETCH_PROBABILITY = 0.30
NEUTRAL_WEIGHT = 1.4

# The 12-layer configuration is already the reported WavLM-PEFT reference.
REFERENCE_FREEZING_DEPTH = 12
ADDITIONAL_FREEZING_DEPTHS = (6, 8, 16, 18)

EXPECTED_SPLIT_COUNTS = {
    "train": 1610,
    "validation": 345,
    "test": 345,
}
EXPECTED_TEST_SPEAKERS = 4
EXPECTED_TOTAL_LAYERS = 24

OUTPUT_DIR = Path(__file__).resolve().parent
RUNS_DIR = OUTPUT_DIR / "runs"
RESULTS_CSV = OUTPUT_DIR / "freezing_depth_results.csv"
TEST_SET_VERIFICATION_CSV = OUTPUT_DIR / "same_test_set_verification.csv"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BNSER-WavLM freezing-depth sensitivity experiment."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "BNSER dataset root containing Train/, Val/, and Test/. "
            "If omitted, BNSER_DATA_DIR is used."
        ),
    )
    parser.add_argument(
        "--reference-predictions",
        default=None,
        help=(
            "Optional CSV containing the existing 12-layer reference test "
            "predictions. If supplied, sample IDs and ground-truth labels "
            "are compared against every additional run."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="HuggingFace WavLM checkpoint. Defaults to microsoft/wavlm-large.",
    )
    return parser.parse_args()


def load_model3_config() -> dict[str, Any]:
    """Read the repository's canonical Model-3 configuration.

    Freezing-depth experiments intentionally inherit the primary PEFT settings
    instead of maintaining a second independent configuration file.
    """
    config_path = (
        REPO_ROOT
        / "experiments"
        / "training"
        / "configs"
        / "model3_wavlm_peft.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            "Canonical Model-3 configuration was not found at "
            f"{config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return config


def validate_configuration(config: dict[str, Any]) -> None:
    """Fail fast if the canonical PEFT configuration drifts from the manuscript."""
    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    aug_cfg = config["augmentation"]
    loss_cfg = config["loss"]

    if model_cfg["num_labels"] != NUM_LABELS:
        raise ValueError("Model-3 must use six emotion classes.")
    if data_cfg["sample_rate"] != SAMPLE_RATE:
        raise ValueError("Model-3 sample rate must remain 16 kHz.")
    if float(data_cfg["duration_seconds"]) != DURATION_SECONDS:
        raise ValueError("Model-3 input duration must remain 4 seconds.")
    if training_cfg["seed"] != SEED:
        raise ValueError("Model-3 seed must remain 42.")
    if training_cfg["batch_size"] != BATCH_SIZE:
        raise ValueError("Model-3 physical batch size must remain 4.")
    if training_cfg["gradient_accumulation_steps"] != ACCUMULATION_STEPS:
        raise ValueError("Model-3 accumulation must remain four steps.")
    if float(training_cfg["learning_rate"]) != LEARNING_RATE:
        raise ValueError("Model-3 learning rate must remain 1e-5.")
    if training_cfg["max_epochs"] != MAX_EPOCHS:
        raise ValueError("Model-3 maximum epochs must remain 50.")
    if training_cfg["early_stopping_patience"] != PATIENCE:
        raise ValueError("Model-3 early-stopping patience must remain 5.")
    if training_cfg["validation_loss_aggregation"] != "batch_mean":
        raise ValueError(
            "The freezing-depth experiment must preserve Model-3's batch-mean "
            "validation-loss aggregation."
        )
    if not aug_cfg["enabled"]:
        raise ValueError("Freezing-depth runs must use training-only augmentation.")
    if float(aug_cfg["probability"]) != AUGMENTATION_PROBABILITY:
        raise ValueError("Augmentation probability must remain 35%.")
    if float(aug_cfg["noise_probability"]) != NOISE_PROBABILITY:
        raise ValueError("Noise probability must remain 40%.")
    if float(aug_cfg["pitch_probability"]) != PITCH_PROBABILITY:
        raise ValueError("Pitch-shift probability must remain 30%.")
    if float(aug_cfg["time_stretch_probability"]) != TIME_STRETCH_PROBABILITY:
        raise ValueError("Time-stretch probability must remain 30%.")
    if loss_cfg["type"] != "weighted_cross_entropy":
        raise ValueError("Freezing-depth runs must use weighted cross-entropy.")
    if float(loss_cfg["class_weights"]["neutral"]) != NEUTRAL_WEIGHT:
        raise ValueError("Neutral class weight must remain 1.4.")


# ---------------------------------------------------------------------------
# Dataset and test-set verification
# ---------------------------------------------------------------------------
def verify_split_sizes(splits: dict[str, list[dict[str, str]]]) -> None:
    observed = {name: len(records) for name, records in splits.items()}
    if observed != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "The predefined dataset split sizes do not match the manuscript: "
            f"expected {EXPECTED_SPLIT_COUNTS}, observed {observed}."
        )

    test_speakers = {record["speaker"] for record in splits["test"]}
    if len(test_speakers) != EXPECTED_TEST_SPEAKERS:
        raise ValueError(
            "The predefined test split should contain four held-out speakers; "
            f"observed {len(test_speakers)}."
        )


def test_sample_key(record: dict[str, str]) -> tuple[str, str, str]:
    """Return a stable key that identifies a test utterance without absolute paths."""
    return (
        record["speaker"],
        record["emotion"],
        Path(record["path"]).name,
    )


def expected_test_keys(records: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {test_sample_key(record) for record in records}


def load_reference_keys(reference_path: str | None) -> set[tuple[str, str, str]] | None:
    """Load test-sample identities from an existing 12-layer prediction CSV."""
    if not reference_path:
        return None

    path = Path(reference_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference prediction CSV not found: {path}")

    keys: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"speaker", "y_true_label"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                "Reference predictions must contain at least 'speaker' and "
                "'y_true_label'."
            )

        for row in reader:
            # Prefer the explicit path if present; otherwise use sample_id.
            if row.get("path"):
                filename = Path(row["path"]).name
            elif row.get("sample_id"):
                filename = row["sample_id"] + ".wav"
            else:
                raise ValueError(
                    "Reference predictions need either 'path' or 'sample_id'."
                )
            keys.add((row["speaker"], row["y_true_label"], filename))

    if len(keys) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError(
            "The supplied 12-layer reference predictions do not contain exactly "
            f"{EXPECTED_SPLIT_COUNTS['test']} unique test samples."
        )

    return keys


# ---------------------------------------------------------------------------
# Model construction and experiment execution
# ---------------------------------------------------------------------------
def build_peft_components(model_name: str, freeze_depth: int, device: torch.device):
    """Build one full-PEFT configuration at the requested freezing depth."""
    model = build_model(
        model_name=model_name,
        num_labels=NUM_LABELS,
        freeze_first_n_layers=freeze_depth,
    ).to(device)

    encoder_layers = model.wavlm.encoder.layers
    if len(encoder_layers) != EXPECTED_TOTAL_LAYERS:
        raise RuntimeError(
            "Expected WavLM-Large to expose 24 Transformer layers, "
            f"but found {len(encoder_layers)}."
        )

    for index, layer in enumerate(encoder_layers):
        expected_frozen = index < freeze_depth
        actual_frozen = not any(parameter.requires_grad for parameter in layer.parameters())
        if actual_frozen != expected_frozen:
            raise RuntimeError(
                f"Layer {index} freeze-state verification failed for depth "
                f"{freeze_depth}: expected_frozen={expected_frozen}, "
                f"actual_frozen={actual_frozen}."
            )

    device_weights = torch.tensor(
        [
            1.0,  # angry
            1.0,  # disgust
            1.0,  # fear
            1.0,  # happy
            NEUTRAL_WEIGHT,
            1.0,  # sad
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=device_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


def run_one_depth(
    *,
    freeze_depth: int,
    model_name: str,
    splits: dict[str, list[dict[str, str]]],
    feature_extractor,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train and evaluate one additional freezing depth."""
    run_dir = RUNS_DIR / f"freeze_{freeze_depth}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # The same seed is reset at the start of each run so that differences are
    # attributable to the requested freezing depth rather than uncontrolled
    # changes in initialization, shuffling, or augmentation sampling.
    set_seed(SEED)

    augmentation = ControlledAugmentation(
        probability=AUGMENTATION_PROBABILITY,
        noise_probability=NOISE_PROBABILITY,
        pitch_probability=PITCH_PROBABILITY,
        time_stretch_probability=TIME_STRETCH_PROBABILITY,
        noise_level_min=float(config["augmentation"]["noise_level_min"]),
        noise_level_max=float(config["augmentation"]["noise_level_max"]),
        pitch_steps_min=float(config["augmentation"]["pitch_steps_min"]),
        pitch_steps_max=float(config["augmentation"]["pitch_steps_max"]),
        stretch_rate_min=float(config["augmentation"]["stretch_rate_min"]),
        stretch_rate_max=float(config["augmentation"]["stretch_rate_max"]),
        sample_rate=SAMPLE_RATE,
    )

    loaders = create_dataloaders(
        splits=splits,
        feature_extractor=feature_extractor,
        sample_rate=SAMPLE_RATE,
        target_length=TARGET_LENGTH,
        batch_size=BATCH_SIZE,
        augmentation=augmentation,
    )

    model, criterion, optimizer = build_peft_components(
        model_name=model_name,
        freeze_depth=freeze_depth,
        device=device,
    )

    parameter_summary = model_parameter_summary(model)

    # WavLM-Large has 24 Transformer layers. The parameter counts are computed
    # from the loaded checkpoint rather than hard-coded into the experiment.
    train_summary, _ = train_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        accumulation_steps=ACCUMULATION_STEPS,
        output_dir=run_dir,
        validation_loss_aggregation="batch_mean",
    )

    best_checkpoint = run_dir / "best_model.pt"
    if not best_checkpoint.is_file():
        raise RuntimeError(f"Best checkpoint was not created: {best_checkpoint}")

    state_dict = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    predictions = predict(model, loaders["test"], device)
    metrics = save_evaluation_outputs(predictions, run_dir)
    macro_f1 = float(
        f1_score(
            predictions["y_true"],
            predictions["y_pred"],
            average="macro",
            zero_division=0,
        )
    )

    prediction_keys = {
        (
            row["speaker"],
            row["y_true_label"],
            Path(row["path"]).name,
        )
        for _, row in predictions.iterrows()
    }
    expected_keys = expected_test_keys(splits["test"])

    if prediction_keys != expected_keys:
        missing = sorted(expected_keys - prediction_keys)
        extra = sorted(prediction_keys - expected_keys)
        raise RuntimeError(
            f"Test-set identity verification failed for freeze={freeze_depth}: "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    save_json(
        {
            "experiment": "freezing_depth_sensitivity",
            "freeze_depth": freeze_depth,
            "configuration": "full_peft",
            "reference_depth": REFERENCE_FREEZING_DEPTH,
            "seed": SEED,
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": DURATION_SECONDS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
            "augmentation_probability": AUGMENTATION_PROBABILITY,
            "augmentation_type_probabilities": {
                "noise": NOISE_PROBABILITY,
                "pitch_shift": PITCH_PROBABILITY,
                "time_stretch": TIME_STRETCH_PROBABILITY,
            },
            "class_weights": dict(config["loss"]["class_weights"]),
            "parameter_summary": parameter_summary,
            "training_summary": train_summary,
            "metrics": metrics,
        },
        run_dir / "experiment_config.json",
    )

    return {
        "freeze_depth": freeze_depth,
        "configuration": "full_peft",
        "test_samples": len(predictions),
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": macro_f1,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_auc_ovr": metrics["macro_auc_ovr"],
        "best_validation_loss": train_summary["best_validation_loss"],
        "best_epoch": train_summary["best_epoch"],
        "total_parameters": parameter_summary["total_parameters"],
        "trainable_parameters": parameter_summary["trainable_parameters"],
        "frozen_parameters": parameter_summary["frozen_parameters"],
        "trainable_parameter_reduction_percent": parameter_summary[
            "trainable_parameter_reduction_percent"
        ],
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Verification and result writing
# ---------------------------------------------------------------------------
def write_results(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "freeze_depth",
        "configuration",
        "test_samples",
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "balanced_accuracy",
        "macro_auc_ovr",
        "best_validation_loss",
        "best_epoch",
        "total_parameters",
        "trainable_parameters",
        "frozen_parameters",
        "trainable_parameter_reduction_percent",
        "status",
    ]

    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_test_set_verification(
    expected_keys: set[tuple[str, str, str]],
    reference_keys: set[tuple[str, str, str]] | None,
) -> None:
    """Write explicit same-test-set checks for all evaluated additional depths."""
    output_rows: list[dict[str, Any]] = []

    for depth in ADDITIONAL_FREEZING_DEPTHS:
        prediction_path = RUNS_DIR / f"freeze_{depth}" / "test_predictions.csv"
        if not prediction_path.is_file():
            continue

        keys: set[tuple[str, str, str]] = set()
        with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                keys.add(
                    (
                        row["speaker"],
                        row["y_true_label"],
                        Path(row["path"]).name,
                    )
                )

        row = {
            "freeze_depth": depth,
            "n_predictions": len(keys),
            "unique_test_samples": len(keys),
            "same_predefined_test_set": keys == expected_keys,
            "missing_samples_vs_predefined_test_set": len(expected_keys - keys),
            "extra_samples_vs_predefined_test_set": len(keys - expected_keys),
        }

        if reference_keys is not None:
            row.update(
                {
                    "same_test_set_as_12_layer_reference": keys == reference_keys,
                    "missing_samples_vs_12_layer_reference": len(reference_keys - keys),
                    "extra_samples_vs_12_layer_reference": len(keys - reference_keys),
                }
            )
        else:
            row.update(
                {
                    "same_test_set_as_12_layer_reference": "not_checked",
                    "missing_samples_vs_12_layer_reference": "not_checked",
                    "extra_samples_vs_12_layer_reference": "not_checked",
                }
            )

        row["verification_pass"] = bool(
            keys == expected_keys
            and (reference_keys is None or keys == reference_keys)
        )
        output_rows.append(row)

    fieldnames = [
        "freeze_depth",
        "n_predictions",
        "unique_test_samples",
        "same_predefined_test_set",
        "missing_samples_vs_predefined_test_set",
        "extra_samples_vs_predefined_test_set",
        "same_test_set_as_12_layer_reference",
        "missing_samples_vs_12_layer_reference",
        "extra_samples_vs_12_layer_reference",
        "verification_pass",
    ]

    with TEST_SET_VERIFICATION_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    args = parse_args()

    config = load_model3_config()
    validate_configuration(config)

    data_dir = args.data_dir or os.environ.get("BNSER_DATA_DIR")
    if not data_dir:
        raise RuntimeError(
            "BNSER_DATA_DIR is not set. Provide --data-dir or set "
            "BNSER_DATA_DIR to the predefined BNSER dataset root."
        )

    data_root = Path(data_dir).expanduser().resolve()
    split_names = config["data"]["split_names"]
    splits = load_predefined_splits(data_root, split_names)
    verify_split_sizes(splits)

    expected_keys = expected_test_keys(splits["test"])
    reference_keys = load_reference_keys(args.reference_predictions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {data_root}")
    print(
        "Freezing-depth sensitivity: "
        f"additional depths={list(ADDITIONAL_FREEZING_DEPTHS)}; "
        f"12-layer reference is not retrained."
    )

    feature_extractor = create_feature_extractor(
        model_name=args.model_name,
        max_length=TARGET_LENGTH,
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for freeze_depth in ADDITIONAL_FREEZING_DEPTHS:
        print("\n" + "=" * 72)
        print(f"Running full-PEFT configuration with {freeze_depth} frozen layers")
        print("=" * 72)

        result = run_one_depth(
            freeze_depth=freeze_depth,
            model_name=args.model_name,
            splits=splits,
            feature_extractor=feature_extractor,
            device=device,
            config=config,
        )
        results.append(result)

    write_results(results)
    write_test_set_verification(expected_keys, reference_keys)

    print("\nFreezing-depth experiment completed.")
    print(f"Results: {RESULTS_CSV}")
    print(f"Test-set verification: {TEST_SET_VERIFICATION_CSV}")
    if reference_keys is None:
        print(
            "Note: the 12-layer reference test predictions were not supplied; "
            "verification against the predefined 345-sample test set was performed."
        )
    else:
        print("The additional runs were also checked against the supplied 12-layer reference test set.")


if __name__ == "__main__":
    main()
