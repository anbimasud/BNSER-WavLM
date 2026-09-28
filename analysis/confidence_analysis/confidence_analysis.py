"""Compute descriptive prediction-confidence summaries for the three models.

This is a confidence-separation analysis, not a formal calibration analysis.
No ECE, reliability diagram, Brier score, or calibration test is computed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

EXPECTED_TEST_SAMPLES = 345
MODEL_FIELDS = {
    "WavLM-FT": "model1_confidence",
    "WavLM-FT+Aug": "model2_confidence",
    "WavLM-PEFT": "model3_confidence",
}


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Provide the canonical per-sample prediction file before running this analysis."
        )
    df = pd.read_csv(path)
    if len(df) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, found {len(df)}."
        )

    status_col = find_column(df, ["status"])
    if status_col is None:
        true_col = find_column(df, ["y_true", "true_label", "actual_label"])
        pred_cols = [find_column(df, [f"model{i}_pred"]) for i in range(1, 4)]
        if true_col is None or any(c is None for c in pred_cols):
            raise ValueError(
                "The canonical prediction file must contain either status/model-specific "
                "evaluation fields or y_true plus model1_pred/model2_pred/model3_pred."
            )

    return df


def confidence_summary(df: pd.DataFrame, model: str, confidence_col: str) -> dict:
    col = find_column(df, [confidence_col])
    if col is None:
        raise ValueError(
            f"Raw per-sample confidence values for {model} are missing. "
            f"Expected column '{confidence_col}'. The script will not reconstruct them from summary statistics."
        )

    values = pd.to_numeric(df[col], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Non-numeric or missing confidence values found for {model}.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"Confidence values for {model} must lie in [0, 1].")

    # Prefer an explicitly model-specific status field. Otherwise derive
    # correctness from the corresponding prediction and common ground truth.
    model_num = list(MODEL_FIELDS).index(model) + 1
    status_col = find_column(df, [f"model{model_num}_status", f"model_{model_num}_status"])
    if status_col is not None:
        status = df[status_col].astype(str).str.strip().str.lower()
        correct_mask = status.eq("correct")
        wrong_mask = status.eq("wrong")
    else:
        true_col = find_column(df, ["y_true", "true_label", "actual_label", "actual_emotion"])
        pred_col = find_column(df, [f"model{model_num}_pred"])
        if true_col is None or pred_col is None:
            raise ValueError(f"Cannot determine correct/incorrect status for {model}.")
        correct_mask = df[true_col].astype(str).str.strip().str.lower().eq(
            df[pred_col].astype(str).str.strip().str.lower()
        )
        wrong_mask = ~correct_mask

    correct = values[correct_mask]
    wrong = values[wrong_mask]
    if len(correct) + len(wrong) != len(df) or len(correct) == 0 or len(wrong) == 0:
        raise ValueError(f"Invalid correct/incorrect partition for {model}.")

    mean_correct = float(correct.mean())
    mean_wrong = float(wrong.mean())
    return {
        "Model": model,
        "Correct_Predictions": int(len(correct)),
        "Incorrect_Predictions": int(len(wrong)),
        "Error_Rate_percent": float(len(wrong) / len(df) * 100.0),
        "Avg_Confidence_Correct": mean_correct,
        "Avg_Confidence_Incorrect": mean_wrong,
        "Confidence_Gap": mean_correct - mean_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    predictions_path = (args.predictions or (repo_root / "results" / "test_predictions.csv")).resolve()
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(predictions_path)
    rows = [confidence_summary(df, model, field) for model, field in MODEL_FIELDS.items()]
    output = pd.DataFrame(rows)
    output.to_csv(results_dir / "confidence_summary.csv", index=False)

    print("Prediction-confidence analysis completed.")
    print("This is a descriptive confidence-separation analysis, not a calibration assessment.")
    print(f"Input: {predictions_path}")
    print(f"Results: {results_dir / 'confidence_summary.csv'}")


if __name__ == "__main__":
    main()
