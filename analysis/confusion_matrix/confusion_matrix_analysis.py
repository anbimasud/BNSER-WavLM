"""Generate confusion matrices and descriptive error summaries for the three models.

Expected input (default):
    <repo_root>/results/test_predictions.csv

The canonical prediction file should contain one row per test sample and the
following fields (case-insensitive matching is supported):
    sample_id, y_true,
    model1_pred, model2_pred, model3_pred

The analysis is deliberately descriptive. It does not infer causes of
misclassification or attribute an error pattern to a single Model-3 component.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
EXPECTED_TEST_SAMPLES = 345

MODEL_COLUMNS = {
    "WavLM-FT": "model1_pred",
    "WavLM-FT+Aug": "model2_pred",
    "WavLM-PEFT": "model3_pred",
}


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    normalized = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(
            f"Required column not found. Tried {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Create results/test_predictions.csv from the archived model evaluation "
            "outputs before running this analysis. No prediction values are generated here."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Prediction file is empty.")

    sample_col = find_column(df, ["sample_id", "file_name", "filename", "file_path", "path", "id"])
    true_col = find_column(df, ["y_true", "true_label", "actual_label", "actual_emotion", "ground_truth"])

    out = pd.DataFrame({
        "sample_id": df[sample_col].astype(str).str.strip(),
        "y_true": df[true_col].astype(str).str.strip().str.lower(),
    })

    if out["sample_id"].duplicated().any():
        dup = out.loc[out["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"Duplicate sample identifiers detected, e.g. {dup}.")

    if len(out) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test predictions, found {len(out)}. "
            "The manuscript evaluates a fixed 345-sample test set."
        )

    unknown_true = sorted(set(out["y_true"]) - set(LABELS))
    if unknown_true:
        raise ValueError(f"Unknown ground-truth labels: {unknown_true}")

    for model, canonical in MODEL_COLUMNS.items():
        col = find_column(df, [canonical, model.replace("-", "_")], required=False)
        if col is None:
            raise ValueError(
                f"Missing prediction column for {model}. Expected a column named '{canonical}'."
            )
        out[canonical] = df[col].astype(str).str.strip().str.lower()
        unknown_pred = sorted(set(out[canonical]) - set(LABELS))
        if unknown_pred:
            raise ValueError(f"Unknown predicted labels for {model}: {unknown_pred}")

    return out


def write_model_matrix(y_true: pd.Series, y_pred: pd.Series, model: str, output_path: Path) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    matrix_df = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
    matrix_df.index.name = "True"
    matrix_df.to_csv(output_path)
    return matrix_df


def top_confusions(y_true: pd.Series, y_pred: pd.Series, model: str) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    rows = []
    for i, actual in enumerate(LABELS):
        for j, predicted in enumerate(LABELS):
            if i != j and matrix[i, j] > 0:
                rows.append({"Model": model, "Actual": actual, "Predicted": predicted, "Count": int(matrix[i, j])})
    return rows


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
    top_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model, pred_col in MODEL_COLUMNS.items():
        matrix_path = results_dir / {
            "WavLM-FT": "model1_confusion_matrix.csv",
            "WavLM-FT+Aug": "model2_confusion_matrix.csv",
            "WavLM-PEFT": "model3_confusion_matrix.csv",
        }[model]
        matrix = write_model_matrix(df["y_true"], df[pred_col], model, matrix_path)
        correct = int(np.trace(matrix.to_numpy()))
        n = len(df)
        summary_rows.append({
            "Model": model,
            "Test_Samples": n,
            "Correct": correct,
            "Accuracy_percent": correct / n * 100.0,
            "Errors": n - correct,
        })
        top_rows.extend(top_confusions(df["y_true"], df[pred_col], model))

    pd.DataFrame(summary_rows).to_csv(results_dir / "confusion_matrix_summary.csv", index=False)
    all_errors = pd.DataFrame(top_rows).sort_values(
        ["Model", "Count", "Actual", "Predicted"],
        ascending=[True, False, True, True],
    )
    top_df = all_errors.groupby("Model", sort=False, group_keys=False).head(5).reset_index(drop=True)
    top_df.to_csv(results_dir / "top_confusions.csv", index=False)

    print("Confusion-matrix analysis completed.")
    print(f"Input: {predictions_path}")
    print(f"Test samples: {len(df)}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
