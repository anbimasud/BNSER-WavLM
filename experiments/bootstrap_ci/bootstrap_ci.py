from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------
SEED = 42
N_BOOT = 10_000
CONFIDENCE_LEVEL = 0.95
EXPECTED_TEST_SIZE = 345


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_FILE = REPO_ROOT / "results" / "test_predictions.csv"
OUTPUT_FILE = REPO_ROOT / "results" / "bootstrap_results_recomputed.csv"

REQUIRED_COLUMNS = [
    "sample_id",
    "y_true",
    "model1_pred",
    "model2_pred",
    "model3_pred",
]

MODEL_COLUMNS = {
    "WavLM-FT": "model1_pred",
    "WavLM-FT+Aug": "model2_pred",
    "WavLM-PEFT": "model3_pred",
}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def load_and_validate_predictions(path: Path) -> pd.DataFrame:
    """Load the fixed-test-set predictions and validate their structure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Provide the archived prediction table with one row per test "
            "utterance and the required columns."
        )

    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required prediction columns: " + ", ".join(missing)
        )

    if len(df) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Expected exactly {EXPECTED_TEST_SIZE} test instances; "
            f"found {len(df)}."
        )

    if df["sample_id"].isna().any():
        raise ValueError("sample_id contains missing values.")

    if df["sample_id"].duplicated().any():
        duplicates = int(df["sample_id"].duplicated().sum())
        raise ValueError(
            f"sample_id must be unique; found {duplicates} duplicate row(s)."
        )

    for column in REQUIRED_COLUMNS[1:]:
        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

    # All models must use the same ground-truth labels for the same test rows.
    # The single y_true column is therefore the canonical ground truth.
    if df["y_true"].nunique() != 6:
        raise ValueError(
            "The BNSER evaluation is defined over six emotion classes; "
            f"found {df["y_true"].nunique()} classes in y_true."
        )

    # Prediction columns must contain labels that are compatible with the
    # observed ground-truth label set. This catches accidental numeric/string
    # mixing or predictions from a different label mapping.
    true_labels = set(df["y_true"].astype(str).unique())
    for model_name, column in MODEL_COLUMNS.items():
        predicted_labels = set(df[column].astype(str).unique())
        unexpected = sorted(predicted_labels - true_labels)
        if unexpected:
            raise ValueError(
                f"{model_name} contains prediction label(s) not present in "
                f"y_true: {unexpected}"
            )

    # Use a consistent label representation for all three model outputs.
    for column in REQUIRED_COLUMNS[1:]:
        df[column] = df[column].astype(str)

    return df


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------
def _confusion_counts(
    y_true: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return a fixed-label confusion matrix using vectorized counting."""
    true_codes = pd.Categorical(y_true, categories=labels).codes
    pred_codes = pd.Categorical(predictions, categories=labels).codes

    if np.any(true_codes < 0) or np.any(pred_codes < 0):
        raise ValueError("Found a label outside the six-class BNSER label set.")

    n_classes = len(labels)
    flat_indices = true_codes * n_classes + pred_codes
    return np.bincount(
        flat_indices,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)


def metrics_from_confusion_matrix(confusion: np.ndarray) -> dict:
    """Calculate the manuscript metrics from a six-class confusion matrix."""
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    total = confusion.sum()

    accuracy = np.trace(confusion) / total if total else 0.0

    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros_like(support, dtype=float),
        where=support != 0,
    )

    precision = np.divide(
        np.diag(confusion),
        predicted,
        out=np.zeros_like(predicted, dtype=float),
        where=predicted != 0,
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )

    present_classes = support > 0
    macro_f1 = f1[present_classes].mean() if np.any(present_classes) else 0.0
    weighted_f1 = (
        np.average(f1[present_classes], weights=support[present_classes])
        if np.any(present_classes)
        else 0.0
    )
    balanced_accuracy = recall[present_classes].mean() if np.any(present_classes) else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """Calculate the four metrics used in the manuscript."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
    }


# ---------------------------------------------------------------------------
# Bootstrap calculations
# ---------------------------------------------------------------------------
def bootstrap_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Estimate point values and 95% percentile CIs for one model."""
    n = len(y_true)
    point_estimates = calculate_metrics(y_true, predictions)

    bootstrap_values = {
        metric: np.empty(N_BOOT, dtype=float)
        for metric in point_estimates
    }

    for replicate in range(N_BOOT):
        indices = rng.integers(0, n, size=n)
        values = metrics_from_confusion_matrix(
            _confusion_counts(
                y_true[indices],
                predictions[indices],
                labels,
            )
        )

        for metric in bootstrap_values:
            bootstrap_values[metric][replicate] = values[metric]

    lower_quantile = (1.0 - CONFIDENCE_LEVEL) / 2.0
    upper_quantile = 1.0 - lower_quantile

    results = {}
    for metric, estimate in point_estimates.items():
        ci_lower, ci_upper = np.quantile(
            bootstrap_values[metric],
            [lower_quantile, upper_quantile],
        )
        results[metric] = {
            "estimate": estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return results


def paired_accuracy_bootstrap(
    y_true: np.ndarray,
    reference_predictions: np.ndarray,
    comparison_predictions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Estimate a paired Accuracy difference and its percentile CI.

    The comparison is defined as:
        Accuracy(comparison) - Accuracy(reference)

    The same bootstrap indices are applied to both configurations, preserving
    the pairing of predictions from the same test instances.
    """
    n = len(y_true)

    reference_correct = reference_predictions == y_true
    comparison_correct = comparison_predictions == y_true
    point_difference = comparison_correct.mean() - reference_correct.mean()

    bootstrap_differences = np.empty(N_BOOT, dtype=float)

    for replicate in range(N_BOOT):
        indices = rng.integers(0, n, size=n)
        bootstrap_differences[replicate] = (
            comparison_correct[indices].mean()
            - reference_correct[indices].mean()
        )

    lower_quantile = (1.0 - CONFIDENCE_LEVEL) / 2.0
    upper_quantile = 1.0 - lower_quantile
    ci_lower, ci_upper = np.quantile(
        bootstrap_differences,
        [lower_quantile, upper_quantile],
    )

    return point_difference, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------
def build_results(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete bootstrap analysis and return a tidy result table."""
    y_true = df["y_true"].to_numpy()
    labels = np.sort(df["y_true"].unique())

    # One deterministic RNG stream is used throughout, matching the original
    # analysis design and keeping the procedure fully reproducible.
    rng = np.random.default_rng(SEED)

    rows = []

    for model_name, prediction_column in MODEL_COLUMNS.items():
        predictions = df[prediction_column].to_numpy()
        bootstrap_results = bootstrap_metrics(y_true, predictions, labels, rng)

        for metric, values in bootstrap_results.items():
            rows.append(
                {
                    "comparison": model_name,
                    "metric": metric,
                    "n_test": len(df),
                    "estimate_percent": values["estimate"] * 100.0,
                    "ci_lower_percent": values["ci_lower"] * 100.0,
                    "ci_upper_percent": values["ci_upper"] * 100.0,
                    "bootstrap_replicates": N_BOOT,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "method": "percentile bootstrap",
                    "seed": SEED,
                }
            )

    paired_comparisons = [
        ("WavLM-FT", "WavLM-FT+Aug"),
        ("WavLM-FT+Aug", "WavLM-PEFT"),
        ("WavLM-FT", "WavLM-PEFT"),
    ]

    for reference_model, comparison_model in paired_comparisons:
        reference_predictions = df[MODEL_COLUMNS[reference_model]].to_numpy()
        comparison_predictions = df[MODEL_COLUMNS[comparison_model]].to_numpy()

        estimate, ci_lower, ci_upper = paired_accuracy_bootstrap(
            y_true,
            reference_predictions,
            comparison_predictions,
            rng,
        )

        rows.append(
            {
                "comparison": f"{comparison_model} - {reference_model}",
                "metric": "accuracy_difference_pp",
                "n_test": len(df),
                "estimate_percent": estimate * 100.0,
                "ci_lower_percent": ci_lower * 100.0,
                "ci_upper_percent": ci_upper * 100.0,
                "bootstrap_replicates": N_BOOT,
                "confidence_level": CONFIDENCE_LEVEL,
                "method": "paired percentile bootstrap",
                "seed": SEED,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "metric",
            "n_test",
            "estimate_percent",
            "ci_lower_percent",
            "ci_upper_percent",
            "bootstrap_replicates",
            "confidence_level",
            "method",
            "seed",
        ],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the bootstrap analysis and save the manuscript-facing CSV."""
    df = load_and_validate_predictions(PREDICTION_FILE)
    results = build_results(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")

    print(f"Test instances: {len(df)}")
    print(f"Bootstrap resamples: {N_BOOT}")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
