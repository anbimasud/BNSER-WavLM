#!/usr/bin/env python3
"""Source-stratified evaluation of the fixed BNSER test set.

This script reproduces the source-stratified accuracy analysis reported in the
revised manuscript. It uses one canonical prediction table containing the
same 345 speaker-independent test instances for all three model configurations.

The analysis is descriptive. It compares performance on scripted and
drama-derived recordings for the five non-Neutral emotion classes represented
by both sources. Neutral is excluded because the BNSER corpus contains no
drama-derived Neutral recordings.
"""

from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Repository paths and analysis constants
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_FILE = REPO_ROOT / "results" / "test_predictions.csv"
OUTPUT_FILE = REPO_ROOT / "results" / "source_stratified_results.csv"

EXPECTED_TEST_SIZE = 345
EXPECTED_NEUTRAL = 50
EXPECTED_NON_NEUTRAL = 295
EXPECTED_SCRIPTED = 250
EXPECTED_DRAMA = 45

EMOTION_LABELS = {"angry", "disgust", "fear", "happy", "neutral", "sad"}
SOURCES = {"Scripted", "Drama-derived"}
MODELS = {
    "WavLM-FT": "model1_pred",
    "WavLM-FT+Aug": "model2_pred",
    "WavLM-PEFT": "model3_pred",
}

# Stable, anonymized speaker IDs used in the fixed test split.
EXPECTED_SPEAKER_SOURCE = {
    "A14": "Scripted",
    "A04": "Drama-derived",
    "A06": "Drama-derived",
    "A03": "Drama-derived",
}
EXPECTED_SPEAKER_COUNTS = {
    "A14": 300,
    "A04": 20,
    "A06": 19,
    "A03": 6,
}

# These are manuscript verification targets, not inputs to the calculation.
# The script computes the values from test_predictions.csv and checks them
# against the reported revision values before writing the result file.
EXPECTED_RESULTS = {
    "WavLM-FT": {
        "Scripted": (250, 200),
        "Drama-derived": (45, 22),
    },
    "WavLM-FT+Aug": {
        "Scripted": (250, 230),
        "Drama-derived": (45, 30),
    },
    "WavLM-PEFT": {
        "Scripted": (250, 232),
        "Drama-derived": (45, 29),
    },
}

REQUIRED_COLUMNS = {
    "sample_id",
    "speaker_id",
    "source",
    "y_true",
    "model1_pred",
    "model2_pred",
    "model3_pred",
}


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------


def load_predictions() -> pd.DataFrame:
    """Load and validate the canonical 345-instance test prediction table."""
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            "Canonical prediction file not found: "
            f"{PREDICTION_FILE}\n"
            "Create results/test_predictions.csv from the actual fixed "
            "speaker-independent test predictions before running this script."
        )

    df = pd.read_csv(PREDICTION_FILE)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "test_predictions.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SIZE} test instances; found {len(df)}."
        )

    required_values = [
        "sample_id",
        "speaker_id",
        "source",
        "y_true",
        "model1_pred",
        "model2_pred",
        "model3_pred",
    ]
    if df[required_values].isna().any().any():
        missing_by_column = df[required_values].isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]
        raise ValueError(
            "Missing values found in required prediction columns: "
            + ", ".join(
                f"{column}={count}" for column, count in missing_by_column.items()
            )
        )

    if df["sample_id"].astype(str).duplicated().any():
        duplicate_ids = (
            df.loc[df["sample_id"].astype(str).duplicated(keep=False), "sample_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "sample_id must be unique. Duplicate IDs include: "
            + ", ".join(duplicate_ids[:10])
        )

    # Normalize text fields so the validation and grouping are insensitive to
    # capitalization or accidental surrounding whitespace.
    text_columns = [
        "sample_id",
        "speaker_id",
        "source",
        "y_true",
        "model1_pred",
        "model2_pred",
        "model3_pred",
    ]
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    df["y_true"] = df["y_true"].str.lower()
    for column in ("model1_pred", "model2_pred", "model3_pred"):
        df[column] = df[column].str.lower()

    validate_dataset_structure(df)
    return df


def validate_dataset_structure(df: pd.DataFrame) -> None:
    """Check the fixed-test-set source and speaker structure."""
    if set(df["y_true"]) != EMOTION_LABELS:
        raise ValueError(
            "The test set must contain exactly the six BNSER emotion labels: "
            + ", ".join(sorted(EMOTION_LABELS))
        )

    unexpected_sources = set(df["source"]) - SOURCES
    if unexpected_sources:
        raise ValueError(
            "Unexpected source labels found: "
            + ", ".join(sorted(unexpected_sources))
        )

    unexpected_speakers = set(df["speaker_id"]) - set(EXPECTED_SPEAKER_SOURCE)
    if unexpected_speakers:
        raise ValueError(
            "Unexpected held-out speaker IDs found: "
            + ", ".join(sorted(unexpected_speakers))
        )

    for speaker, expected_source in EXPECTED_SPEAKER_SOURCE.items():
        observed = set(df.loc[df["speaker_id"] == speaker, "source"])
        if observed != {expected_source}:
            raise ValueError(
                f"Speaker {speaker} must be assigned only to {expected_source}; "
                f"observed source labels: {sorted(observed)}"
            )

    observed_speaker_counts = df["speaker_id"].value_counts().to_dict()
    for speaker, expected_count in EXPECTED_SPEAKER_COUNTS.items():
        observed_count = observed_speaker_counts.get(speaker, 0)
        if observed_count != expected_count:
            raise ValueError(
                f"Speaker {speaker}: expected {expected_count} test instances; "
                f"found {observed_count}."
            )

    full_source_counts = df["source"].value_counts().to_dict()
    if full_source_counts.get("Scripted", 0) != 300:
        raise ValueError("Expected 300 Scripted test instances in total.")
    if full_source_counts.get("Drama-derived", 0) != 45:
        raise ValueError("Expected 45 Drama-derived test instances in total.")

    if int(df["y_true"].eq("neutral").sum()) != EXPECTED_NEUTRAL:
        raise ValueError(
            f"Expected {EXPECTED_NEUTRAL} Neutral test instances."
        )

    neutral_sources = set(df.loc[df["y_true"] == "neutral", "source"])
    if neutral_sources != {"Scripted"}:
        raise ValueError(
            "All Neutral test instances must be from the Scripted source; "
            f"observed: {sorted(neutral_sources)}"
        )

    non_neutral = df[df["y_true"] != "neutral"]
    if len(non_neutral) != EXPECTED_NON_NEUTRAL:
        raise ValueError(
            f"Expected {EXPECTED_NON_NEUTRAL} non-Neutral test instances; "
            f"found {len(non_neutral)}."
        )

    source_counts = non_neutral["source"].value_counts().to_dict()
    if source_counts.get("Scripted", 0) != EXPECTED_SCRIPTED:
        raise ValueError(
            f"Expected {EXPECTED_SCRIPTED} non-Neutral Scripted instances; "
            f"found {source_counts.get('Scripted', 0)}."
        )
    if source_counts.get("Drama-derived", 0) != EXPECTED_DRAMA:
        raise ValueError(
            f"Expected {EXPECTED_DRAMA} non-Neutral Drama-derived instances; "
            f"found {source_counts.get('Drama-derived', 0)}."
        )

    # Predictions must use the same six-class label set as the ground truth.
    for model_name, prediction_column in MODELS.items():
        unexpected_predictions = set(df[prediction_column]) - EMOTION_LABELS
        if unexpected_predictions:
            raise ValueError(
                f"{model_name} contains unexpected prediction labels: "
                + ", ".join(sorted(unexpected_predictions))
            )


# -----------------------------------------------------------------------------
# Source-stratified calculation
# -----------------------------------------------------------------------------


def calculate_source_results(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate accuracy for each model and source on non-Neutral samples."""
    non_neutral = df.loc[df["y_true"] != "neutral"].copy()
    rows = []

    for model_name, prediction_column in MODELS.items():
        correct = non_neutral[prediction_column] == non_neutral["y_true"]

        source_summary = {}
        for source in ("Scripted", "Drama-derived"):
            subset = non_neutral.loc[non_neutral["source"] == source]
            n = len(subset)
            n_correct = int(correct.loc[subset.index].sum())
            source_summary[source] = {
                "n": n,
                "correct": n_correct,
                "accuracy": round(100.0 * n_correct / n, 2),
            }

        scripted = source_summary["Scripted"]
        drama = source_summary["Drama-derived"]

        rows.append(
            {
                "Configuration": model_name,
                "Scripted_N": scripted["n"],
                "Scripted_Correct": scripted["correct"],
                "Scripted_Accuracy_percent": scripted["accuracy"],
                "Drama_derived_N": drama["n"],
                "Drama_derived_Correct": drama["correct"],
                "Drama_derived_Accuracy_percent": drama["accuracy"],
                "Gap_percentage_points": round(
                    scripted["accuracy"] - drama["accuracy"], 2
                ),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Manuscript consistency check
# -----------------------------------------------------------------------------


def verify_manuscript_values(results: pd.DataFrame) -> None:
    """Verify that computed counts match the revised manuscript table."""
    for _, row in results.iterrows():
        model = row["Configuration"]
        observed = {
            "Scripted": (
                int(row["Scripted_N"]),
                int(row["Scripted_Correct"]),
            ),
            "Drama-derived": (
                int(row["Drama_derived_N"]),
                int(row["Drama_derived_Correct"]),
            ),
        }

        for source, value in observed.items():
            expected = EXPECTED_RESULTS[model][source]
            if value != expected:
                raise AssertionError(
                    f"{model} / {source}: computed {value}, "
                    f"but manuscript reports {expected}."
                )


def save_results(results: pd.DataFrame) -> None:
    """Write the validated source-stratified table to the repository results."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False, float_format="%.2f")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    df = load_predictions()
    results = calculate_source_results(df)
    verify_manuscript_values(results)
    save_results(results)

    print("Source-stratified analysis completed successfully.")
    print(f"Input:  {PREDICTION_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
