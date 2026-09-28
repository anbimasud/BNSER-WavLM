# Prediction-Confidence Analysis

This analysis reports the mean prediction confidence for correct and incorrect predictions and their difference (`Confidence_Gap`). It is intentionally descriptive.

## Input

The default input is:

```text
results/test_predictions.csv
```

The file must contain 345 test samples and raw per-sample confidence values for all three models:

```text
model1_confidence
model2_confidence
model3_confidence
```

The same file must provide enough information to identify correct versus incorrect predictions, either through model-specific status fields or through `y_true` plus the three model prediction fields.

## Important reproducibility rule

The script does **not** reconstruct confidence values from manuscript summary statistics. If a model's per-instance confidence column is absent, execution stops with a clear error.

This is important for the current archived evidence: WavLM-FT confidence statistics are directly supported by archived evaluation evidence, whereas the previously reported WavLM-FT+Aug and WavLM-PEFT mean-confidence summaries were not accompanied by complete archived per-instance confidence values. Those values should therefore not be presented as newly recomputed by this script until the raw confidence predictions are supplied.

## Outputs

The script writes:

```text
results/confidence_summary.csv
```

## Interpretation boundary

A confidence gap is not equivalent to expected calibration error (ECE), a reliability diagram, Brier score, or a formal calibration assessment. This script computes none of those measures. It also does not infer a causal explanation for differences in confidence between model configurations.
