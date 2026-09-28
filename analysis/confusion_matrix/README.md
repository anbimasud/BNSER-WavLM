# Confusion-Matrix Analysis

This script generates the three model-specific confusion matrices and a compact summary of the observed test-set errors.

## Input

The default input is:

```text
results/test_predictions.csv
```

The file must contain one row for each of the 345 held-out test samples and, at minimum:

```text
sample_id, y_true, model1_pred, model2_pred, model3_pred
```

Column-name matching is case-insensitive and tolerant of common separators. The script validates the six emotion labels and rejects duplicate sample identifiers or an unexpected test-set size.

## Outputs

The script writes to the repository-level `results/` directory:

- `model1_confusion_matrix.csv`
- `model2_confusion_matrix.csv`
- `model3_confusion_matrix.csv`
- `confusion_matrix_summary.csv`
- `top_confusions.csv`

Rows are true classes and columns are predicted classes. Class order is:

`angry, disgust, fear, happy, neutral, sad`

## Interpretation boundary

The analysis is descriptive. A recurring confusion does not establish why the error occurs. In particular, the Model-3 configuration combines augmentation, class weighting, gradient accumulation, and layer freezing; therefore these matrices must not be used to attribute a particular error reduction to one component in isolation.

The script does not reconstruct or hard-code the manuscript's reported confusion counts. All counts are calculated directly from the supplied prediction file.
