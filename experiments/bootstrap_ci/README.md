# Bootstrap Confidence Intervals

This directory contains the bootstrap analysis used to quantify uncertainty for the three BNSER-WavLM configurations on the fixed speaker-independent test set.

The analysis is intentionally kept as a single script so that the reviewer-facing repository remains simple and easy to inspect.

## Scope

The script reproduces the manuscript's uncertainty-analysis protocol:

- fixed speaker-independent test set: **345 utterances**;
- **10,000** non-parametric bootstrap resamples with replacement;
- **95% percentile confidence intervals** using the 2.5th and 97.5th percentiles;
- Accuracy, Macro-F1, Weighted F1, and Balanced Accuracy for each configuration;
- paired bootstrap confidence intervals for **Accuracy differences** between configurations evaluated on the same test instances.

Macro-AUC is not bootstrapped in the manuscript and is therefore not calculated by this script.

The three model configurations are:

- **WavLM-FT** — full fine-tuning;
- **WavLM-FT+Aug** — full fine-tuning with training-only augmentation;
- **WavLM-PEFT** — the combined PEFT configuration reported in the manuscript.

## Required input

The script reads:

```text
results/test_predictions.csv
```

The file must contain exactly one row for each of the 345 test utterances and these columns:

```text
sample_id,y_true,model1_pred,model2_pred,model3_pred
```

`sample_id` must be unique. The prediction columns must correspond to the same test utterances, in the same evaluation set, and `y_true` is the common ground-truth label for those utterances.

The script checks the row count, required columns, duplicate IDs, missing values, six-class ground truth, and prediction-label compatibility before running the bootstrap analysis. It stops with an error if these checks fail.

The prediction table is a model-output artifact rather than the speech dataset itself. No audio files are required by this analysis.

## Method

For each model, bootstrap samples are drawn from the fixed 345 test instances with replacement. For every resample, Accuracy, Macro-F1, Weighted F1, and Balanced Accuracy are recomputed. The 2.5th and 97.5th percentiles of the 10,000 bootstrap estimates form the 95% percentile confidence interval.

For paired comparisons, the same bootstrap indices are applied to both models. The reported difference is defined as:

```text
Accuracy(comparison model) - Accuracy(reference model)
```

The paired comparisons are:

```text
WavLM-FT+Aug - WavLM-FT
WavLM-PEFT - WavLM-FT+Aug
WavLM-PEFT - WavLM-FT
```

These differences are reported in **percentage points (pp)**.

A fixed random seed of 42 is used so that the analysis is deterministic. The bootstrap intervals describe uncertainty conditional on the evaluated test set and the reported trained configurations. They do not quantify variability across alternative speaker partitions, repeated training runs, or a broader population of speakers.

## Run

From the repository root:

```bash
python experiments/bootstrap_ci/bootstrap_ci.py
```

The script writes:

```text
results/bootstrap_results.csv
```

The output table contains the model/comparison name, metric, test-set size, point estimate, lower and upper confidence limits, number of bootstrap replicates, confidence level, method, and seed. Metric estimates and confidence limits are expressed as percentages; paired Accuracy differences are expressed as percentage-point differences.

## Manuscript verification targets

The following values are the reported values in the revised manuscript and can be used to verify the prediction artifact and bootstrap implementation. They are **checking targets only**; the script does not read or hard-code these values.

| Model | Metric | Point estimate | 95% CI |
|---|---|---:|---:|
| WavLM-FT | Accuracy | 75.07% | 70.43–79.42% |
| WavLM-FT | Weighted F1 | 75.68% | 71.18–80.01% |
| WavLM-FT | Balanced Accuracy | 75.31% | 70.97–79.51% |
| WavLM-FT+Aug | Accuracy | 82.61% | 78.55–86.38% |
| WavLM-FT+Aug | Weighted F1 | 82.57% | 78.53–86.44% |
| WavLM-FT+Aug | Balanced Accuracy | 81.84% | 78.00–85.59% |
| WavLM-PEFT | Accuracy | 88.70% | 85.22–91.88% |
| WavLM-PEFT | Weighted F1 | 88.72% | 85.30–91.88% |
| WavLM-PEFT | Balanced Accuracy | 88.81% | 85.41–91.95% |

Reported paired Accuracy differences:

| Comparison | Observed difference | 95% CI |
|---|---:|---:|
| WavLM-FT+Aug − WavLM-FT | +7.54 pp | +2.90 to +12.46 pp |
| WavLM-PEFT − WavLM-FT+Aug | +6.09 pp | +1.74 to +10.43 pp |
| WavLM-PEFT − WavLM-FT | +13.62 pp | +8.70 to +18.55 pp |

Small differences in the final displayed decimal places can occur if the prediction artifact, software versions, or numerical environment differs from the archived analysis. The reported manuscript values remain the publication reference values.

## Reproducibility boundary

This analysis is conditional on one fixed speaker-independent partition and one trained instance of each configuration. Therefore, the bootstrap intervals should be interpreted as uncertainty for the evaluated test set, not as uncertainty across alternative speaker splits or independent training runs.

The bootstrap analysis also does not establish population-level generalization beyond the evaluated BNSER corpus and held-out speakers.
