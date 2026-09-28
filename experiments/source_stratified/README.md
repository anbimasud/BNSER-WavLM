# Source-Stratified Test Analysis

## Purpose

This directory contains the reproducible analysis for the source-stratified
comparison reported in the revised BNSER-WavLM manuscript. It addresses the
concern that the BNSER corpus contains both scripted and drama-derived speech.

The analysis is **descriptive**. It reports test-set accuracy separately for
scripted and drama-derived recordings for the five non-Neutral emotion classes
for which both source types are represented. It does not estimate a causal
effect of recording source, performance style, or recording condition.

## Evaluation scope

The analysis uses the fixed speaker-independent test set of 345 recordings:

- 300 Scripted recordings
- 45 Drama-derived recordings
- 50 Neutral recordings, all Scripted
- 295 non-Neutral recordings used for the source comparison
  - 250 Scripted
  - 45 Drama-derived

Neutral is excluded from the source comparison because the BNSER corpus
contains no drama-derived Neutral recordings. A Scripted-versus-Drama-derived
Neutral accuracy comparison is therefore not defined by the available data.

The four held-out test speakers are represented by the stable anonymized IDs
used in the manuscript/repository:

| Speaker ID | Source | Test recordings |
|---|---|---:|
| A14 | Scripted | 300 |
| A04 | Drama-derived | 20 |
| A06 | Drama-derived | 19 |
| A03 | Drama-derived | 6 |
| **Total** | | **345** |

## Input

The script reads the canonical prediction artifact:

```text
results/test_predictions.csv
```

The file must contain one row for each of the 345 fixed test instances and at
least these columns:

```text
sample_id
speaker_id
source
y_true
model1_pred
model2_pred
model3_pred
```

The three prediction columns correspond to:

- `model1_pred` — WavLM-FT
- `model2_pred` — WavLM-FT+Aug
- `model3_pred` — WavLM-PEFT

The `source` column is stored with the canonical prediction table so that the
analysis does not depend on model-specific evaluation directory names or on
reconstruction of archived misclassification logs.

No audio files are required by this analysis script.

## Validation

Before calculating the source-stratified results, the script checks:

- exactly 345 test instances are present;
- `sample_id` is unique and required fields contain no missing values;
- exactly six BNSER emotion labels are present;
- all prediction labels belong to the same six-class label set;
- the four expected held-out speaker IDs are present with the expected counts;
- each held-out speaker has one documented source assignment;
- the complete test set contains 300 Scripted and 45 Drama-derived recordings;
- exactly 50 Neutral test recordings are present and all are Scripted;
- the non-Neutral comparison set contains exactly 250 Scripted and 45 Drama-derived recordings; and
- the computed source-specific counts reproduce the values reported in the revised manuscript.

A validation failure stops execution rather than producing a potentially
misleading result file.

## Calculation

For each model, correctness is determined from the actual prediction and true
label in each row:

```text
correct = prediction == true label
```

Neutral rows are then excluded. Accuracy is calculated independently for the
Scripted and Drama-derived subsets:

```text
Accuracy = correct predictions / number of test instances × 100
```

The reported gap is:

```text
Gap = Scripted accuracy − Drama-derived accuracy
```

The gap is reported in percentage points (pp).

## Manuscript verification targets

The following values are the publication values that the script is expected to
reproduce from the canonical prediction artifact. They are **not hard-coded
inputs to the calculation**.

| Configuration | Scripted | Drama-derived | Gap |
|---|---:|---:|---:|
| WavLM-FT | 80.00% (200/250) | 48.89% (22/45) | 31.11 pp |
| WavLM-FT+Aug | 92.00% (230/250) | 66.67% (30/45) | 25.33 pp |
| WavLM-PEFT | 92.80% (232/250) | 64.44% (29/45) | 28.36 pp |

If the calculated counts differ from these revision values, the script raises
an error. This is intentional: a changed prediction artifact should not
silently overwrite the manuscript-consistent result.

## Output

The validated result is written to:

```text
results/source_stratified_results.csv
```

The output contains the sample counts, correct counts, source-specific
accuracies, and the scripted-minus-drama-derived gap for each configuration.

## Run

From the repository root:

```bash
python experiments/source_stratified/source_stratified_analysis.py
```

The script requires Python with `pandas` installed. The repository-level
`requirements.txt` provides the environment used by the project.

## Interpretation boundary

The source-stratified results show a source-associated performance difference
within the evaluated BNSER test corpus. They do **not** establish that source
or performance style itself causes the observed gap. Source type is entangled
with recording conditions, performance context, and speaker composition in the
present corpus.

Accordingly, the manuscript describes these results as **source-associated
performance differences**, not as a causal recording-style or source effect.

The analysis also does not establish population-level generalization beyond
the four held-out test speakers.

## Reproducibility design

The source analysis deliberately uses the same canonical test-prediction
artifact as the other fixed-test-set analyses. This keeps the evaluation unit
consistent across the repository and avoids reconstructing predictions from
model-specific archived error files.

The analysis does not require or upload raw speech recordings.
