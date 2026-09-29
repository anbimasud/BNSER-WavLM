# Inter-Annotator Cohen's Kappa

This script recomputes the three pairwise Cohen's kappa statistics directly from the annotation matrix supplied with the repository.

## Input

Default input:

```text
supplementary/merged_annotations.csv
```

Required annotation columns:

```text
Annotator_1, Annotator_2, Annotator_3
```

contains the 2,200 samples with complete labels from all three annotators used for the reproducible Cohen's κ analysis. The full BNSER corpus contains 2,300 samples; therefore, the annotation-agreement analysis sample count (2,200) should not be interpreted as the corpus size.

## Outputs

The script writes:
- results/pairwise_kappa.csv
- results/kappa_mean.csv

The repository also retains the archived
results/kappa_results.xlsx workbook as supporting evidence.

The mean is calculated from the full-precision pairwise kappa values and then rounded to three decimal places.

## Manuscript consistency

The resulting values should reproduce the revised manuscript's reported pairwise values (approximately 0.805, 0.802, and 0.831) and mean of approximately 0.813 when the supplied 2,200-row complete annotation matrix is used.

The previously disputed **90.56% inter-annotator agreement** figure is intentionally not calculated or reported here. Cohen's kappa and raw percent agreement are different statistics; a percent-agreement value should only be reported if a separately defined and reproducible analysis is provided.
