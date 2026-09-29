# Inter-Annotator Cohen's Kappa

This script recomputes the three pairwise Cohen's kappa statistics directly from the annotation matrix supplied with the repository.

## Input

Default input:

```text
supplementary/merged_all_annotators.csv
```

Required annotation columns:

```text
Annotator_1, Annotator_2, Annotator_3
```

The supplied annotation matrix contains **2,300 annotation rows**. The script uses the rows actually present and explicitly warns if the row count differs from 2,300.

## Outputs

The script writes to the repository-level `results/` directory:

* `kappa_results.xlsx`

The mean is calculated from the full-precision pairwise kappa values and then rounded to three decimal places.

## Manuscript consistency

The resulting values should reproduce the revised manuscript's reported pairwise values (approximately 0.805, 0.802, and 0.831) and mean of approximately 0.813 when the supplied 2,300-sample annotation matrix is used.

The previously disputed **90.56% inter-annotator agreement** figure is intentionally not calculated or reported here. Cohen's kappa and raw percent agreement are different statistics; a percent-agreement value should only be reported if a separately defined and reproducible analysis is provided.
