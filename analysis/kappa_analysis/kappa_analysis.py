"""Recompute pairwise Cohen's kappa from the supplied annotation matrix.

Default input:
    <repo_root>/supplementary/merged_annotations.csv

The analysis uses the complete annotation rows present in the supplied matrix.
It does not hard-code the manuscript's kappa values and does not calculate or
report the previously disputed 90.56% agreement figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ANNOTATORS = ["Annotator_1", "Annotator_2", "Annotator_3"]
EXPECTED_ANNOTATION_ROWS = 2200


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotation matrix not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Annotation matrix is empty.")

    resolved = []
    for annotator in ANNOTATORS:
        col = find_column(df, [annotator, annotator.replace("_", " ")])
        if col is None:
            raise ValueError(
                f"Missing annotation column for {annotator}. Available columns: {list(df.columns)}"
            )
        resolved.append(col)

    out = df[resolved].copy()
    out.columns = ANNOTATORS
    for col in ANNOTATORS:
        out[col] = out[col].astype(str).str.strip().str.lower()
        if out[col].eq("").any() or out[col].eq("nan").any():
            raise ValueError(f"Missing annotation labels found in {col}.")

    if len(out) != EXPECTED_ANNOTATION_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_ANNOTATION_ROWS} complete annotation rows; "
            f"found {len(out)}."
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--annotations", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    annotations_path = (args.annotations or (repo_root / "supplementary" / "merged_annotations.csv")).resolve()
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(annotations_path)

    pairs = [("Annotator 1 vs Annotator 2", "Annotator_1", "Annotator_2"),
             ("Annotator 1 vs Annotator 3", "Annotator_1", "Annotator_3"),
             ("Annotator 2 vs Annotator 3", "Annotator_2", "Annotator_3")]

    rows = []
    for pair_name, left, right in pairs:
        kappa = float(cohen_kappa_score(df[left], df[right]))
        rows.append({
            "Annotator_Pair": pair_name,
            "Cohen_kappa": kappa,
            "N_annotations": len(df),
        })

    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(results_dir / "pairwise_kappa.csv", index=False)

    exact_mean = float(pairwise["Cohen_kappa"].mean())
    mean_df = pd.DataFrame([{
        "Statistic": "Mean of pairwise Cohen_kappa",
        "Exact_Mean": exact_mean,
        "Rounded_Mean": round(exact_mean, 3),
        "N_annotations": len(df),
    }])
    mean_df.to_csv(results_dir / "kappa_mean.csv", index=False)

    print("Cohen's kappa analysis completed.")
    print(f"Annotation rows used: {len(df)}")
    print(pairwise.to_string(index=False))
    print(f"Mean pairwise kappa: {exact_mean:.6f} (rounded: {exact_mean:.3f})")
    print("No percent-agreement statistic is computed by this script.")


if __name__ == "__main__":
    main()
