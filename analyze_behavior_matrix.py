#!/usr/bin/env python3
"""Explore latent organization in CALE evaluator behavior.

This script is intentionally exploratory. It does not assign pass/fail grades
or infer factual-status labels from final scores. It reads the behavior matrix
exported by experiment.py and analyzes numeric behavior variables such as
construct subscores, uncertainty, score shifts, and explicitly named proxies.

Recommended workflow:
  1. Run experiment.py with --behavior-matrix-output.
  2. Run this script on that CSV.
  3. Interpret correlation/PCA outputs as measurement-procedure-specific
     evidence, not as proof of stable psychological traits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "final_score",
}
"""Columns excluded from PCA by default.

`final_score` is a composite of the construct subscores, so including it in the
same PCA would partly re-analyze an already aggregated score. Use
--include-final-score only when you explicitly want to inspect how the composite
score aligns with the behavior dimensions.
"""


METADATA_COLUMNS = {
    "id",
    "model_name",
    "variant",
    "score_variant",
    "framing_style",
    "reference_label",
    "dataset",
    "dataset_role",
    "evaluation_setting",
    "domain",
    "risk_level",
    "quality_label",
}


def numeric_behavior_columns(df: pd.DataFrame, include_final_score: bool) -> list[str]:
    excluded = set(METADATA_COLUMNS)
    if not include_final_score:
        excluded.update(DEFAULT_EXCLUDE_COLUMNS)
    columns = []
    for column in df.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() > 1 and values.nunique(dropna=True) > 1:
            columns.append(column)
    return columns


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - means) / stds


def run_pca(values: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute PCA with SVD on z-scored behavior variables.

    Component signs are arbitrary in PCA. Interpret loadings by magnitude and
    pattern, not by assuming that a positive sign has intrinsic meaning.
    """
    matrix = zscore(values).dropna(axis=0, how="any")
    if matrix.empty:
        raise ValueError("No complete rows are available for PCA after dropping missing values.")
    u, singular_values, vt = np.linalg.svd(matrix.to_numpy(), full_matrices=False)
    max_components = min(n_components, vt.shape[0])
    components = vt[:max_components]
    scores = u[:, :max_components] * singular_values[:max_components]
    explained = (singular_values ** 2) / (len(matrix) - 1)
    explained_ratio = explained / explained.sum()
    component_names = [f"PC{i + 1}" for i in range(max_components)]
    loadings = pd.DataFrame(components.T, index=matrix.columns, columns=component_names)
    variance = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": explained_ratio[:max_components],
        }
    )
    score_df = pd.DataFrame(scores, index=matrix.index, columns=component_names)
    return loadings, variance, score_df


def save_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(corr.columns)), max(6, 0.55 * len(corr.index))))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Behavior Variable Correlations")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_loading_heatmap(loadings: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(loadings.columns)), max(6, 0.45 * len(loadings.index))))
    image = ax.imshow(loadings.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(loadings.columns)))
    ax.set_xticklabels(loadings.columns)
    ax.set_yticks(range(len(loadings.index)))
    ax.set_yticklabels(loadings.index, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PCA Loadings")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CALE behavior matrix.")
    parser.add_argument("--input", required=True, help="Behavior matrix CSV exported by experiment.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and PNG outputs.")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--include-final-score", action="store_true")
    parser.add_argument(
        "--group-by",
        choices=["none", "framing_style", "variant"],
        default="none",
        help="Optionally repeat correlation/PCA within a grouping variable.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    columns = numeric_behavior_columns(df, include_final_score=args.include_final_score)
    if not columns:
        raise ValueError(
            "No numeric behavior columns with nonzero variation were found. "
            "This often happens on a tiny smoke-test matrix; rerun on a larger "
            "experiment report, or pass --include-final-score only for debugging."
        )

    selected = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = selected.corr()
    corr.to_csv(output_dir / "behavior_correlation_matrix.csv")
    save_correlation_heatmap(corr, output_dir / "behavior_correlation_heatmap.png")

    loadings, variance, score_df = run_pca(selected, args.n_components)
    loadings.to_csv(output_dir / "behavior_pca_loadings.csv")
    variance.to_csv(output_dir / "behavior_pca_explained_variance.csv", index=False)
    score_df.to_csv(output_dir / "behavior_pca_scores.csv", index=True)
    save_loading_heatmap(loadings, output_dir / "behavior_pca_loadings_heatmap.png")

    if args.group_by != "none":
        for group_value, group_df in df.groupby(args.group_by, dropna=False):
            safe_value = str(group_value).replace("/", "_").replace(" ", "_") or "missing"
            group_dir = output_dir / f"{args.group_by}_{safe_value}"
            group_dir.mkdir(parents=True, exist_ok=True)
            group_selected = group_df[columns].apply(pd.to_numeric, errors="coerce")
            group_corr = group_selected.corr()
            group_corr.to_csv(group_dir / "behavior_correlation_matrix.csv")
            save_correlation_heatmap(group_corr, group_dir / "behavior_correlation_heatmap.png")
            try:
                group_loadings, group_variance, _ = run_pca(group_selected, args.n_components)
            except ValueError:
                continue
            group_loadings.to_csv(group_dir / "behavior_pca_loadings.csv")
            group_variance.to_csv(group_dir / "behavior_pca_explained_variance.csv", index=False)
            save_loading_heatmap(group_loadings, group_dir / "behavior_pca_loadings_heatmap.png")

    (output_dir / "behavior_columns.txt").write_text("\n".join(columns) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
