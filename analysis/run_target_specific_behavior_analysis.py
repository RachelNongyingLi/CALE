#!/usr/bin/env python3
"""Run target-specific robustness analysis for CALE behavior matrices.

This script keeps the main pooled analysis intact, then repeats the same
behavior-profile and PCA summaries within each target model. The goal is to
check whether the pooled latent structure is stable, or whether it is driven by
one target model's error distribution.

It does not regenerate model responses and does not require a GPU. Use it after
`experiment.py` has already exported a behavior matrix CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_behavior_matrix import (
    numeric_behavior_columns,
    run_pca,
    save_correlation_heatmap,
    save_loading_heatmap,
)
from visualize_behavior_matrix import (
    CORE_BEHAVIOR_COLUMNS,
    PROXY_COLUMNS,
    existing_numeric_columns,
    save_heatmap,
    save_missingness_plot,
    summarize_by,
)


CALE_VARIANTS = {
    "generic_cale",
    "attack_aware_cale",
    "full_attack_aware_cale",
}

TARGET_SPLITS = {
    "pooled_all_targets": None,
    "target_qwen25_15b_only": "qwen",
    "target_llama32_1b_only": "llama",
}


def target_mask(df: pd.DataFrame, target_key: str | None) -> pd.Series:
    """Return a boolean mask for a named target-model split."""
    if target_key is None:
        return pd.Series(True, index=df.index)
    if "model_name" not in df.columns:
        raise ValueError("Behavior matrix must include `model_name` for target-specific splits.")
    model_name = df["model_name"].astype(str).str.lower()
    if target_key == "qwen":
        return model_name.str.contains("qwen", na=False)
    if target_key == "llama":
        return model_name.str.contains("llama", na=False)
    raise ValueError(f"Unknown target split key: {target_key}")


def write_variant_difference_summary(profile: pd.DataFrame, output_path: Path) -> None:
    """Write simple CALE variant differences relative to generic CALE.

    These differences are descriptive deltas in observed behavior variables, not
    calibrated effect sizes or validated improvement thresholds.
    """
    rows = []
    if "generic_cale" not in profile.index:
        pd.DataFrame().to_csv(output_path)
        return
    generic = profile.loc["generic_cale"]
    for variant in ["attack_aware_cale", "full_attack_aware_cale"]:
        if variant not in profile.index:
            continue
        diff = profile.loc[variant] - generic
        diff.name = f"{variant}_minus_generic"
        rows.append(diff)
    if rows:
        pd.DataFrame(rows).to_csv(output_path)
    else:
        pd.DataFrame().to_csv(output_path)


def run_one_split(
    df: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    n_components: int,
    max_missing_share: float,
) -> dict[str, object]:
    """Run behavior-profile and CALE-only PCA analysis for one target split."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_dir / "behavior_matrix_subset.csv", index=False)

    behavior_columns = existing_numeric_columns(df, CORE_BEHAVIOR_COLUMNS)
    proxy_columns = existing_numeric_columns(df, PROXY_COLUMNS)
    summary_columns = behavior_columns + proxy_columns
    if not summary_columns:
        raise ValueError(f"No behavior columns found for split {split_name}.")

    save_missingness_plot(df, summary_columns, split_dir / "behavior_variable_availability.png")
    pd.DataFrame(
        {
            "column": summary_columns,
            "missing_share": df[summary_columns].isna().mean().values,
        }
    ).to_csv(split_dir / "behavior_variable_availability.csv", index=False)

    if "variant" in df.columns:
        profile = summarize_by(df, "variant", summary_columns)
        profile.to_csv(split_dir / "behavior_profile_by_variant.csv")
        save_heatmap(profile, split_dir / "behavior_profile_by_variant.png", f"{split_name}: Behavior Profile by Variant")
        write_variant_difference_summary(profile, split_dir / "variant_difference_summary.csv")
    else:
        profile = pd.DataFrame()

    if "variant" in df.columns:
        pca_df = df[df["variant"].isin(CALE_VARIANTS)].copy()
    else:
        pca_df = df.copy()
    pca_columns = numeric_behavior_columns(pca_df, include_final_score=False)
    selected = pca_df[pca_columns].apply(pd.to_numeric, errors="coerce")
    corr = selected.corr()
    corr.to_csv(split_dir / "cale_only_behavior_correlation_matrix.csv")
    save_correlation_heatmap(corr, split_dir / "cale_only_behavior_correlation_heatmap.png")

    loadings, variance, scores = run_pca(selected, n_components, max_missing_share)
    loadings.to_csv(split_dir / "cale_only_behavior_pca_loadings.csv")
    variance.to_csv(split_dir / "cale_only_behavior_pca_explained_variance.csv", index=False)
    scores.to_csv(split_dir / "cale_only_behavior_pca_scores.csv")
    save_loading_heatmap(loadings, split_dir / "cale_only_behavior_pca_loadings_heatmap.png")

    top_loading_rows = []
    for component in loadings.columns:
        ordered = loadings[component].sort_values(key=lambda s: s.abs(), ascending=False)
        for rank, (variable, loading) in enumerate(ordered.head(8).items(), start=1):
            top_loading_rows.append(
                {
                    "split": split_name,
                    "component": component,
                    "rank": rank,
                    "variable": variable,
                    "loading": loading,
                    "absolute_loading": abs(loading),
                }
            )
    pd.DataFrame(top_loading_rows).to_csv(split_dir / "cale_only_pca_top_loadings.csv", index=False)

    return {
        "split": split_name,
        "rows": len(df),
        "target_models": sorted(df["model_name"].dropna().astype(str).unique().tolist()) if "model_name" in df.columns else [],
        "variants": sorted(df["variant"].dropna().astype(str).unique().tolist()) if "variant" in df.columns else [],
        "pca_cumulative_variance": float(variance["explained_variance_ratio"].sum()),
        "pc1_top_variables": ", ".join(loadings["PC1"].abs().sort_values(ascending=False).head(4).index.tolist())
        if "PC1" in loadings.columns
        else "",
    }


def run_target_specific_analysis(
    input_path: Path,
    output_dir: Path,
    n_components: int = 4,
    max_missing_share: float = 0.5,
) -> pd.DataFrame:
    """Run pooled and target-specific robustness summaries."""
    df = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split_name, target_key in TARGET_SPLITS.items():
        split_df = df.loc[target_mask(df, target_key)].copy()
        if split_df.empty:
            continue
        summaries.append(run_one_split(split_df, split_name, output_dir, n_components, max_missing_share))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "target_specific_analysis_summary.csv", index=False)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pooled and target-specific CALE behavior analysis.")
    parser.add_argument("--input", required=True, help="Behavior matrix CSV exported by experiment.py.")
    parser.add_argument("--output-dir", required=True, help="Output directory for target-specific tables and figures.")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--max-missing-share", type=float, default=0.5)
    args = parser.parse_args()

    summary = run_target_specific_analysis(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        n_components=args.n_components,
        max_missing_share=args.max_missing_share,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
