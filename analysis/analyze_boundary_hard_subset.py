#!/usr/bin/env python3
"""Analyze a real-response boundary-control hard subset behavior matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from plot_style import save_paper_heatmap


DEFAULT_MATRIX = Path("outputs/subsets/boundary_hard_real_subset_heuristic_behavior_matrix.csv")
DEFAULT_MANIFEST = Path("outputs/subsets/boundary_hard_real_subset_manifest.csv")
DEFAULT_OUTDIR = Path("figures/boundary_hard_real_subset")

BOUNDARY_COLUMNS = [
    "hallucination_control",
    "uncertainty_handling",
    "uncertainty",
    "nei_uncertainty_failure_proxy",
    "supports_status_failure_proxy",
    "refutes_correction_credit_proxy",
]


def normalize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "evaluator_variant" not in out.columns and "variant" in out.columns:
        out["evaluator_variant"] = out["variant"]
    if "target_model" not in out.columns and "model_name" in out.columns:
        out["target_model"] = out["model_name"]
    if "source_id" not in out.columns and "id" in out.columns:
        out["source_id"] = out["id"].astype(str).str.replace(r"::boundary_hard$", "", regex=True)
    for col in ["final_score"] + BOUNDARY_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_heatmap(df: pd.DataFrame, path: Path, title: str, cbar_label: str, fmt: str = ".2f") -> None:
    if df.empty:
        return
    values = df.astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    save_paper_heatmap(
        df,
        path,
        title,
        cbar_label,
        vmin=0,
        vmax=float(finite.max()) if finite.size else 1.0,
        fmt=fmt,
        pretty_columns=True,
        pretty_index=True,
        figsize=(max(8, 1.25 * len(df.columns) + 3), max(4, 0.55 * len(df.index) + 2.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior = normalize_behavior(pd.read_csv(args.behavior_matrix))
    manifest = pd.read_csv(args.manifest)
    merge_keys = ["target_model", "id"] if "target_model" in manifest.columns and "target_model" in behavior.columns else ["id"]
    merged = behavior.merge(
        manifest[[*merge_keys, "source_id", "case_categories", "selection_reason"]],
        on=merge_keys,
        how="left",
        suffixes=("", "_manifest"),
        validate="many_to_one",
    )
    if "source_id_manifest" in merged.columns:
        merged["source_id"] = merged["source_id"].fillna(merged["source_id_manifest"])
    merged.to_csv(args.output_dir / "boundary_hard_behavior_with_manifest.csv", index=False)

    metric_cols = [col for col in ["final_score"] + BOUNDARY_COLUMNS if col in merged.columns]
    summary = (
        merged.groupby(["evaluator_variant", "reference_label"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    summary["rows"] = merged.groupby(["evaluator_variant", "reference_label"], dropna=False).size().values
    summary.to_csv(args.output_dir / "boundary_hard_summary_by_variant_label.csv", index=False)

    if {"hallucination_control", "uncertainty_handling"} <= set(merged.columns):
        combo = merged.copy()
        combo["hc_uh_combo"] = (
            "HC="
            + combo["hallucination_control"].fillna(-1).astype(int).astype(str)
            + ", UH="
            + combo["uncertainty_handling"].fillna(-1).astype(int).astype(str)
        )
        combo_counts = combo.pivot_table(
            index="evaluator_variant",
            columns="hc_uh_combo",
            values="id",
            aggfunc="count",
            fill_value=0,
        )
        combo_counts.to_csv(args.output_dir / "boundary_hard_hc_uh_combo_counts.csv")
        save_heatmap(
            combo_counts,
            args.output_dir / "boundary_hard_hc_uh_combo_counts.png",
            "Boundary Hard Subset HC/UH Combinations",
            "case count",
            fmt=".0f",
        )

    if metric_cols:
        profile = merged.pivot_table(
            index="evaluator_variant",
            columns="reference_label",
            values=metric_cols,
            aggfunc="mean",
        )
        profile.to_csv(args.output_dir / "boundary_hard_metric_profile.csv")

    print(f"Wrote boundary hard-subset analysis to {args.output_dir}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
