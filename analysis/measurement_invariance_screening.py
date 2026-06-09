#!/usr/bin/env python3
"""Screen measurement-structure stability across backends and target splits.

This is not formal multi-group CFA invariance. It is a PCA-loading congruence
screening step for deciding whether the Full CALE construct structure looks
stable enough to discuss as preliminary internal-structure evidence.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from plot_style import save_paper_heatmap


DEFAULT_MATRIX = Path("figures/global_evaluator_audit/combined_available_behavior_matrix.csv")
DEFAULT_OUTDIR = Path("figures/measurement_invariance_screening")
FULL_CALE = "full_attack_aware_cale"

CONSTRUCT_COLUMNS = [
    "misinformation_detection",
    "framing_resistance",
    "claim_status_recognition",
    "error_rejection",
    "correction_accuracy",
    "evidence_grounding",
    "source_faithfulness",
    "hallucination_control",
    "uncertainty_handling",
    "uncertainty",
]


def short_backend(label: object) -> str:
    text = str(label)
    lower = text.lower()
    if "rule-based" in lower or "heuristic" in lower:
        return "heuristic"
    if "deepseek" in lower:
        return "deepseek_v4"
    if "qwen3" in lower:
        return "qwen3_14b"
    if "qwen2.5-7b" in lower:
        return "qwen25_7b"
    if "gemma" in lower:
        return "gemma3_12b"
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_") or "unknown"


def target_split(model: object) -> str:
    text = str(model)
    if "Qwen/Qwen2.5-1.5B" in text:
        return "target_qwen_only"
    if "meta-llama/Llama-3.2-1B" in text:
        return "target_llama_only"
    return "target_unknown"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "evaluator_variant" not in out.columns and "variant" in out.columns:
        out["evaluator_variant"] = out["variant"]
    if "target_model" not in out.columns and "model_name" in out.columns:
        out["target_model"] = out["model_name"]
    if "evaluator_backend_label" not in out.columns:
        out["evaluator_backend_label"] = out.get("evaluator_backend_model", "unknown_backend")
    if "target_split" not in out.columns:
        out["target_split"] = out["target_model"].map(target_split)
    out["backend_short"] = out["evaluator_backend_label"].map(short_backend)
    for col in CONSTRUCT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def pca_first_component(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, float, int, list[str]]:
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    valid_cols = [col for col in cols if x[col].notna().any() and x[col].std(skipna=True) > 0]
    if len(valid_cols) < 2:
        raise ValueError("Need at least two nonconstant construct columns for PCA.")
    x = x[valid_cols].copy()
    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0, np.nan)
    z = ((x - means) / stds).fillna(0.0).to_numpy(dtype=float)
    if z.shape[0] < 2:
        raise ValueError("Need at least two rows for PCA.")
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    component = vt[0]
    variance_ratio = float((s[0] ** 2) / np.sum(s**2))
    return component, variance_ratio, z.shape[0], valid_cols


def congruence(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denom) if denom else np.nan


def save_heatmap(df: pd.DataFrame, path: Path, title: str, cbar_label: str, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    if df.empty:
        return
    save_paper_heatmap(
        df,
        path,
        title,
        cbar_label,
        vmin=vmin,
        vmax=vmax,
        diverging=(vmin is not None and vmax is not None and vmin < 0 < vmax),
        pretty_columns=True,
        pretty_index=True,
        figsize=(max(8, 1.2 * len(df.columns) + 3), max(4, 0.45 * len(df.index) + 2.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--variant", default=FULL_CALE)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = normalize(pd.read_csv(args.behavior_matrix))
    data = data[data["evaluator_variant"].eq(args.variant)].copy()
    cols = [col for col in CONSTRUCT_COLUMNS if col in data.columns]
    if len(cols) < 2:
        raise ValueError("No usable construct columns found.")

    ref_component, ref_variance, ref_rows, ref_cols = pca_first_component(data, cols)
    rows = [
        {
            "group_id": "all_backends_pooled",
            "backend_short": "all",
            "evaluator_backend_label": "All available backends",
            "target_split": "pooled_qwen_llama",
            "rows": ref_rows,
            "pc1_variance": ref_variance,
            "columns_used": ", ".join(ref_cols),
            **{col: ref_component[ref_cols.index(col)] if col in ref_cols else np.nan for col in cols},
        }
    ]

    group_specs: list[tuple[str, pd.DataFrame]] = []
    for backend, group in data.groupby("evaluator_backend_label", dropna=False):
        group_specs.append((f"{short_backend(backend)}::pooled_qwen_llama", group))
        for split, split_group in group.groupby("target_split", dropna=False):
            group_specs.append((f"{short_backend(backend)}::{split}", split_group))

    for group_id, group in group_specs:
        try:
            component, variance, n_rows, used_cols = pca_first_component(group, cols)
        except ValueError as exc:
            rows.append(
                {
                    "group_id": group_id,
                    "backend_short": group["backend_short"].iloc[0] if "backend_short" in group else "",
                    "evaluator_backend_label": group["evaluator_backend_label"].iloc[0],
                    "target_split": group["target_split"].iloc[0] if group["target_split"].nunique() == 1 else "pooled_qwen_llama",
                    "rows": len(group),
                    "pc1_variance": np.nan,
                    "columns_used": "",
                    "error": str(exc),
                }
            )
            continue
        aligned_ref = np.array([ref_component[ref_cols.index(col)] if col in ref_cols else 0 for col in used_cols])
        if np.dot(component, aligned_ref) < 0:
            component = -component
        row = {
            "group_id": group_id,
            "backend_short": group["backend_short"].iloc[0],
            "evaluator_backend_label": group["evaluator_backend_label"].iloc[0],
            "target_split": group["target_split"].iloc[0] if group["target_split"].nunique() == 1 else "pooled_qwen_llama",
            "rows": n_rows,
            "pc1_variance": variance,
            "columns_used": ", ".join(used_cols),
            "tucker_phi_to_all_pooled": congruence(component, aligned_ref),
            "error": "",
        }
        for col in cols:
            row[col] = component[used_cols.index(col)] if col in used_cols else np.nan
        rows.append(row)

    profiles = pd.DataFrame(rows)
    profiles.to_csv(args.output_dir / "pca_loading_profiles.csv", index=False)

    summary_cols = [
        "group_id",
        "backend_short",
        "evaluator_backend_label",
        "target_split",
        "rows",
        "pc1_variance",
        "tucker_phi_to_all_pooled",
        "columns_used",
        "error",
    ]
    summary = profiles[[col for col in summary_cols if col in profiles.columns]].copy()
    summary.to_csv(args.output_dir / "pca_structure_summary.csv", index=False)

    loading_view = profiles.set_index("group_id")[cols]
    save_heatmap(
        loading_view,
        args.output_dir / "pc1_loading_profiles_heatmap.png",
        "Full CALE PC1 Loading Profiles by Backend and Target Split",
        "aligned PC1 loading",
        cmap="coolwarm",
        vmin=-0.8,
        vmax=0.8,
    )

    phi_view = summary[summary["group_id"].ne("all_backends_pooled")].pivot_table(
        index="backend_short",
        columns="target_split",
        values="tucker_phi_to_all_pooled",
        aggfunc="mean",
    )
    save_heatmap(
        phi_view,
        args.output_dir / "loading_congruence_to_all_pooled_heatmap.png",
        "PC1 Loading Congruence to All-Backend Pooled Reference",
        "Tucker phi",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    split_rows = []
    for backend, group in profiles[profiles["target_split"].isin(["target_qwen_only", "target_llama_only"])].groupby("backend_short"):
        by_split = group.set_index("target_split")
        if {"target_qwen_only", "target_llama_only"} <= set(by_split.index):
            diff = by_split.loc["target_qwen_only", cols].astype(float) - by_split.loc["target_llama_only", cols].astype(float)
            split_rows.append(
                {
                    "backend_short": backend,
                    "max_abs_loading_delta": float(diff.abs().max()),
                    "mean_abs_loading_delta": float(diff.abs().mean()),
                    "largest_drift_construct": str(diff.abs().sort_values(ascending=False).index[0]),
                    **{f"delta_{col}": diff[col] for col in cols},
                }
            )
    drift = pd.DataFrame(split_rows)
    drift.to_csv(args.output_dir / "target_split_loading_drift.csv", index=False)

    print(f"Wrote measurement-invariance screening outputs to {args.output_dir}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
