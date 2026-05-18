#!/usr/bin/env python3
"""Create paper-facing visualizations for CALE behavior matrices.

These plots are designed for the "evaluator as measurement object" story. They
visualize continuous construct subscores and explicitly named behavior proxies.
They do not turn quality tiers into factual-status labels, and they do not add
new pass/fail thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_BEHAVIOR_COLUMNS = [
    "misinformation_detection",
    "framing_resistance",
    "claim_status_recognition",
    "error_rejection",
    "correction_accuracy",
    "evidence_grounding",
    "source_faithfulness",
    "hallucination_control",
    "uncertainty_handling",
]

PROXY_COLUMNS = [
    "nei_uncertainty_failure_proxy",
    "refutes_correction_credit_proxy",
    "supports_status_failure_proxy",
]


def existing_numeric_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    columns = []
    for column in candidates:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


def pretty_label(name: str) -> str:
    return name.replace("_proxy", "").replace("_", " ").title()


def save_heatmap(
    table: pd.DataFrame,
    path: Path,
    title: str,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
) -> None:
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(table.columns)), max(4.8, 0.42 * len(table.index))))
    image = ax.imshow(table.values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([pretty_label(c) for c in table.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=9)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.62 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def save_missingness_plot(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = df[columns].isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3.6, 0.32 * len(missing))))
    ax.barh([pretty_label(c) for c in missing.index], missing.values, color="#6B7280")
    ax.set_xlim(0, 1)
    ax.set_xlabel("missing share")
    ax.set_title("Behavior Variable Availability")
    for i, value in enumerate(missing.values):
        ax.text(value + 0.01, i, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def summarize_by(df: pd.DataFrame, group: str, columns: list[str]) -> pd.DataFrame:
    numeric = df.copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.groupby(group, dropna=False)[columns].mean().dropna(how="all")


def summarize_by_two_keys(df: pd.DataFrame, row_key: str, col_key: str, value: str) -> pd.DataFrame:
    numeric = df.copy()
    numeric[value] = pd.to_numeric(numeric[value], errors="coerce")
    return numeric.pivot_table(index=row_key, columns=col_key, values=value, aggfunc="mean").dropna(how="all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CALE behavior matrix.")
    parser.add_argument("--input", required=True, help="Behavior matrix CSV exported by experiment.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-final-score",
        action="store_true",
        help="Add final_score to summary heatmaps. Off by default because it is already an aggregate.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    behavior_columns = existing_numeric_columns(df, CORE_BEHAVIOR_COLUMNS)
    proxy_columns = existing_numeric_columns(df, PROXY_COLUMNS)
    summary_columns = behavior_columns + proxy_columns
    if args.include_final_score and "final_score" in df.columns:
        summary_columns = ["final_score"] + summary_columns
    if not summary_columns:
        raise ValueError("No behavior columns found. Check that experiment.py exported behavior_matrix or behavior CSV.")

    save_missingness_plot(df, summary_columns, output_dir / "behavior_variable_availability.png")
    pd.DataFrame({"column": summary_columns, "missing_share": df[summary_columns].isna().mean().values}).to_csv(
        output_dir / "behavior_variable_availability.csv", index=False
    )

    if "variant" in df.columns:
        by_variant = summarize_by(df, "variant", summary_columns)
        by_variant.to_csv(output_dir / "behavior_profile_by_variant.csv")
        save_heatmap(by_variant, output_dir / "behavior_profile_by_variant.png", "Mean Behavior Profile by Evaluator Variant")

    if "framing_style" in df.columns:
        by_framing = summarize_by(df, "framing_style", summary_columns)
        by_framing.to_csv(output_dir / "behavior_profile_by_framing.csv")
        save_heatmap(by_framing, output_dir / "behavior_profile_by_framing.png", "Mean Behavior Profile by Framing")

    if {"variant", "framing_style"}.issubset(df.columns):
        for proxy in proxy_columns:
            proxy_table = summarize_by_two_keys(df, "variant", "framing_style", proxy)
            proxy_table.to_csv(output_dir / f"{proxy}_by_variant_framing.csv")
            save_heatmap(
                proxy_table,
                output_dir / f"{proxy}_by_variant_framing.png",
                f"{pretty_label(proxy)} by Variant and Framing",
                cmap="magma" if "failure" in proxy else "Blues",
            )

    (output_dir / "visualized_behavior_columns.txt").write_text("\n".join(summary_columns) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
