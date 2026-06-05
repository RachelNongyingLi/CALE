#!/usr/bin/env python3
"""Rebuild paper-facing CALE heatmaps from saved CSV summaries.

This script is intentionally layout-only: it does not recompute experiment
statistics. It restyles existing paper-ready summary tables into consistent,
print-friendly heatmaps and copies the exported PNGs into a repository-local
appendix figure directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from plot_style import save_paper_heatmap


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "figures" / "global_evaluator_audit"
TARGETED = ROOT / "figures" / "qwen3_14b_targeted_validity_summary"
PAPER_FIGURE_DIR = ROOT / "figures" / "paper_appendix"


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
]

VARIANT_LABELS = {
    "direct_trustllm_heuristic": "Direct TrustLLM heuristic",
    "direct_llm_judge": "Direct LLM judge",
    "generic_cale": "Generic CALE",
    "attack_aware_cale": "Attack-Aware CALE",
    "full_attack_aware_cale": "Full Attack-Aware CALE",
}

FRAMING_LABELS = {
    "assertive_minus_neutral": "Assertive - neutral",
    "authoritative_minus_neutral": "Authoritative - neutral",
}


def _copy_to_paper(path: Path, paper_name: str | None = None) -> None:
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, PAPER_FIGURE_DIR / (paper_name or path.name))


def _row_label(*parts: object) -> str:
    return "\n".join(str(part) for part in parts if pd.notna(part) and str(part))


def _variant_label(value: object) -> str:
    return VARIANT_LABELS.get(str(value), str(value).replace("_", " ").title())


def rebuild_backend_agreement() -> Path:
    data = pd.read_csv(AUDIT / "paper_backend_agreement_heatmap.csv").set_index("pair")
    path = AUDIT / "paper_backend_agreement_heatmap.png"
    save_paper_heatmap(
        data,
        path,
        "Backend agreement on shared Full-CALE rows",
        "Spearman rank correlation",
        vmin=-1,
        vmax=1,
        diverging=True,
        fmt=".2f",
        pretty_columns=False,
        pretty_index=False,
        figsize=(8.8, 3.4),
        xrotation=0,
        cbar_ticks=[-1, -0.5, 0, 0.5, 1],
    )
    _copy_to_paper(path)
    return path


def rebuild_pc1_loading() -> Path:
    data = pd.read_csv(AUDIT / "paper_pc1_loading_heatmap_data.csv")
    view = data.set_index("label")[[col for col in CONSTRUCT_COLUMNS + ["uncertainty"] if col in data.columns]]
    path = AUDIT / "paper_pc1_loading_heatmap.png"
    save_paper_heatmap(
        view,
        path,
        "PC1 loading profiles under Full Attack-Aware CALE",
        "absolute PC1 loading",
        vmin=0,
        vmax=max(0.6, float(view.max().max())),
        fmt=".2f",
        pretty_columns=True,
        pretty_index=False,
        figsize=(11.2, 6.6),
        xrotation=35,
    )
    _copy_to_paper(path)
    return path


def rebuild_target_split_sensitivity() -> Path:
    data = pd.read_csv(AUDIT / "paper_target_split_sensitivity_heatmap.csv").set_index("backend")
    path = AUDIT / "paper_target_split_sensitivity_heatmap.png"
    save_paper_heatmap(
        data,
        path,
        "Qwen-minus-Llama Full-CALE construct sensitivity",
        "Qwen target minus Llama target",
        diverging=True,
        fmt=".2f",
        pretty_columns=False,
        pretty_index=False,
        figsize=(9.2, 3.6),
        xrotation=30,
    )
    _copy_to_paper(path)
    return path


def rebuild_goal2_compact_profile() -> Path:
    data = pd.read_csv(AUDIT / "paper_table_compact_capability_core.csv")
    data["row_label"] = data.apply(lambda row: _row_label(row["backend_short_paper"], row["target_model_paper"]), axis=1)
    columns = ["factual_handling_mean", "resistance_boundary_mean", *[c for c in CONSTRUCT_COLUMNS if c in data.columns]]
    view = data.set_index("row_label")[columns]
    view = view.rename(
        columns={
            "factual_handling_mean": "Factual handling",
            "resistance_boundary_mean": "Resistance/boundary",
        }
    )
    path = AUDIT / "paper_goal2_compact_target_capability_heatmap.png"
    save_paper_heatmap(
        view,
        path,
        "Compact target-response diagnostic profile",
        "mean construct signal",
        vmin=0,
        vmax=1,
        fmt=".2f",
        pretty_columns=True,
        pretty_index=False,
        figsize=(11.8, 6.4),
        xrotation=35,
    )
    _copy_to_paper(path)
    _copy_to_paper(path, "paper_goal2_compact_target_response_heatmap.png")
    return path


def rebuild_targeted_controlled_framing() -> Path:
    data = pd.read_csv(TARGETED / "controlled_framing_backend_comparison.csv")
    data["row_label"] = data.apply(lambda row: _row_label(row["evaluator_backend_label"], _variant_label(row["evaluator_variant"])), axis=1)
    view = data.pivot_table(
        index="row_label",
        columns="framing_comparison",
        values="mean_abs_score_shift",
        aggfunc="mean",
    )
    view = view.rename(columns=FRAMING_LABELS)
    path = TARGETED / "controlled_framing_backend_comparison_heatmap.png"
    save_paper_heatmap(
        view,
        path,
        "Controlled-framing sensitivity by evaluator backend",
        "mean absolute score shift",
        vmin=0,
        vmax=max(0.1, float(view.max().max())),
        fmt=".3f",
        pretty_columns=True,
        pretty_index=False,
        figsize=(8.8, 4.6),
    )
    _copy_to_paper(path, "qwen3_targeted_controlled_framing_backend_comparison_heatmap.png")
    return path


def rebuild_targeted_boundary_hard() -> Path:
    data = pd.read_csv(TARGETED / "boundary_hard_backend_comparison.csv")
    data["row_label"] = data.apply(lambda row: _row_label(row["evaluator_backend_label"], _variant_label(row["evaluator_variant"])), axis=1)
    view = data.pivot_table(
        index="row_label",
        columns="reference_label",
        values="final_score",
        aggfunc="mean",
    )
    view = view[[col for col in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"] if col in view.columns]]
    path = TARGETED / "boundary_hard_final_score_backend_comparison_heatmap.png"
    save_paper_heatmap(
        view,
        path,
        "Boundary-hard final-score profile by evaluator backend",
        "mean final score",
        vmin=0,
        vmax=1,
        fmt=".2f",
        pretty_columns=False,
        pretty_index=False,
        figsize=(8.8, 5.2),
        xrotation=25,
    )
    _copy_to_paper(path, "qwen3_targeted_boundary_hard_final_score_backend_comparison_heatmap.png")
    return path


def main() -> None:
    outputs = [
        rebuild_backend_agreement(),
        rebuild_pc1_loading(),
        rebuild_target_split_sensitivity(),
        rebuild_goal2_compact_profile(),
        rebuild_targeted_controlled_framing(),
        rebuild_targeted_boundary_hard(),
    ]
    print("Rebuilt paper heatmaps:")
    for path in outputs:
        print(f"- {path}")
    print(f"Copied updated paper PNGs to {PAPER_FIGURE_DIR}")


if __name__ == "__main__":
    main()
