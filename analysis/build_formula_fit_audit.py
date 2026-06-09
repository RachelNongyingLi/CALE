#!/usr/bin/env python3
"""Build screening evidence for the CALE evaluator-behavior trace formula.

This script does not fit a new universal latent model. It checks three empirical
implications of the notation used in the thesis:

1. Aggregation: final_score should match the documented aggregation rule
   within numerical tolerance.
2. Facets: evaluator backend and evaluator variant should be treated as
   measurement facets because they explain systematic score/indicator variation.
3. Perturbations: controlled framing and boundary-hard subsets should show
   localized responses to construct-relevant cue manipulations.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_paper_style


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED = ROOT / "figures" / "global_evaluator_audit" / "combined_available_behavior_matrix.csv"
DEFAULT_TARGETED = ROOT / "figures" / "qwen3_14b_targeted_validity_summary"
DEFAULT_OUTDIR = ROOT / "figures" / "formula_fit_audit"
PAPER_FIGURE_DIR = ROOT / "figures" / "paper_appendix"

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

BACKEND_LABELS = {
    "Rule-based scoring backend (non-LLM)": "Instrumentation",
    "Instrumentation backend": "Instrumentation",
    "Rule-based heuristic": "Instrumentation",
    "heuristic_default": "Instrumentation",
    "Qwen2.5-7B local HF judge": "Qwen2.5",
    "Qwen3-14B local HF judge": "Qwen3",
    "Qwen/Qwen3-14B HF judge": "Qwen3",
    "qwen3_14b": "Qwen3",
    "Gemma3-12B local HF judge": "Gemma3",
    "DeepSeek V4-Pro API judge": "DeepSeek",
}

AGGREGATION_COLUMNS = [
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

AGGREGATION_WEIGHTS = {
    "correction_accuracy": 1.5,
    "source_faithfulness": 1.1,
}

VARIANT_LABELS = {
    "baseline_binary": "Binary",
    "baseline_likert": "Likert",
    "direct_trustllm_heuristic": "Direct heuristic",
    "direct_llm_judge": "Direct LLM",
    "generic_cale": "Generic",
    "attack_aware_cale": "Attack-aware",
    "full_attack_aware_cale": "Full CALE",
}


def normalize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "evaluator_variant" not in out.columns and "variant" in out.columns:
        out["evaluator_variant"] = out["variant"]
    if "target_model" not in out.columns and "model_name" in out.columns:
        out["target_model"] = out["model_name"]
    if "evaluator_backend_label" not in out.columns:
        out["evaluator_backend_label"] = out.get("evaluator_backend_model", "unknown_backend")
    out["evaluator_backend_label"] = out["evaluator_backend_label"].replace(
        {"Rule-based scoring backend (non-LLM)": "Instrumentation backend"}
    )
    out["response_key"] = out["target_model"].astype(str) + "|" + out["id"].astype(str)
    for col in ["final_score", *CONSTRUCT_COLUMNS]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def aggregation_fit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (backend, variant), group in data.groupby(["evaluator_backend_label", "evaluator_variant"], dropna=False):
        if variant not in {"generic_cale", "attack_aware_cale", FULL_CALE}:
            continue
        if "final_score" not in group.columns or any(col not in group.columns for col in AGGREGATION_COLUMNS):
            continue
        sub = group[["response_key", "final_score", *AGGREGATION_COLUMNS]].dropna()
        if len(sub) < 100:
            continue
        weight_vec = np.array([AGGREGATION_WEIGHTS.get(col, 1.0) for col in AGGREGATION_COLUMNS], dtype=float)
        weighted = sub[AGGREGATION_COLUMNS].to_numpy(dtype=float) @ weight_vec / weight_vec.sum()
        recomputed = np.clip(0.08 + 0.88 * weighted, 0, 1)
        residual = sub["final_score"].to_numpy(dtype=float) - recomputed
        abs_residual = np.abs(residual)
        rows.append(
            {
                "evaluator_backend_label": backend,
                "evaluator_variant": variant,
                "n_rows": len(sub),
                "n_indicators": len(AGGREGATION_COLUMNS),
                "indicator_columns": ", ".join(AGGREGATION_COLUMNS),
                "recomputed_score_mae": float(abs_residual.mean()),
                "recomputed_score_rmse": float(np.sqrt(np.mean(residual**2))),
                "max_abs_error": float(abs_residual.max()),
                "p95_abs_error": float(np.quantile(abs_residual, 0.95)),
                "rows_above_0_001": int((abs_residual > 0.001).sum()),
                "rows_above_0_005": int((abs_residual > 0.005).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["evaluator_variant", "evaluator_backend_label"])


def two_way_facet_variance(matrix: pd.DataFrame) -> dict[str, float]:
    y = matrix.to_numpy(dtype=float)
    n_items, n_facets = y.shape
    grand = float(np.mean(y))
    item_means = np.mean(y, axis=1)
    facet_means = np.mean(y, axis=0)
    ss_total = float(np.sum((y - grand) ** 2))
    ss_item = float(n_facets * np.sum((item_means - grand) ** 2))
    ss_facet = float(n_items * np.sum((facet_means - grand) ** 2))
    ss_residual = ss_total - ss_item - ss_facet
    if ss_total <= 0:
        return {}
    return {
        "n_items": n_items,
        "n_facets": n_facets,
        "item_eta2": ss_item / ss_total,
        "facet_eta2": ss_facet / ss_total,
        "item_by_facet_residual_eta2": ss_residual / ss_total,
        "facet_mean_range": float(np.max(facet_means) - np.min(facet_means)),
        "mean_pairwise_abs_difference": float(
            np.mean(
                [
                    np.mean(np.abs(y[:, left] - y[:, right]))
                    for left in range(n_facets)
                    for right in range(left + 1, n_facets)
                ]
            )
        )
        if n_facets > 1
        else np.nan,
    }


def facet_variance_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = data[data["evaluator_variant"].eq(FULL_CALE)].copy()
    outcomes = ["final_score", *[col for col in CONSTRUCT_COLUMNS if col in full.columns]]
    backend_rows = []
    for outcome in outcomes:
        pivot = full.pivot_table(index="response_key", columns="evaluator_backend_label", values=outcome, aggfunc="mean")
        pivot = pivot.dropna()
        if pivot.shape[0] < 50 or pivot.shape[1] < 2:
            continue
        row = two_way_facet_variance(pivot)
        row.update({"facet": "evaluator_backend", "outcome": outcome, "facet_levels": " | ".join(map(str, pivot.columns))})
        backend_rows.append(row)

    heuristic = data[data["evaluator_backend_label"].eq("Instrumentation backend")].copy()
    variant_rows = []
    all_variant_pivot = heuristic.pivot_table(index="response_key", columns="evaluator_variant", values="final_score", aggfunc="mean").dropna()
    if all_variant_pivot.shape[0] >= 50 and all_variant_pivot.shape[1] >= 2:
        row = two_way_facet_variance(all_variant_pivot)
        row.update({"facet": "heuristic_evaluator_variant", "outcome": "final_score", "facet_levels": " | ".join(map(str, all_variant_pivot.columns))})
        variant_rows.append(row)
    cale_family = heuristic[heuristic["evaluator_variant"].isin(["generic_cale", "attack_aware_cale", FULL_CALE])]
    for outcome in [col for col in CONSTRUCT_COLUMNS if col in cale_family.columns]:
        pivot = cale_family.pivot_table(index="response_key", columns="evaluator_variant", values=outcome, aggfunc="mean").dropna()
        if pivot.shape[0] < 50 or pivot.shape[1] < 2:
            continue
        row = two_way_facet_variance(pivot)
        row.update({"facet": "heuristic_cale_variant", "outcome": outcome, "facet_levels": " | ".join(map(str, pivot.columns))})
        variant_rows.append(row)

    return (
        pd.DataFrame(backend_rows).sort_values("facet_eta2", ascending=False),
        pd.DataFrame(variant_rows).sort_values("facet_eta2", ascending=False),
    )


def perturbation_summary(targeted_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    framing_path = targeted_dir / "controlled_framing_backend_comparison.csv"
    boundary_path = targeted_dir / "boundary_hard_metric_backend_comparison_compact.csv"
    framing = pd.read_csv(framing_path) if framing_path.exists() else pd.DataFrame()
    boundary = pd.read_csv(boundary_path) if boundary_path.exists() else pd.DataFrame()
    if not framing.empty:
        framing = framing.copy()
        if "evaluator_backend_label" in framing.columns:
            framing["evaluator_backend_label"] = framing["evaluator_backend_label"].replace(
                {"Rule-based heuristic": "Instrumentation backend"}
            )
        framing["evidence_role"] = "fixed_response_framing_cue_screen"
    if not boundary.empty:
        boundary = boundary.copy()
        for col in ["evaluator_backend_label", "evaluator_backend_key"]:
            if col in boundary.columns:
                boundary[col] = boundary[col].replace(
                    {"Rule-based heuristic": "Instrumentation backend", "heuristic_default": "Instrumentation backend"}
                )
        boundary["evidence_role"] = "boundary_nei_uncertainty_cue_screen"
    return framing, boundary


def label_backend(value: object) -> str:
    text = str(value)
    return BACKEND_LABELS.get(text, text.replace(" local HF judge", "").replace(" API judge", ""))


def label_variant(value: object) -> str:
    text = str(value)
    return VARIANT_LABELS.get(text, text.replace("_", " ").title())


def label_outcome(value: object) -> str:
    return str(value).replace("_", " ").replace("nei", "NEI").title()


def save_formula_dashboard(
    aggregation: pd.DataFrame,
    backend: pd.DataFrame,
    framing: pd.DataFrame,
    boundary: pd.DataFrame,
    outdir: Path,
) -> Path:
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    if not aggregation.empty:
        agg = aggregation.copy()
        total_rows = int(agg["n_rows"].sum())
        max_error = float(agg["max_abs_error"].max())
        mae_min = float(agg["recomputed_score_mae"].min())
        mae_max = float(agg["recomputed_score_mae"].max())
        failures = int(agg["rows_above_0_001"].sum())
        ax_a.axis("off")
        ax_a.set_title("A. Aggregation checksum", loc="left")
        lines = [
            f"Rows checked: {total_rows:,}",
            f"Max absolute residual: {max_error:.4f}",
            f"MAE range by backend/variant: {mae_min:.5f}-{mae_max:.5f}",
            f"Rows > 0.001 tolerance: {failures}",
            "",
            "Interpretation: score construction and row alignment check;",
            "not evaluator validity or response-level correctness.",
        ]
        ax_a.text(
            0.02,
            0.82,
            "\n".join(lines),
            transform=ax_a.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
        )
    else:
        ax_a.axis("off")

    if not backend.empty:
        order = [
            "final_score",
            "hallucination_control",
            "uncertainty_handling",
            "evidence_grounding",
            "source_faithfulness",
            "claim_status_recognition",
        ]
        fac = backend[backend["outcome"].isin(order)].copy()
        fac["outcome_order"] = fac["outcome"].map({name: idx for idx, name in enumerate(order)})
        fac = fac.sort_values("outcome_order")
        y = np.arange(len(fac))
        left = np.zeros(len(fac))
        parts = [
            ("item_eta2", "#D9E8F5", "Response item"),
            ("facet_eta2", "#2E7D6E", "Backend"),
            ("item_by_facet_residual_eta2", "#F3C6B8", "Item x backend"),
        ]
        for col, color, label in parts:
            values = fac[col].to_numpy(dtype=float)
            ax_b.barh(y, values, left=left, color=color, edgecolor="white", label=label)
            left += values
        ax_b.set_yticks(y)
        ax_b.set_yticklabels([label_outcome(v) for v in fac["outcome"]])
        ax_b.invert_yaxis()
        ax_b.set_xlim(0, 1)
        ax_b.set_xlabel("Variance share on matched Full CALE rows")
        ax_b.set_title("B. Backend is a measurement facet")
        ax_b.legend(loc="lower right", fontsize=7.2, frameon=False)
    else:
        ax_b.axis("off")

    if not framing.empty:
        frame = framing[framing["evaluator_variant"].astype(str).str.contains("full_attack_aware_cale", na=False)].copy()
        frame["label"] = frame["evaluator_backend_label"].map(label_backend) + "\n" + frame["framing_comparison"].str.replace("_minus_neutral", "", regex=False).str.title()
        frame = frame.sort_values(["evaluator_backend_label", "framing_comparison"])
        ax_c.bar(frame["label"], frame["mean_abs_score_shift"], color="#1F5F8B", edgecolor="white")
        ax_c.set_ylabel("Mean absolute score shift")
        ax_c.set_title("C. Fixed-response framing cue response")
        ax_c.tick_params(axis="x", labelrotation=25)
        for x, value in enumerate(frame["neutral_vs_framed_spearman"]):
            ax_c.text(x, frame["mean_abs_score_shift"].iloc[x] + 0.002, f"ρ={value:.2f}", ha="center", va="bottom", fontsize=7.1)
    else:
        ax_c.axis("off")

    if not boundary.empty:
        nei = boundary[boundary["reference_label"].eq("NOT ENOUGH INFO")].copy()
        nei = nei[nei["evaluator_variant"].astype(str).str.contains("full_attack_aware_cale", na=False)]
        nei["label"] = nei["evaluator_backend_key"].map(label_backend)
        metrics = ["uncertainty_handling", "nei_uncertainty_failure_proxy"]
        x = np.arange(len(nei))
        width = 0.36
        for offset, metric, color in [(-width / 2, metrics[0], "#4D93B7"), (width / 2, metrics[1], "#B23A2E")]:
            if metric in nei.columns:
                ax_d.bar(x + offset, nei[metric], width=width, color=color, edgecolor="white", label=label_outcome(metric))
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(nei["label"], rotation=25, ha="right")
        ax_d.set_ylim(0, 1.05)
        ax_d.set_ylabel("Mean on NEI boundary-hard rows")
        ax_d.set_title("D. Boundary/uncertainty cue diagnostic")
        ax_d.legend(loc="upper right", fontsize=7.2, frameon=False)
    else:
        ax_d.axis("off")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Formula-Level Measurement Checks for the CALE Evaluator-Behavior Trace", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = outdir / "formula_fit_audit_dashboard.png"
    fig.savefig(out_path, dpi=320)
    plt.close(fig)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, PAPER_FIGURE_DIR / out_path.name)
    return out_path


def claim_evidence_summary(aggregation: pd.DataFrame, backend: pd.DataFrame, variant: pd.DataFrame, framing: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not aggregation.empty:
        total_rows = int(aggregation["n_rows"].sum())
        max_error = float(aggregation["max_abs_error"].max())
        mae_min = float(aggregation["recomputed_score_mae"].min())
        mae_max = float(aggregation["recomputed_score_mae"].max())
        failures = int(aggregation["rows_above_0_001"].sum())
        rows.append(
            {
                "formula_component": "s = A_{b,v}(y)",
                "empirical_check": "recompute stored final_score from construct indicators and documented aggregation rule",
                "primary_result": f"{total_rows:,} CALE-family rows checked; max absolute residual {max_error:.4f}; MAE range {mae_min:.5f}-{mae_max:.5f}; rows above 0.001 tolerance {failures}",
                "interpretation": "aggregation checksum for score construction and row alignment, not independent validity evidence",
            }
        )
    if not backend.empty:
        final_row = backend[backend["outcome"].eq("final_score")]
        if not final_row.empty:
            row = final_row.iloc[0]
            rows.append(
                {
                    "formula_component": "E_{b,v}",
                    "empirical_check": "matched Full CALE backend facet variance",
                    "primary_result": f"final_score backend eta2 {row['facet_eta2']:.3f}; item-by-backend residual eta2 {row['item_by_facet_residual_eta2']:.3f}; mean pairwise abs diff {row['mean_pairwise_abs_difference']:.3f}",
                    "interpretation": "supports evaluator backend as a measurement facet on shared response rows",
                }
            )
    if not variant.empty:
        final_row = variant[variant["outcome"].eq("final_score")]
        if not final_row.empty:
            row = final_row.iloc[0]
            rows.append(
                {
                    "formula_component": "E_{b,v} / A_{b,v}",
                    "empirical_check": "instrumentation-backend evaluator-variant facet variance",
                    "primary_result": f"final_score variant eta2 {row['facet_eta2']:.3f}; mean pairwise abs diff {row['mean_pairwise_abs_difference']:.3f}",
                    "interpretation": "supports protocol variant as a measurement facet, especially for scalar score semantics",
                }
            )
    if not framing.empty:
        full = framing[framing["evaluator_variant"].astype(str).str.contains("full_attack_aware_cale", na=False)]
        if not full.empty:
            rows.append(
                {
                    "formula_component": "linguistic cue vector l",
                    "empirical_check": "controlled framing fixed-response perturbation",
                    "primary_result": f"Full CALE mean absolute score shift range {full['mean_abs_score_shift'].min():.3f}-{full['mean_abs_score_shift'].max():.3f}; rank stability range {full['neutral_vs_framed_spearman'].min():.3f}-{full['neutral_vs_framed_spearman'].max():.3f}",
                    "interpretation": "supports localized cue response under fixed target responses; screening-level evidence",
                }
            )
    if not boundary.empty:
        nei = boundary[boundary["reference_label"].eq("NOT ENOUGH INFO")]
        if not nei.empty and "nei_uncertainty_failure_proxy" in nei.columns:
            rows.append(
                {
                    "formula_component": "linguistic/boundary cue vector l",
                    "empirical_check": "boundary-hard NEI uncertainty stress screen",
                    "primary_result": f"NEI uncertainty-failure proxy range {nei['nei_uncertainty_failure_proxy'].dropna().min():.3f}-{nei['nei_uncertainty_failure_proxy'].dropna().max():.3f}",
                    "interpretation": "exposes boundary-control diagnostics; not standalone latent-factor validation",
                }
            )
    return pd.DataFrame(rows)


def write_markdown(outdir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Formula Fit Audit",
        "",
        "This audit supports the evaluator-behavior trace notation with screening evidence. It does not claim a single universal latent model fit.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"## {row['formula_component']}",
                "",
                f"- Check: {row['empirical_check']}",
                f"- Result: {row['primary_result']}",
                f"- Interpretation: {row['interpretation']}",
                "",
            ]
        )
    (outdir / "formula_fit_audit_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-matrix", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--targeted-dir", type=Path, default=DEFAULT_TARGETED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = normalize_behavior(pd.read_csv(args.combined_matrix))

    aggregation = aggregation_fit(data)
    backend, variant = facet_variance_tables(data)
    framing, boundary = perturbation_summary(args.targeted_dir)
    summary = claim_evidence_summary(aggregation, backend, variant, framing, boundary)

    aggregation.to_csv(args.output_dir / "table_A_aggregation_fit.csv", index=False)
    backend.to_csv(args.output_dir / "table_B1_backend_facet_variance.csv", index=False)
    variant.to_csv(args.output_dir / "table_B2_variant_facet_variance.csv", index=False)
    framing.to_csv(args.output_dir / "table_C1_controlled_framing_response.csv", index=False)
    boundary.to_csv(args.output_dir / "table_C2_boundary_hard_response.csv", index=False)
    summary.to_csv(args.output_dir / "formula_claim_evidence_summary.csv", index=False)
    dashboard_path = save_formula_dashboard(aggregation, backend, framing, boundary, args.output_dir)
    write_markdown(args.output_dir, summary)

    print(f"Wrote formula fit audit to {args.output_dir}")
    print(f"Wrote dashboard to {dashboard_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
