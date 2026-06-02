#!/usr/bin/env python3
"""Select real FEVER cases from fixed target responses and CALE behavior matrices.

This script is for qualitative/diagnostic case selection, not for creating a
new leaderboard. It joins the fixed response JSONL to the global behavior
matrix using (target_model, id), then exports readable cases and paper-facing
visual checks for:

- NEI boundary-control failures.
- Boundary-indicator mismatches.
- Backend disagreement on the same fixed target response.
- Direct-vs-Full CALE score shifts.
- Representative factual-handling cases.

The key safety rule is that `id` alone is not unique: every FEVER id appears
for both target models. Always merge on `(target_model, id)`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FULL_CALE = "full_attack_aware_cale"
DIRECT_LLM = "direct_llm_judge"
DIRECT_HEURISTIC = "direct_trustllm_heuristic"

BEHAVIOR_COLUMNS = [
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
    "nei_uncertainty_failure_proxy",
    "refutes_correction_credit_proxy",
    "supports_status_failure_proxy",
]

RESPONSE_COLUMNS = [
    "id",
    "target_model",
    "model_name",
    "base_claim",
    "user_prompt",
    "candidate_response",
    "reference_label",
    "reference_fact",
    "reference_evidence",
    "fever_evidence",
    "supporting_evidence",
    "dataset",
    "domain",
    "risk_level",
]

OUTPUT_CASE_COLUMNS = [
    "case_id",
    "case_category",
    "selection_reason",
    "id",
    "target_model",
    "target_short",
    "target_split",
    "run_id",
    "evaluator_backend_model",
    "evaluator_backend_label",
    "backend_short",
    "evaluator_variant",
    "reference_label",
    "base_claim",
    "candidate_response_excerpt",
    "candidate_response",
    "reference_evidence_short",
    "final_score",
    "quality_label",
    "direct_score",
    "full_score",
    "full_minus_direct_score",
    "score_heuristic",
    "score_qwen7b",
    "score_deepseek",
    "backend_score_range",
] + BEHAVIOR_COLUMNS

LEGACY_BACKEND_SCORE_COLUMNS = {
    "heuristic": "score_heuristic",
    "qwen25_7b": "score_qwen7b",
    "deepseek_v4_pro": "score_deepseek",
}
NON_BACKEND_SCORE_COLUMNS = {"score_variant"}


def short_target(model: str) -> str:
    text = str(model)
    if "Qwen2.5-1.5B" in text:
        return "Qwen2.5-1.5B target"
    if "Llama-3.2-1B" in text:
        return "Llama-3.2-1B target"
    return text


def short_backend(model: str) -> str:
    text = str(model)
    lower = text.lower()
    if text == "heuristic/default":
        return "heuristic"
    if "Qwen2.5-7B" in text:
        return "qwen25_7b"
    if "qwen3-14b" in lower or "qwen/qwen3-14b" in lower:
        return "qwen3_14b"
    if "gemma-3-12b" in lower or "gemma3-12b" in lower or "gemma_3_12b" in lower:
        return "gemma3_12b"
    if "deepseek" in lower:
        return "deepseek_v4_pro"
    slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    return slug or "unknown_backend"


def score_column_for_backend(backend_short: str) -> str:
    backend = str(backend_short)
    return LEGACY_BACKEND_SCORE_COLUMNS.get(backend, f"score_{backend}")


def score_label_for_backend_score_column(column: str) -> str:
    reverse_legacy = {value: key for key, value in LEGACY_BACKEND_SCORE_COLUMNS.items()}
    if column in reverse_legacy:
        return reverse_legacy[column]
    if column.startswith("score_"):
        return column.removeprefix("score_")
    return column


def dynamic_output_columns(df: pd.DataFrame) -> list[str]:
    score_cols = sorted(
        c
        for c in df.columns
        if c.startswith("score_") and c not in OUTPUT_CASE_COLUMNS and c not in NON_BACKEND_SCORE_COLUMNS
    )
    if not score_cols:
        return OUTPUT_CASE_COLUMNS
    insert_at = OUTPUT_CASE_COLUMNS.index("backend_score_range")
    return OUTPUT_CASE_COLUMNS[:insert_at] + score_cols + OUTPUT_CASE_COLUMNS[insert_at:]


def excerpt(value: object, limit: int = 320) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_responses(path: Path) -> pd.DataFrame:
    responses = read_jsonl(path)
    if "target_model" not in responses.columns:
        responses["target_model"] = responses["model_name"]
    responses["target_short"] = responses["target_model"].map(short_target)
    responses["candidate_response_excerpt"] = responses["candidate_response"].map(excerpt)
    evidence_source = responses.get("reference_evidence", responses.get("fever_evidence", ""))
    responses["reference_evidence_short"] = evidence_source.map(excerpt)
    keep = [c for c in RESPONSE_COLUMNS + ["target_short", "candidate_response_excerpt", "reference_evidence_short"] if c in responses.columns]
    responses = responses[keep].copy()
    duplicate_keys = responses.duplicated(["target_model", "id"]).sum()
    if duplicate_keys:
        raise ValueError(f"Response file has {duplicate_keys} duplicate (target_model, id) keys.")
    return responses


def load_behavior(path: Path) -> pd.DataFrame:
    behavior = pd.read_csv(path)
    if "target_model" not in behavior.columns:
        behavior["target_model"] = behavior.get("model_name")
    if "evaluator_variant" not in behavior.columns:
        behavior["evaluator_variant"] = behavior.get("variant")
    if "run_id" not in behavior.columns:
        behavior["run_id"] = "unknown_run"
    if "evaluator_backend_model" not in behavior.columns:
        behavior["evaluator_backend_model"] = "unknown_backend"
    if "evaluator_backend_label" not in behavior.columns:
        behavior["evaluator_backend_label"] = behavior["evaluator_backend_model"]
    behavior["backend_short"] = behavior["evaluator_backend_model"].map(short_backend)
    behavior["target_short"] = behavior["target_model"].map(short_target)
    return behavior


def merge_behavior_responses(behavior: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    merged = behavior.merge(
        responses,
        on=["target_model", "id"],
        how="left",
        suffixes=("", "_response"),
        validate="many_to_one",
    )
    missing = merged["candidate_response"].isna().sum()
    if missing:
        raise ValueError(f"{missing} behavior rows did not match a fixed response on (target_model, id).")
    if "reference_label_response" in merged.columns:
        mismatch = (merged["reference_label"] != merged["reference_label_response"]).sum()
        if mismatch:
            raise ValueError(f"{mismatch} merged rows have conflicting reference_label values.")
    if "target_short_response" in merged.columns and "target_short" in merged.columns:
        merged["target_short"] = merged["target_short"].fillna(merged["target_short_response"])
    return merged


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def top_per_group(df: pd.DataFrame, group_cols: list[str], sort_col: str, n: int, ascending: bool = False) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.sort_values(sort_col, ascending=ascending)
        .groupby(group_cols, dropna=False, group_keys=False)
        .head(n)
        .copy()
    )


def add_case_id(df: pd.DataFrame, category: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "case_id", [f"{category}_{idx + 1:03d}" for idx in range(len(out))])
    return out


def tidy_case_rows(df: pd.DataFrame, category: str, reason: str) -> pd.DataFrame:
    output_columns = dynamic_output_columns(df)
    if df.empty:
        return pd.DataFrame(columns=output_columns)
    out = df.copy()
    out["case_category"] = category
    out["selection_reason"] = reason
    out = add_case_id(out, category)
    output_columns = dynamic_output_columns(out)
    for col in output_columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[output_columns].copy()


def select_nei_boundary_failures(full: pd.DataFrame, per_backend_target: int) -> pd.DataFrame:
    candidates = full[full["reference_label"].eq("NOT ENOUGH INFO")].copy()
    candidates = numeric(candidates, ["hallucination_control", "uncertainty_handling", "nei_uncertainty_failure_proxy", "final_score"])
    candidates["boundary_failure_severity"] = (
        (1 - candidates["hallucination_control"].fillna(1))
        + (1 - candidates["uncertainty_handling"].fillna(1))
        + candidates["nei_uncertainty_failure_proxy"].fillna(0)
        + (1 - candidates["final_score"].fillna(1))
    )
    candidates = candidates[
        (candidates["hallucination_control"].eq(0))
        | (candidates["uncertainty_handling"].eq(0))
        | (candidates["nei_uncertainty_failure_proxy"].eq(1))
    ]
    selected = top_per_group(
        candidates,
        ["backend_short", "target_short"],
        "boundary_failure_severity",
        per_backend_target,
        ascending=False,
    )
    return tidy_case_rows(
        selected,
        "nei_boundary_failure",
        "NEI case where Full CALE boundary-control indicators flag overclaim or uncertainty failure.",
    )


def select_boundary_mismatches(full: pd.DataFrame, per_backend_target: int) -> pd.DataFrame:
    candidates = full[full["reference_label"].eq("NOT ENOUGH INFO")].copy()
    candidates = numeric(candidates, ["hallucination_control", "uncertainty_handling", "final_score"])
    candidates = candidates[candidates["hallucination_control"].notna() & candidates["uncertainty_handling"].notna()]
    candidates = candidates[candidates["hallucination_control"] != candidates["uncertainty_handling"]].copy()
    candidates["mismatch_score"] = (0.5 - candidates["final_score"].fillna(0.5)).abs()
    selected = top_per_group(candidates, ["backend_short", "target_short"], "mismatch_score", per_backend_target)
    return tidy_case_rows(
        selected,
        "boundary_indicator_mismatch",
        "Real NEI case where hallucination_control and uncertainty_handling separate.",
    )


def select_backend_disagreements(merged: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidates = merged[merged["evaluator_variant"].isin([FULL_CALE, DIRECT_LLM, DIRECT_HEURISTIC])].copy()
    candidates = numeric(candidates, ["final_score"])

    canonical_variant = candidates["evaluator_variant"].replace({DIRECT_HEURISTIC: DIRECT_LLM})
    candidates["comparison_variant"] = np.where(
        candidates["evaluator_variant"].eq(FULL_CALE),
        FULL_CALE,
        canonical_variant,
    )
    key_cols = ["target_model", "target_short", "target_split", "id", "comparison_variant", "reference_label"]
    pivot = candidates.pivot_table(
        index=key_cols,
        columns="backend_short",
        values="final_score",
        aggfunc="mean",
    ).reset_index()
    backend_cols = [c for c in pivot.columns if c not in key_cols]
    if not backend_cols:
        return tidy_case_rows(pd.DataFrame(), "backend_disagreement", "")
    pivot = pivot[pivot[backend_cols].notna().sum(axis=1) >= 2].copy()
    if pivot.empty:
        return tidy_case_rows(pd.DataFrame(), "backend_disagreement", "")
    pivot["backend_score_range"] = pivot[backend_cols].max(axis=1) - pivot[backend_cols].min(axis=1)
    pivot = pivot[pivot["backend_score_range"] >= 0.35].sort_values("backend_score_range", ascending=False).head(top_n)

    rename_scores = {backend: score_column_for_backend(backend) for backend in backend_cols}
    rename_scores["comparison_variant"] = "evaluator_variant"
    pivot = pivot.rename(columns=rename_scores)
    response_cols = [
        "target_model",
        "id",
        "base_claim",
        "candidate_response",
        "candidate_response_excerpt",
        "reference_evidence_short",
        "domain",
        "risk_level",
    ]
    response_lookup = merged[response_cols].drop_duplicates(["target_model", "id"])
    out = pivot.merge(response_lookup, on=["target_model", "id"], how="left", validate="many_to_one")
    out["evaluator_backend_model"] = "multiple"
    out["evaluator_backend_label"] = "Backend comparison"
    out["backend_short"] = "multiple"
    return tidy_case_rows(
        out,
        "backend_disagreement",
        "Same fixed target response receives substantially different scores across evaluator backends.",
    )


def select_direct_vs_full_shifts(merged: pd.DataFrame, per_backend_target_direction: int) -> pd.DataFrame:
    candidates = merged[merged["evaluator_variant"].isin([FULL_CALE, DIRECT_LLM, DIRECT_HEURISTIC])].copy()
    candidates = numeric(candidates, ["final_score"])
    candidates["protocol_pair"] = np.where(candidates["evaluator_variant"].eq(FULL_CALE), "full", "direct")
    index_cols = [
        "run_id",
        "evaluator_backend_model",
        "evaluator_backend_label",
        "backend_short",
        "target_model",
        "target_short",
        "target_split",
        "id",
        "reference_label",
    ]
    wide = candidates.pivot_table(index=index_cols, columns="protocol_pair", values="final_score", aggfunc="mean").reset_index()
    if "direct" not in wide.columns or "full" not in wide.columns:
        return tidy_case_rows(pd.DataFrame(), "direct_vs_full_shift", "")
    wide = wide.dropna(subset=["direct", "full"]).copy()
    wide["direct_score"] = wide["direct"]
    wide["full_score"] = wide["full"]
    wide["full_minus_direct_score"] = wide["full"] - wide["direct"]
    wide["abs_shift"] = wide["full_minus_direct_score"].abs()
    wide = wide[wide["abs_shift"] >= 0.25].copy()
    if wide.empty:
        return tidy_case_rows(pd.DataFrame(), "direct_vs_full_shift", "")

    positive = top_per_group(
        wide[wide["full_minus_direct_score"] > 0],
        ["backend_short", "target_short"],
        "full_minus_direct_score",
        per_backend_target_direction,
        ascending=False,
    )
    negative = top_per_group(
        wide[wide["full_minus_direct_score"] < 0],
        ["backend_short", "target_short"],
        "full_minus_direct_score",
        per_backend_target_direction,
        ascending=True,
    )
    selected = pd.concat([positive, negative], ignore_index=True, sort=False)

    response_cols = [
        "target_model",
        "id",
        "base_claim",
        "candidate_response",
        "candidate_response_excerpt",
        "reference_evidence_short",
        "domain",
        "risk_level",
    ]
    response_lookup = merged[response_cols].drop_duplicates(["target_model", "id"])
    selected = selected.merge(response_lookup, on=["target_model", "id"], how="left", validate="many_to_one")
    selected["evaluator_variant"] = "direct_vs_full_cale"
    return tidy_case_rows(
        selected,
        "direct_vs_full_shift",
        "Full CALE changes the score substantially relative to the backend-specific direct baseline.",
    )


def select_representative_factual_handling(full: pd.DataFrame, per_label_target: int) -> pd.DataFrame:
    candidates = full[full["backend_short"].eq("heuristic")].copy()
    factual_cols = [
        "source_faithfulness",
        "evidence_grounding",
        "claim_status_recognition",
        "correction_accuracy",
        "misinformation_detection",
    ]
    candidates = numeric(candidates, factual_cols + ["final_score"])
    candidates["factual_handling_mean"] = candidates[factual_cols].mean(axis=1, skipna=True)
    candidates = candidates[candidates["factual_handling_mean"].notna()].copy()
    selected_parts = []
    for _, group in candidates.groupby(["reference_label", "target_short"], dropna=False):
        group = group.sort_values("factual_handling_mean")
        if group.empty:
            continue
        idxs = sorted(set([0, len(group) // 2, len(group) - 1]))
        selected_parts.append(group.iloc[idxs].head(per_label_target))
    selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()
    return tidy_case_rows(
        selected,
        "representative_factual_handling",
        "Representative Full CALE heuristic examples spanning low, middle, and high factual-handling signals.",
    )


def save_barh(series: pd.Series, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(series))))
    series.sort_values().plot(kind="barh", ax=ax, color="#4c78a8")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, path: Path, title: str, cbar_label: str, fmt: str = ".2f") -> None:
    if df.empty:
        return
    fig_width = max(8, 1.2 * len(df.columns) + 3)
    fig_height = max(4, 0.5 * len(df.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    values = df.astype(float).to_numpy()
    im = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "NA" if np.isnan(value) else format(value, fmt)
            ax.text(j, i, label, ha="center", va="center", color="white" if value > np.nanmax(values) * 0.55 else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_backend_coverage_outputs(merged: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    coverage = merged.copy()
    coverage["response_key"] = coverage["target_model"].astype(str) + "||" + coverage["id"].astype(str)
    group_cols = [
        "evaluator_backend_model",
        "evaluator_backend_label",
        "backend_short",
        "evaluator_variant",
        "target_short",
    ]
    coverage = (
        coverage.groupby(group_cols, dropna=False)
        .agg(
            rows=("id", "size"),
            unique_response_keys=("response_key", "nunique"),
            unique_fever_ids=("id", "nunique"),
        )
        .reset_index()
        .sort_values(["backend_short", "evaluator_variant", "target_short"])
    )
    coverage.to_csv(outdir / "real_case_selection_backend_coverage.csv", index=False)

    if not coverage.empty:
        view = coverage.assign(backend_variant=coverage["backend_short"] + " / " + coverage["evaluator_variant"])
        heatmap = view.pivot_table(
            index="backend_variant",
            columns="target_short",
            values="unique_response_keys",
            aggfunc="sum",
            fill_value=0,
        )
        save_heatmap(
            heatmap,
            outdir / "backend_variant_target_coverage.png",
            "Backend/Variant Coverage by Target",
            "unique target responses",
            fmt=".0f",
        )
    return coverage


def make_visualizations(merged: pd.DataFrame, selected: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if not selected.empty:
        counts = selected.groupby(["case_category", "target_short"], dropna=False).size().rename("selected_cases")
        save_barh(
            counts,
            outdir / "real_case_selection_counts.png",
            "Selected Real Cases by Category and Target Model",
            "selected cases",
        )

        coverage = selected.pivot_table(
            index="case_category",
            columns="reference_label",
            values="id",
            aggfunc="count",
            fill_value=0,
        )
        save_heatmap(
            coverage,
            outdir / "selected_cases_reference_label_coverage.png",
            "Selected Case Coverage by Reference Label",
            "case count",
            fmt=".0f",
        )

    full = merged[merged["evaluator_variant"].eq(FULL_CALE)].copy()
    full = numeric(full, ["hallucination_control", "uncertainty_handling", "nei_uncertainty_failure_proxy", "final_score"])
    nei = full[full["reference_label"].eq("NOT ENOUGH INFO")].copy()
    if not nei.empty:
        nei["boundary_failure_rate"] = (
            nei["hallucination_control"].eq(0)
            | nei["uncertainty_handling"].eq(0)
            | nei["nei_uncertainty_failure_proxy"].eq(1)
        ).astype(float)
        rate = nei.pivot_table(
            index="evaluator_backend_label",
            columns="target_short",
            values="boundary_failure_rate",
            aggfunc="mean",
        )
        save_heatmap(
            rate,
            outdir / "nei_boundary_failure_rate_by_backend_target.png",
            "NEI Boundary-Failure Rate by Evaluator Backend and Target",
            "failure rate",
        )

        combo = nei.assign(
            combo=(
                "HC="
                + nei["hallucination_control"].fillna(-1).astype(int).astype(str)
                + ", UH="
                + nei["uncertainty_handling"].fillna(-1).astype(int).astype(str)
            )
        )
        combo_counts = combo.pivot_table(
            index="evaluator_backend_label",
            columns="combo",
            values="id",
            aggfunc="count",
            fill_value=0,
        )
        save_heatmap(
            combo_counts,
            outdir / "boundary_indicator_combo_counts_by_backend.png",
            "Boundary Indicator Combinations on NEI Cases",
            "case count",
            fmt=".0f",
        )

    shifts = selected[selected["case_category"].eq("direct_vs_full_shift")].copy()
    if not shifts.empty:
        shift_pivot = shifts.pivot_table(
            index="evaluator_backend_label",
            columns="target_short",
            values="full_minus_direct_score",
            aggfunc=lambda s: float(np.mean(np.abs(s))),
        )
        save_heatmap(
            shift_pivot,
            outdir / "selected_direct_vs_full_abs_shift_by_backend_target.png",
            "Selected Direct-vs-Full Absolute Score Shifts",
            "mean abs shift",
        )

    disagreements = selected[selected["case_category"].eq("backend_disagreement")].copy()
    if not disagreements.empty:
        score_range = disagreements.set_index("case_id")["backend_score_range"].sort_values(ascending=False).head(25)
        save_barh(
            score_range,
            outdir / "top_backend_disagreement_score_ranges.png",
            "Top Backend Disagreement Cases",
            "max-min final score",
        )
        score_cols = sorted(c for c in disagreements.columns if c.startswith("score_") and c not in NON_BACKEND_SCORE_COLUMNS)
        score_view = disagreements.sort_values("backend_score_range", ascending=False).head(20).set_index("case_id")
        score_view = score_view[[c for c in score_cols if c in score_view.columns]]
        score_view = score_view.rename(columns={c: score_label_for_backend_score_column(c) for c in score_view.columns})
        save_heatmap(
            score_view,
            outdir / "top_backend_disagreement_score_heatmap.png",
            "Top Backend Disagreement Cases: Scores by Backend",
            "final score",
        )


def write_readable_markdown(selected: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Real CALE Case Selection",
        "",
        "These cases are selected from fixed target-model responses and behavior matrices.",
        "They are diagnostic examples for interpretation, not a replacement for aggregate evidence.",
        "",
        "Merge key: `(target_model, id)`.",
        "",
    ]
    if selected.empty:
        lines.append("No cases selected.")
    for category, group in selected.groupby("case_category", dropna=False):
        lines.extend([f"## {category}", ""])
        for _, row in group.head(8).iterrows():
            lines.extend(
                [
                    f"### {row.get('case_id')} | {row.get('target_short')} | {row.get('reference_label')}",
                    "",
                    f"- Backend/protocol: {row.get('evaluator_backend_label')} / {row.get('evaluator_variant')}",
                    f"- Selection reason: {row.get('selection_reason')}",
                    f"- Score: {row.get('final_score')}",
                    f"- Boundary: HC={row.get('hallucination_control')}, UH={row.get('uncertainty_handling')}, NEI proxy={row.get('nei_uncertainty_failure_proxy')}",
                    "",
                    "**Claim**",
                    "",
                    textwrap.fill(str(row.get("base_claim", "")), width=100),
                    "",
                    "**Candidate response**",
                    "",
                    textwrap.fill(str(row.get("candidate_response_excerpt", "")), width=100),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select real CALE diagnostic cases from fixed responses.")
    parser.add_argument(
        "--responses",
        default="outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl",
        help="Fixed response JSONL with candidate_response.",
    )
    parser.add_argument(
        "--behavior",
        default="figures/global_evaluator_audit/combined_available_behavior_matrix.csv",
        help="Combined behavior matrix from the global evaluator audit notebook.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/real_case_selection",
        help="Output directory for selected cases and plots.",
    )
    parser.add_argument("--per-backend-target", type=int, default=5)
    parser.add_argument("--backend-disagreement-top-n", type=int, default=30)
    args = parser.parse_args()

    responses_path = Path(args.responses)
    behavior_path = Path(args.behavior)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    responses = load_responses(responses_path)
    behavior = load_behavior(behavior_path)
    merged = merge_behavior_responses(behavior, responses)
    merged = numeric(merged, BEHAVIOR_COLUMNS + ["final_score"])
    backend_coverage = save_backend_coverage_outputs(merged, outdir)

    full = merged[merged["evaluator_variant"].eq(FULL_CALE)].copy()
    selected_parts = [
        select_nei_boundary_failures(full, args.per_backend_target),
        select_boundary_mismatches(full, args.per_backend_target),
        select_backend_disagreements(merged, args.backend_disagreement_top_n),
        select_direct_vs_full_shifts(merged, args.per_backend_target),
        select_representative_factual_handling(full, 3),
    ]
    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    selected = selected.reindex(columns=dynamic_output_columns(selected))

    summary = pd.DataFrame(
        [
            {"metric": "response_rows", "value": len(responses)},
            {"metric": "behavior_rows", "value": len(behavior)},
            {"metric": "merged_rows", "value": len(merged)},
            {"metric": "selected_case_rows", "value": len(selected)},
            {"metric": "unique_selected_target_ids", "value": selected[["target_model", "id"]].drop_duplicates().shape[0] if not selected.empty else 0},
        ]
    )
    coverage = (
        selected.groupby(["case_category", "target_short"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        if not selected.empty
        else pd.DataFrame(columns=["case_category", "target_short", "rows"])
    )

    merged.head(0).to_csv(outdir / "real_case_selection_long_schema.csv", index=False)
    selected.to_csv(outdir / "real_case_selection_selected_cases.csv", index=False)
    summary.to_csv(outdir / "real_case_selection_summary.csv", index=False)
    coverage.to_csv(outdir / "real_case_selection_coverage.csv", index=False)

    for category, group in selected.groupby("case_category", dropna=False):
        safe_category = str(category).replace("/", "_")
        group.to_csv(outdir / f"selected_{safe_category}.csv", index=False)

    make_visualizations(merged, selected, outdir)
    write_readable_markdown(selected, outdir / "selected_cases_readable.md")

    print(f"Loaded responses: {len(responses)}")
    print(f"Loaded behavior rows: {len(behavior)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Backend coverage rows: {len(backend_coverage)}")
    print(f"Selected case rows: {len(selected)}")
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
