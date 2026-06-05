#!/usr/bin/env python3
"""Build fixed-order construct-family tables for CALE paper figures.

The global audit originally reported "top PC1 variables" as variable-name
strings. That is useful for exploration, but awkward in the thesis because the
same variables appear in different orders across evaluator backends. This
script converts the existing paper-ready audit outputs into fixed construct
families so readers can compare the same structure across backends and target
splits.

Important interpretation guardrail: PCA loading magnitudes are
structure/concentration evidence, not evaluator-quality scores or a model
leaderboard. The construct families below are exploratory summaries of CALE
signals and should be used as screening evidence unless a later CFA/invariance
study validates them more strongly.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from plot_style import save_paper_heatmap


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "figures" / "global_evaluator_audit"
PAPER_TABLE_DIR = ROOT / "figures" / "paper_tables"
PAPER_FIGURE_DIR = ROOT / "figures" / "paper_appendix"

FULL_CALE = "full_attack_aware_cale"

FAMILY_DEFINITIONS: dict[str, list[str]] = {
    "A evidence/source": ["source_faithfulness", "evidence_grounding"],
    "B status/correction": [
        "claim_status_recognition",
        "correction_accuracy",
        "misinformation_detection",
    ],
    "C resistance": ["framing_resistance", "error_rejection"],
    "D boundary": ["hallucination_control", "uncertainty_handling", "uncertainty"],
}

PROFILE_FAMILY_DEFINITIONS: dict[str, list[str]] = {
    "A evidence/source": ["source_faithfulness", "evidence_grounding"],
    "B status/correction": [
        "claim_status_recognition",
        "correction_accuracy",
        "misinformation_detection",
    ],
    "C resistance": ["framing_resistance", "error_rejection"],
    # The compact construct profile does not carry the free-form uncertainty
    # score for all backends, so boundary is restricted to the two binary
    # boundary indicators that are consistently available.
    "D boundary": ["hallucination_control", "uncertainty_handling"],
}

BACKEND_ORDER = [
    "DeepSeek V4-Pro",
    "Gemma3-12B",
    "Qwen2.5-7B",
    "Qwen3-14B",
    "Rule-based heuristic",
]
TARGET_ORDER = ["Qwen target", "Llama target"]


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.3f}"
    except Exception:
        return "--"


def _backend_label(value: object) -> str:
    text = str(value)
    lower = text.lower()
    if "deepseek" in lower:
        return "DeepSeek V4-Pro"
    if "gemma" in lower:
        return "Gemma3-12B"
    if "qwen3" in lower:
        return "Qwen3-14B"
    if "qwen2.5-7b" in lower:
        return "Qwen2.5-7B"
    if "rule-based" in lower or "heuristic" in lower:
        return "Rule-based heuristic"
    return text


def _target_label(value: object) -> str:
    text = str(value)
    if text == "target_qwen_only":
        return "Qwen target"
    if text == "target_llama_only":
        return "Llama target"
    if text == "pooled_qwen_llama":
        return "Pooled"
    if "Qwen" in text:
        return "Qwen target"
    if "Llama" in text:
        return "Llama target"
    return text


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["backend_sort"] = out["backend_paper"].map({v: i for i, v in enumerate(BACKEND_ORDER)}).fillna(99)
    out["target_sort"] = out["target_paper"].map({v: i for i, v in enumerate(TARGET_ORDER)}).fillna(99)
    return out.sort_values(["backend_sort", "target_sort"]).drop(columns=["backend_sort", "target_sort"])


def _family_means(df: pd.DataFrame, family_map: dict[str, list[str]], absolute: bool) -> pd.DataFrame:
    out = df.copy()
    for family, cols in family_map.items():
        present = [col for col in cols if col in out.columns]
        if not present:
            out[family] = np.nan
            continue
        values = out[present].apply(pd.to_numeric, errors="coerce")
        if absolute:
            values = values.abs()
        out[family] = values.mean(axis=1)
    return out


def _save_heatmap(df: pd.DataFrame, path: Path, title: str, cbar_label: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    if df.empty:
        return
    save_paper_heatmap(
        df,
        path,
        title,
        cbar_label,
        vmin=vmin,
        vmax=vmax,
        pretty_columns=False,
        pretty_index=False,
        figsize=(8.5, max(4.5, 0.42 * len(df.index) + 1.8)),
        xrotation=25,
    )


def _write_latex_table(path: Path, df: pd.DataFrame, *, caption_note: str = "") -> None:
    families = list(FAMILY_DEFINITIONS)
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{Evaluator backend} & \textbf{Target split} & \textbf{PC1 var.} & \textbf{PC1--PC4 var.} & \textbf{A} & \textbf{B} & \textbf{C} & \textbf{D} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["backend_paper"]),
                    _latex_escape(row["target_paper"]),
                    _fmt(row.get("pc1_variance")),
                    _fmt(row.get("pc1_pc4_cumulative_variance")),
                    *[_fmt(row.get(family)) for family in families],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if caption_note:
        lines.append("% " + caption_note)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_profile_latex_table(path: Path, df: pd.DataFrame) -> None:
    families = list(PROFILE_FAMILY_DEFINITIONS)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Evaluator backend} & \textbf{Target split} & \textbf{Rows} & \textbf{A} & \textbf{B} & \textbf{C} & \textbf{D} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["backend_paper"]),
                    _latex_escape(row["target_paper"]),
                    f"{int(row['rows']):,}" if pd.notna(row.get("rows")) else "--",
                    *[_fmt(row.get(family)) for family in families],
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% A=evidence/source, B=status/correction, C=resistance, D=boundary.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pc1_family_table() -> pd.DataFrame:
    loadings = pd.read_csv(AUDIT / "paper_pc1_loading_heatmap_data.csv")
    pca = pd.read_csv(AUDIT / "protocol_backend_pca_summary.csv")
    pca = pca[
        pca["evaluator_variant"].eq(FULL_CALE)
        & pca["target_split"].isin(["target_qwen_only", "target_llama_only"])
    ].copy()
    pca["backend_paper"] = pca["backend_short"].map(_backend_label)
    pca["target_paper"] = pca["target_split"].map(_target_label)

    loadings = loadings[loadings["target_split"].isin(["Qwen target", "Llama target"])].copy()
    loadings["backend_paper"] = loadings["backend"].map(_backend_label)
    loadings["target_paper"] = loadings["target_split"].map(_target_label)
    loadings = _family_means(loadings, FAMILY_DEFINITIONS, absolute=True)

    families = list(FAMILY_DEFINITIONS)
    table = loadings[["backend_paper", "target_paper", *families]].merge(
        pca[
            [
                "backend_paper",
                "target_paper",
                "rows",
                "pc1_variance",
                "pc1_pc4_cumulative_variance",
            ]
        ],
        on=["backend_paper", "target_paper"],
        how="left",
    )
    table = _ordered(table[["backend_paper", "target_paper", "rows", "pc1_variance", "pc1_pc4_cumulative_variance", *families]])
    table["dominant_family"] = table[families].idxmax(axis=1)
    table.to_csv(AUDIT / "paper_table_fixed_family_pc1_loadings.csv", index=False)
    _write_latex_table(
        PAPER_TABLE_DIR / "target_split_construct_family_pc1.tex",
        table,
        caption_note="A=evidence/source, B=status/correction, C=resistance, D=boundary.",
    )

    heat = table.copy()
    heat.index = heat["backend_paper"] + "\n" + heat["target_paper"]
    _save_heatmap(
        heat[families],
        AUDIT / "paper_construct_family_pc1_loading_heatmap.png",
        "Fixed-order PC1 loading magnitudes by construct family",
        "Mean |PC1 loading|",
        vmin=0,
        vmax=max(0.7, float(np.nanmax(heat[families].to_numpy()))),
    )
    return table


def build_construct_profile_table() -> pd.DataFrame:
    profile = pd.read_csv(AUDIT / "full_cale_construct_profile_by_backend_protocol_target_split.csv")
    profile = profile[profile["target_split"].isin(["target_qwen_only", "target_llama_only"])].copy()
    profile["backend_paper"] = profile["evaluator_backend_label"].map(_backend_label)
    profile["target_paper"] = profile["target_split"].map(_target_label)
    profile = _family_means(profile, PROFILE_FAMILY_DEFINITIONS, absolute=False)
    families = list(PROFILE_FAMILY_DEFINITIONS)
    table = _ordered(profile[["backend_paper", "target_paper", "rows", *families]].copy())
    table.to_csv(AUDIT / "paper_table_construct_family_profile_by_backend_target.csv", index=False)
    _write_profile_latex_table(PAPER_TABLE_DIR / "compact_construct_family_profile.tex", table)

    heat = table.copy()
    heat.index = heat["backend_paper"] + "\n" + heat["target_paper"]
    _save_heatmap(
        heat[families],
        AUDIT / "paper_construct_family_profile_heatmap.png",
        "Full CALE target-response diagnostic profile by construct family",
        "Mean construct-family score",
        vmin=0,
        vmax=1,
    )
    return table


def main() -> None:
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    loading_table = build_pc1_family_table()
    profile_table = build_construct_profile_table()

    for name in [
        "paper_construct_family_pc1_loading_heatmap.png",
        "paper_construct_family_profile_heatmap.png",
    ]:
        src = AUDIT / name
        dst = PAPER_FIGURE_DIR / name
        if src.exists():
            dst.write_bytes(src.read_bytes())

    print("Wrote:")
    print(f"- {AUDIT / 'paper_table_fixed_family_pc1_loadings.csv'}")
    print(f"- {AUDIT / 'paper_construct_family_pc1_loading_heatmap.png'}")
    print(f"- {AUDIT / 'paper_table_construct_family_profile_by_backend_target.csv'}")
    print(f"- {AUDIT / 'paper_construct_family_profile_heatmap.png'}")
    print(f"- {PAPER_TABLE_DIR / 'target_split_construct_family_pc1.tex'}")
    print(f"- {PAPER_TABLE_DIR / 'compact_construct_family_profile.tex'}")
    print("\nPC1 family loading preview:")
    print(loading_table.round(3).to_string(index=False))
    print("\nConstruct family profile preview:")
    print(profile_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
