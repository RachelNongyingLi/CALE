#!/usr/bin/env python3
"""Infer recurrent PC-derived structure types from CALE PCA loadings.

This script is intentionally different from a fixed PC1 loading table. It
treats every Full-CALE component (PC1--PC4) from every evaluator backend and
target split as a component instance, maps its variable loadings into a small
family coordinate system, and then assigns a descriptive structure type.

The output is a screening taxonomy, not a formal factor solution. The purpose
is to ask whether similar component structures recur across backends and target
splits, and whether those recurrent structures can be used as diagnostic lenses
for target-response profiles.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("figures/.matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "figures" / "global_evaluator_audit"
PAPER_TABLE_DIR = ROOT / "figures" / "paper_tables"
PAPER_FIGURE_DIR = ROOT / "figures" / "paper_appendix"

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
    "D boundary": ["hallucination_control", "uncertainty_handling"],
}

FAMILY_SHORT = {
    "A evidence/source": "A",
    "B status/correction": "B",
    "C resistance": "C",
    "D boundary": "D",
}

TYPE_ORDER = [
    "S1 broad CALE-general",
    "S2 evidence/source",
    "S3 status/correction",
    "S4 resistance",
    "S5 boundary",
]

TYPE_LABELS = {
    "S1 broad CALE-general": "Broad CALE-general structure",
    "S2 evidence/source": "Evidence/source structure",
    "S3 status/correction": "Claim-status/correction structure",
    "S4 resistance": "Framing/error-resistance structure",
    "S5 boundary": "Boundary/uncertainty structure",
}

VARIABLE_LABELS = {
    "source_faithfulness": "source faithfulness",
    "evidence_grounding": "evidence grounding",
    "claim_status_recognition": "claim-status recognition",
    "correction_accuracy": "correction accuracy",
    "misinformation_detection": "misinformation detection",
    "framing_resistance": "framing resistance",
    "error_rejection": "error rejection",
    "hallucination_control": "hallucination control",
    "uncertainty_handling": "uncertainty handling",
    "uncertainty": "uncertainty",
}

BACKEND_ORDER = [
    "DeepSeek V4-Pro",
    "Gemma3-12B",
    "Qwen2.5-7B",
    "Qwen3-14B",
    "Rule-based heuristic",
]
SPLIT_ORDER = ["Pooled", "Qwen target", "Llama target"]
PC_ORDER = ["PC1", "PC2", "PC3", "PC4"]


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
    if "heuristic" in lower or "rule-based" in lower:
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
    return text


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


def _parse_loading_file(path: Path) -> tuple[str, str]:
    stem = path.name
    for suffix in ["_full_cale_pca_loadings.csv", "_full_cale_pca_explained_variance.csv"]:
        stem = stem.removesuffix(suffix)
    for split in ["pooled_qwen_llama", "target_qwen_only", "target_llama_only"]:
        suffix = "_" + split
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    raise ValueError(f"Cannot parse backend/split from {path.name}")


def _family_weights(loadings: pd.Series, family_map: dict[str, list[str]]) -> dict[str, float]:
    magnitudes = loadings.abs()
    raw = {}
    for family, variables in family_map.items():
        present = [var for var in variables if var in magnitudes.index]
        raw[family] = float(magnitudes[present].mean()) if present else np.nan
    total = sum(value for value in raw.values() if np.isfinite(value))
    return {family: (value / total if np.isfinite(value) and total else np.nan) for family, value in raw.items()}


def _classify_structure(weights: dict[str, float]) -> str:
    valid = [(family, value) for family, value in weights.items() if np.isfinite(value)]
    if not valid:
        return "unclassified"
    ranked = sorted(valid, key=lambda item: item[1], reverse=True)
    top_family, top_value = ranked[0]
    second_value = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_value - second_value

    # With four family coordinates, a perfectly broad component has 0.25 in
    # each family. We call a component broad if its largest family share remains
    # close to uniform or if the leading families are nearly tied. These
    # thresholds are descriptive screeners, not inferential cutoffs.
    if top_value < 0.30 or margin < 0.035:
        return "S1 broad CALE-general"
    if top_family == "A evidence/source":
        return "S2 evidence/source"
    if top_family == "B status/correction":
        return "S3 status/correction"
    if top_family == "C resistance":
        return "S4 resistance"
    if top_family == "D boundary":
        return "S5 boundary"
    return "unclassified"


def _top_variables(loadings: pd.Series, n: int = 3) -> str:
    variables = loadings.abs().sort_values(ascending=False).head(n).index.tolist()
    return ", ".join(VARIABLE_LABELS.get(variable, variable) for variable in variables)


def _component_instances() -> pd.DataFrame:
    variance_summary = pd.read_csv(AUDIT / "protocol_backend_pca_summary.csv")
    variance_summary = variance_summary[
        variance_summary["evaluator_variant"].eq("full_attack_aware_cale")
    ].copy()
    variance_by_key = {}
    for _, row in variance_summary.iterrows():
        key = (_backend_label(row["backend_short"]), row["target_split"])
        variance_by_key[key] = {
            "rows": row.get("rows"),
            "pc1_variance": row.get("pc1_variance"),
            "pc1_pc4_cumulative_variance": row.get("pc1_pc4_cumulative_variance"),
        }

    evr_by_file: dict[tuple[str, str], dict[str, float]] = {}
    for path in AUDIT.glob("*_full_cale_pca_explained_variance.csv"):
        backend_slug, split = _parse_loading_file(path)
        evr = pd.read_csv(path)
        evr_by_file[(_backend_label(backend_slug), split)] = dict(
            zip(evr["component"], evr["explained_variance_ratio"])
        )

    rows = []
    for path in sorted(AUDIT.glob("*_full_cale_pca_loadings.csv")):
        backend_slug, split = _parse_loading_file(path)
        backend = _backend_label(backend_slug)
        target = _target_label(split)
        loadings = pd.read_csv(path, index_col=0)
        for component in [pc for pc in PC_ORDER if pc in loadings.columns]:
            vector = pd.to_numeric(loadings[component], errors="coerce")
            weights = _family_weights(vector, FAMILY_DEFINITIONS)
            structure_type = _classify_structure(weights)
            rows.append(
                {
                    "backend_paper": backend,
                    "target_split": split,
                    "target_paper": target,
                    "component": component,
                    "explained_variance_ratio": evr_by_file.get((backend, split), {}).get(component, np.nan),
                    "structure_type": structure_type,
                    "structure_label": TYPE_LABELS.get(structure_type, structure_type),
                    "top_variables": _top_variables(vector),
                    **weights,
                    **variance_by_key.get((backend, split), {}),
                }
            )
    out = pd.DataFrame(rows)
    out["backend_sort"] = out["backend_paper"].map({v: i for i, v in enumerate(BACKEND_ORDER)}).fillna(99)
    out["split_sort"] = out["target_paper"].map({v: i for i, v in enumerate(SPLIT_ORDER)}).fillna(99)
    out["pc_sort"] = out["component"].map({v: i for i, v in enumerate(PC_ORDER)}).fillna(99)
    return out.sort_values(["backend_sort", "split_sort", "pc_sort"]).drop(
        columns=["backend_sort", "split_sort", "pc_sort"]
    )


def _structure_summary(instances: pd.DataFrame) -> pd.DataFrame:
    families = list(FAMILY_DEFINITIONS)
    rows = []
    for structure_type in TYPE_ORDER:
        group = instances[instances["structure_type"].eq(structure_type)].copy()
        if group.empty:
            continue
        component_counts = group["component"].value_counts().reindex(PC_ORDER).dropna().astype(int)
        backend_counts = group["backend_paper"].value_counts().reindex(BACKEND_ORDER).dropna().astype(int)
        target_counts = group["target_paper"].value_counts().reindex(SPLIT_ORDER).dropna().astype(int)
        example = group.sort_values("explained_variance_ratio", ascending=False).iloc[0]
        row = {
            "structure_type": structure_type,
            "structure_label": TYPE_LABELS[structure_type],
            "n_component_instances": len(group),
            "appears_as_components": ", ".join(f"{k}={v}" for k, v in component_counts.items()),
            "appears_in_backends": ", ".join(f"{k}={v}" for k, v in backend_counts.items()),
            "appears_in_target_splits": ", ".join(f"{k}={v}" for k, v in target_counts.items()),
            "example_component": f"{example['backend_paper']} / {example['target_paper']} / {example['component']}",
            "example_top_variables": example["top_variables"],
            "mean_explained_variance_ratio": group["explained_variance_ratio"].mean(),
        }
        for family in families:
            row[family] = group[family].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _construct_profile() -> pd.DataFrame:
    profile = pd.read_csv(AUDIT / "full_cale_construct_profile_by_backend_protocol_target_split.csv")
    profile = profile[profile["target_split"].isin(["target_qwen_only", "target_llama_only"])].copy()
    profile["backend_paper"] = profile["evaluator_backend_label"].map(_backend_label)
    profile["target_paper"] = profile["target_split"].map(_target_label)
    for family, variables in PROFILE_FAMILY_DEFINITIONS.items():
        present = [var for var in variables if var in profile.columns]
        profile[family] = profile[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    profile["backend_sort"] = profile["backend_paper"].map({v: i for i, v in enumerate(BACKEND_ORDER)}).fillna(99)
    profile["split_sort"] = profile["target_paper"].map({"Qwen target": 0, "Llama target": 1}).fillna(99)
    return profile.sort_values(["backend_sort", "split_sort"]).drop(columns=["backend_sort", "split_sort"])


def _structure_expression(profile: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    families = list(PROFILE_FAMILY_DEFINITIONS)
    prototypes = summary.set_index("structure_type")[families].copy()
    prototypes = prototypes.div(prototypes.sum(axis=1), axis=0)
    rows = []
    for _, row in profile.iterrows():
        out = {
            "backend_paper": row["backend_paper"],
            "target_paper": row["target_paper"],
            "rows": row["rows"],
        }
        scores = row[families].astype(float)
        for structure_type in TYPE_ORDER:
            if structure_type not in prototypes.index:
                continue
            weights = prototypes.loc[structure_type].astype(float)
            out[structure_type] = float((scores * weights).sum())
        rows.append(out)
    return pd.DataFrame(rows)


def _save_heatmap(df: pd.DataFrame, path: Path, title: str, cbar_label: str, vmin: float = 0.0, vmax: float | None = None) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    values = df.to_numpy(dtype=float)
    if vmax is None:
        vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.42 * len(df.index) + 1.8)))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    threshold = vmin + (vmax - vmin) * 0.55
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(
                j,
                i,
                "--" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if np.isfinite(value) and value > threshold else "black",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _save_occurrence_heatmap(instances: pd.DataFrame) -> pd.DataFrame:
    target_instances = instances[instances["target_paper"].isin(["Qwen target", "Llama target"])].copy()
    target_instances["row_label"] = target_instances["backend_paper"] + "\n" + target_instances["target_paper"]
    occurrence = (
        target_instances.pivot_table(
            index="row_label",
            columns="component",
            values="structure_type",
            aggfunc=lambda values: next(iter(values)),
        )
        .reindex(columns=PC_ORDER)
    )
    codes = {structure_type: i + 1 for i, structure_type in enumerate(TYPE_ORDER)}
    numeric = occurrence.replace(codes).astype(float)
    path = AUDIT / "paper_pc_structure_type_occurrence_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.48 * len(numeric.index) + 1.8)))
    im = ax.imshow(numeric.to_numpy(), aspect="auto", cmap="Set3", vmin=1, vmax=len(codes))
    ax.set_title("PC-derived structure types across target splits", fontsize=13, pad=10)
    ax.set_xticks(range(len(numeric.columns)))
    ax.set_xticklabels(numeric.columns)
    ax.set_yticks(range(len(numeric.index)))
    ax.set_yticklabels(numeric.index, fontsize=9)
    short = {
        "S1 broad CALE-general": "S1",
        "S2 evidence/source": "S2",
        "S3 status/correction": "S3",
        "S4 resistance": "S4",
        "S5 boundary": "S5",
    }
    for i, row in enumerate(occurrence.index):
        for j, col in enumerate(occurrence.columns):
            value = occurrence.loc[row, col]
            ax.text(j, i, short.get(str(value), "--"), ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=list(codes.values()))
    cbar.ax.set_yticklabels([key.split(" ", 1)[0] for key in codes])
    cbar.set_label("Structure type")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    occurrence.to_csv(AUDIT / "paper_pc_structure_type_occurrence.csv")
    return occurrence


def _write_structure_summary_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{p{0.14\textwidth}p{0.27\textwidth}rp{0.19\textwidth}p{0.24\textwidth}}",
        r"\toprule",
        r"\textbf{Type} & \textbf{Interpretation} & \textbf{N} & \textbf{Common PCs} & \textbf{Example top variables} \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["structure_type"].split(" ", 1)[0]),
                    _latex_escape(row["structure_label"]),
                    str(int(row["n_component_instances"])),
                    _latex_escape(row["appears_as_components"]),
                    _latex_escape(row["example_top_variables"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_expression_latex(expression: pd.DataFrame, path: Path) -> None:
    present_types = [t for t in TYPE_ORDER if t in expression.columns]
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{Evaluator backend} & \textbf{Target split} & \textbf{Rows} & "
        + " & ".join(r"\textbf{" + t.split(" ", 1)[0] + "}" for t in present_types)
        + r" \\",
        r"\midrule",
    ]
    for _, row in expression.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["backend_paper"]),
                    _latex_escape(row["target_paper"]),
                    f"{int(row['rows']):,}",
                    *[_fmt(row.get(t)) for t in present_types],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    instances = _component_instances()
    instances.to_csv(AUDIT / "paper_pc_structure_type_instances.csv", index=False)

    summary = _structure_summary(instances)
    summary.to_csv(AUDIT / "paper_pc_structure_type_summary.csv", index=False)

    _save_occurrence_heatmap(instances)

    families = list(FAMILY_DEFINITIONS)
    proto = summary.set_index("structure_type")[families].reindex(TYPE_ORDER).dropna(how="all")
    _save_heatmap(
        proto,
        AUDIT / "paper_pc_structure_type_prototypes_heatmap.png",
        "Structure type prototypes in family-loading space",
        "Mean family loading share",
        vmin=0,
        vmax=0.65,
    )

    profile = _construct_profile()
    expression = _structure_expression(profile, summary)
    expression.to_csv(AUDIT / "paper_structure_type_expression_by_backend_target.csv", index=False)
    heat = expression.copy()
    heat.index = heat["backend_paper"] + "\n" + heat["target_paper"]
    _save_heatmap(
        heat[[t for t in TYPE_ORDER if t in heat.columns]],
        AUDIT / "paper_structure_type_expression_heatmap.png",
        "Target-response profile under recurrent PC-derived structure types",
        "Prototype-weighted construct score",
        vmin=0,
        vmax=1,
    )

    _write_structure_summary_latex(summary, PAPER_TABLE_DIR / "pc_structure_type_summary.tex")
    _write_expression_latex(expression, PAPER_TABLE_DIR / "structure_type_expression_profile.tex")

    for name in [
        "paper_pc_structure_type_occurrence_heatmap.png",
        "paper_pc_structure_type_prototypes_heatmap.png",
        "paper_structure_type_expression_heatmap.png",
    ]:
        src = AUDIT / name
        if src.exists():
            (PAPER_FIGURE_DIR / name).write_bytes(src.read_bytes())

    print("Wrote structure-type outputs:")
    for name in [
        "paper_pc_structure_type_instances.csv",
        "paper_pc_structure_type_summary.csv",
        "paper_pc_structure_type_occurrence.csv",
        "paper_pc_structure_type_occurrence_heatmap.png",
        "paper_pc_structure_type_prototypes_heatmap.png",
        "paper_structure_type_expression_by_backend_target.csv",
        "paper_structure_type_expression_heatmap.png",
    ]:
        print(f"- {AUDIT / name}")
    print(f"- {PAPER_TABLE_DIR / 'pc_structure_type_summary.tex'}")
    print(f"- {PAPER_TABLE_DIR / 'structure_type_expression_profile.tex'}")
    print("\nStructure summary:")
    print(summary.round(3).to_string(index=False))
    print("\nExpression preview:")
    print(expression.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
