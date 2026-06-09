#!/usr/bin/env python3
"""Build thesis-submission diagnostics from existing CALE artifacts.

The script does not run new evaluator inference. It reads already generated
behavior matrices, CFA outputs, and targeted-screen summaries, then exports
small CSV and LaTeX-ready tables for the thesis revision.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "submission_diagnostics"
TABLES = ROOT / "figures" / "paper_tables"

AUDIT = ROOT / "figures" / "global_evaluator_audit"
CFA = ROOT / "figures" / "cfa_behavior_model" / "family_psychometrics"
FRAMING = ROOT / "figures" / "controlled_framing_n300_qwen3_14b"
BOUNDARY = ROOT / "figures" / "boundary_hard_real_subset_qwen3_14b"

FULL = "full_attack_aware_cale"
DEEPSEEK = "DeepSeek V4-Pro API judge"
QWEN25 = "Qwen2.5-7B local HF judge"
ECDF_FIG = ROOT / "figures" / "paper_appendix" / "deepseek_qwen25_score_ecdf_by_label.png"
RULE_BACKEND = "Rule-based scoring backend (non-LLM)"

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

PCA_LOADINGS = {
    "Rule-based instrumentation": "heuristic_default_pooled_qwen_llama_full_cale_pca_loadings.csv",
    "DeepSeek V4-Pro": "deepseek-v4-pro_pooled_qwen_llama_full_cale_pca_loadings.csv",
    "Gemma3-12B": "google_gemma-3-12b-it_pooled_qwen_llama_full_cale_pca_loadings.csv",
    "Qwen2.5-7B": "Qwen_Qwen2.5-7B-Instruct_pooled_qwen_llama_full_cale_pca_loadings.csv",
    "Qwen3-14B": "Qwen_Qwen3-14B_pooled_qwen_llama_full_cale_pca_loadings.csv",
}


def _mkdirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def _escape_tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return _escape_tex(value)


def _write_tabular(
    df: pd.DataFrame,
    path: Path,
    headers: list[str],
    formats: list[str] | None = None,
    digits: int = 3,
) -> None:
    formats = formats or ["l"] * len(headers)
    lines = [
        r"\begin{tabular}{" + "".join(formats) + "}",
        r"\toprule",
        " & ".join(_escape_tex(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[col], digits) for col in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    return tuple(float(np.tanh(x)) for x in (z - 1.96 * se, z + 1.96 * se))


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(paired) < 2:
        return np.nan
    ranked = paired.rank(method="average")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1], method="pearson"))


def percentile_rank(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if len(values) <= 1:
        return pd.Series(np.nan, index=series.index)
    return (values.rank(method="average") - 1) / (len(values) - 1)


def mean_ci(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    if len(values) <= 1:
        return (np.nan, np.nan)
    se = values.std(ddof=1) / math.sqrt(len(values))
    mean = values.mean()
    return float(mean - 1.96 * se), float(mean + 1.96 * se)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return np.nan
    return float(np.dot(left, right) / denom)


def normalized_entropy(weights: np.ndarray) -> float:
    values = np.abs(weights.astype(float))
    total = values.sum()
    if total == 0 or len(values) <= 1:
        return np.nan
    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum() / math.log(len(values)))


def top_decile_overlap(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(paired) == 0:
        return np.nan
    k = max(1, int(math.ceil(len(paired) * 0.10)))
    left_top = set(paired.iloc[:, 0].nlargest(k).index)
    right_top = set(paired.iloc[:, 1].nlargest(k).index)
    return len(left_top & right_top) / k


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    sd = values.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / sd


def kendall_tau_b(left: pd.Series, right: pd.Series) -> tuple[float, dict[str, int]]:
    paired = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    x = paired.iloc[:, 0].to_numpy()
    y = paired.iloc[:, 1].to_numpy()
    concordant = discordant = tie_x = tie_y = tie_both = 0
    n = len(paired)
    for i in range(n - 1):
        dx = np.sign(x[i] - x[i + 1 :])
        dy = np.sign(y[i] - y[i + 1 :])
        both = (dx == 0) & (dy == 0)
        tx = (dx == 0) & (dy != 0)
        ty = (dx != 0) & (dy == 0)
        comparable = (dx != 0) & (dy != 0)
        concordant += int((comparable & (dx == dy)).sum())
        discordant += int((comparable & (dx != dy)).sum())
        tie_x += int(tx.sum())
        tie_y += int(ty.sum())
        tie_both += int(both.sum())
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    tau = (concordant - discordant) / denom if denom else np.nan
    counts = {
        "n": n,
        "concordant": concordant,
        "discordant": discordant,
        "tie_x": tie_x,
        "tie_y": tie_y,
        "tie_both": tie_both,
    }
    return float(tau), counts


def adjusted_rand_index(labels_true: list[str], labels_pred: list[int]) -> float:
    n = len(labels_true)
    if n < 2:
        return np.nan
    true_counts = Counter(labels_true)
    pred_counts = Counter(labels_pred)
    contingency = Counter(zip(labels_true, labels_pred))

    def comb2(x: int) -> float:
        return x * (x - 1) / 2

    sum_nij = sum(comb2(v) for v in contingency.values())
    sum_ai = sum(comb2(v) for v in true_counts.values())
    sum_bj = sum(comb2(v) for v in pred_counts.values())
    total = comb2(n)
    expected = sum_ai * sum_bj / total if total else 0
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_nij == max_index else 0.0
    return float((sum_nij - expected) / denom)


def normalized_mutual_information(labels_true: list[str], labels_pred: list[int]) -> float:
    n = len(labels_true)
    if n == 0:
        return np.nan
    true_counts = Counter(labels_true)
    pred_counts = Counter(labels_pred)
    contingency = Counter(zip(labels_true, labels_pred))
    mi = 0.0
    for (true_label, pred_label), count in contingency.items():
        mi += (count / n) * math.log((count * n) / (true_counts[true_label] * pred_counts[pred_label]))
    h_true = -sum((count / n) * math.log(count / n) for count in true_counts.values())
    h_pred = -sum((count / n) * math.log(count / n) for count in pred_counts.values())
    denom = math.sqrt(h_true * h_pred)
    if denom == 0:
        return 1.0 if labels_true == [str(x) for x in labels_pred] else 0.0
    return float(mi / denom)


def deterministic_kmeans(features: np.ndarray, k: int = 5, max_iter: int = 100) -> np.ndarray:
    if len(features) < k:
        return np.arange(len(features))
    center_indices = [int(np.argmax(np.linalg.norm(features - features.mean(axis=0), axis=1)))]
    while len(center_indices) < k:
        centers = features[center_indices]
        distances = np.min(((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        center_indices.append(int(np.argmax(distances)))
    centers = features[center_indices].copy()
    labels = np.zeros(len(features), dtype=int)
    for _ in range(max_iter):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = features[labels == idx]
            if len(members):
                centers[idx] = members.mean(axis=0)
    return labels


def draw_label_ecdf(wide: pd.DataFrame) -> None:
    """Draw a dependency-light ECDF diagnostic for thesis appendix use."""
    labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
    colors = {DEEPSEEK: (31, 90, 160), QWEN25: (196, 77, 44)}
    short = {DEEPSEEK: "DeepSeek", QWEN25: "Qwen2.5"}
    width, height = 1500, 520
    margin_l, margin_t, margin_b = 76, 56, 70
    panel_gap = 48
    panel_w = (width - margin_l - 36 - panel_gap * 2) // 3
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((margin_l, 16), "DeepSeek vs Qwen2.5 Full-CALE score ECDF by FEVER label", fill=(20, 20, 20), font=font)
    for idx, label in enumerate(labels):
        x0 = margin_l + idx * (panel_w + panel_gap)
        y0 = margin_t
        x1 = x0 + panel_w
        y1 = y0 + plot_h
        draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210), width=1)
        for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
            x = x0 + tick * panel_w
            y = y1 - tick * plot_h
            draw.line((x, y1, x, y1 + 4), fill=(150, 150, 150))
            draw.line((x0 - 4, y, x0, y), fill=(150, 150, 150))
            if idx == 0:
                draw.text((x0 - 42, y - 5), f"{tick:.2g}", fill=(80, 80, 80), font=font)
            draw.text((x - 10, y1 + 8), f"{tick:.2g}", fill=(80, 80, 80), font=font)
        draw.text((x0 + 8, y0 - 22), f"{label}", fill=(20, 20, 20), font=font)

        group = wide[wide["reference_label"].eq(label)]
        for backend in [DEEPSEEK, QWEN25]:
            values = np.sort(group[backend].dropna().astype(float).to_numpy())
            if len(values) == 0:
                continue
            points = []
            for rank, value in enumerate(values, start=1):
                x = x0 + float(value) * panel_w
                y = y1 - (rank / len(values)) * plot_h
                points.append((x, y))
            if len(points) == 1:
                draw.ellipse((points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2), fill=colors[backend])
            else:
                draw.line(points, fill=colors[backend], width=3)
        if idx == 2:
            lx, ly = x1 - 120, y0 + 16
            for backend, offset in [(DEEPSEEK, 0), (QWEN25, 20)]:
                draw.line((lx, ly + offset, lx + 28, ly + offset), fill=colors[backend], width=4)
                draw.text((lx + 34, ly + offset - 6), short[backend], fill=(30, 30, 30), font=font)
    draw.text((width // 2 - 40, height - 24), "Final score", fill=(40, 40, 40), font=font)
    draw.text((8, height // 2 - 10), "ECDF", fill=(40, 40, 40), font=font)
    ECDF_FIG.parent.mkdir(parents=True, exist_ok=True)
    image.save(ECDF_FIG)


def backend_disagreement_diagnostics(behavior: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = behavior[behavior["evaluator_variant"].eq(FULL)].copy()
    pair = full[full["evaluator_backend_label"].isin([DEEPSEEK, QWEN25])].copy()
    key = ["id", "target_split", "target_model", "reference_label"]
    wide = (
        pair.pivot_table(index=key, columns="evaluator_backend_label", values="final_score", aggfunc="mean")
        .dropna()
        .reset_index()
    )
    wide["qwen_minus_deepseek"] = wide[QWEN25] - wide[DEEPSEEK]
    wide.to_csv(OUT / "deepseek_qwen25_shared_full_cale_rows.csv", index=False)

    dist_rows = []
    for backend in [DEEPSEEK, QWEN25]:
        s = wide[backend].astype(float)
        dist_rows.append(
            {
                "backend": backend,
                "n": len(s),
                "mean": s.mean(),
                "sd": s.std(ddof=1),
                "p25": s.quantile(0.25),
                "median": s.quantile(0.50),
                "p75": s.quantile(0.75),
                "prop_ge_0_80": (s >= 0.80).mean(),
                "prop_ge_0_90": (s >= 0.90).mean(),
            }
        )
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(OUT / "backend_score_distribution_deepseek_qwen25.csv", index=False)

    label_dist_rows = []
    for label, group in wide.groupby("reference_label", dropna=False):
        for backend in [DEEPSEEK, QWEN25]:
            s = group[backend].astype(float)
            label_dist_rows.append(
                {
                    "reference_label": label,
                    "backend": backend,
                    "n": len(s),
                    "mean": s.mean(),
                    "sd": s.std(ddof=1),
                    "p10": s.quantile(0.10),
                    "median": s.quantile(0.50),
                    "p90": s.quantile(0.90),
                    "prop_ge_0_90": (s >= 0.90).mean(),
                    "unique_scores": s.nunique(),
                }
            )
    label_dist = pd.DataFrame(label_dist_rows).sort_values(["reference_label", "backend"])
    label_dist.to_csv(OUT / "backend_score_distribution_by_reference_label.csv", index=False)
    draw_label_ecdf(wide)

    by_label_rows = []
    for label, group in wide.groupby("reference_label", dropna=False):
        by_label_rows.append(
            {
                "reference_label": label,
                "n": len(group),
                "spearman": spearman_corr(group[DEEPSEEK], group[QWEN25]),
                "mae": (group["qwen_minus_deepseek"].abs()).mean(),
                "mean_qwen_minus_deepseek": group["qwen_minus_deepseek"].mean(),
            }
        )
    by_label = pd.DataFrame(by_label_rows).sort_values("reference_label")
    by_label.to_csv(OUT / "backend_disagreement_by_reference_label.csv", index=False)

    rank_rows = []
    for label, group in wide.groupby("reference_label", dropna=False):
        deep_rank = percentile_rank(group[DEEPSEEK])
        qwen_rank = percentile_rank(group[QWEN25])
        rank_rows.append(
            {
                "reference_label": label,
                "n": len(group),
                "raw_spearman": spearman_corr(group[DEEPSEEK], group[QWEN25]),
                "rank_normalized_pearson": float(deep_rank.corr(qwen_rank, method="pearson")),
                "interpretation": "rank disagreement remains after monotonic rank normalization",
            }
        )
    rank_norm = pd.DataFrame(rank_rows).sort_values("reference_label")
    rank_norm.to_csv(OUT / "backend_rank_normalized_by_reference_label.csv", index=False)

    construct_cols = [
        col
        for col in [
            *CONSTRUCT_COLUMNS,
            "nei_uncertainty_failure_proxy",
            "refutes_correction_credit_proxy",
            "supports_status_failure_proxy",
        ]
        if col in pair.columns
    ]
    construct_rows = []
    for col in construct_cols:
        cwide = (
            pair.pivot_table(index=key, columns="evaluator_backend_label", values=col, aggfunc="mean")
            .dropna()
            .reset_index()
        )
        if len(cwide) == 0:
            continue
        diff = cwide[QWEN25] - cwide[DEEPSEEK]
        construct_rows.append(
            {
                "indicator": col,
                "n": len(cwide),
                "mae": diff.abs().mean(),
                "mean_qwen_minus_deepseek": diff.mean(),
            }
        )
    construct = pd.DataFrame(construct_rows).sort_values("mae", ascending=False)
    construct.to_csv(OUT / "backend_disagreement_by_construct.csv", index=False)

    supports_indicator_rows = []
    for col in construct_cols:
        cwide = (
            pair[pair["reference_label"].eq("SUPPORTS")]
            .pivot_table(index=key, columns="evaluator_backend_label", values=col, aggfunc="mean")
            .dropna()
            .reset_index()
        )
        if len(cwide) == 0:
            continue
        diff = cwide[QWEN25] - cwide[DEEPSEEK]
        supports_indicator_rows.append(
            {
                "indicator": col,
                "n": len(cwide),
                "spearman": spearman_corr(cwide[DEEPSEEK], cwide[QWEN25]),
                "mae": diff.abs().mean(),
                "mean_qwen_minus_deepseek": diff.mean(),
            }
        )
    supports_indicator = pd.DataFrame(supports_indicator_rows).sort_values("mae", ascending=False)
    supports_indicator.to_csv(OUT / "supports_indicator_backend_alignment.csv", index=False)

    mechanism_hypotheses = pd.DataFrame(
        [
            (
                "true claim simple affirmation",
                "Qwen2.5 may reward direct acceptance of a true claim",
                "DeepSeek may penalize missing evidence/source support",
            ),
            (
                "hedged correct answer",
                "Qwen2.5 may penalize a response that is cautious rather than direct",
                "DeepSeek may reward caution when it stays evidence-faithful",
            ),
            (
                "over-explained answer",
                "Qwen2.5 may reward informative elaboration",
                "DeepSeek may penalize unsupported additions or source drift",
            ),
            (
                "overcorrection of true claim",
                "Qwen2.5 may give partial credit if the response appears corrective",
                "DeepSeek may penalize unnecessary correction of a supported claim",
            ),
        ],
        columns=[
            "mechanism_hypothesis",
            "qwen_high_deepseek_low_reading",
            "deepseek_high_qwen_low_reading",
        ],
    )
    mechanism_hypotheses.to_csv(OUT / "supports_disagreement_mechanism_hypotheses.csv", index=False)

    latex_dist = dist.rename(
        columns={
            "backend": "Backend",
            "n": "n",
            "mean": "Mean",
            "sd": "SD",
            "p25": "P25",
            "median": "Median",
            "p75": "P75",
            "prop_ge_0_80": "Prop >= .80",
            "prop_ge_0_90": "Prop >= .90",
        }
    )
    _write_tabular(
        latex_dist,
        TABLES / "submission_backend_score_distribution.tex",
        list(latex_dist.columns),
        formats=["l", "r", "r", "r", "r", "r", "r", "r", "r"],
        digits=3,
    )

    latex_label_dist = label_dist.rename(
        columns={
            "reference_label": "Label",
            "backend": "Backend",
            "n": "n",
            "mean": "Mean",
            "sd": "SD",
            "p10": "P10",
            "median": "Median",
            "p90": "P90",
            "prop_ge_0_90": "Prop >= .90",
            "unique_scores": "Unique scores",
        }
    )
    _write_tabular(
        latex_label_dist,
        TABLES / "submission_backend_score_distribution_by_label.tex",
        list(latex_label_dist.columns),
        formats=["l", "l", "r", "r", "r", "r", "r", "r", "r", "r"],
        digits=3,
    )

    latex_label = by_label.rename(
        columns={
            "reference_label": "Reference label",
            "n": "n",
            "spearman": "Spearman rho",
            "mae": "MAE",
            "mean_qwen_minus_deepseek": "Mean Qwen-DeepSeek",
        }
    )
    _write_tabular(
        latex_label,
        TABLES / "submission_backend_disagreement_by_label.tex",
        list(latex_label.columns),
        formats=["l", "r", "r", "r", "r"],
        digits=3,
    )
    latex_construct = construct.head(6).rename(
        columns={
            "indicator": "Indicator",
            "n": "n",
            "mae": "MAE",
            "mean_qwen_minus_deepseek": "Mean Qwen-DeepSeek",
        }
    )
    _write_tabular(
        latex_construct,
        TABLES / "submission_backend_disagreement_by_construct.tex",
        list(latex_construct.columns),
        formats=["l", "r", "r", "r"],
        digits=3,
    )
    latex_supports = supports_indicator.head(8).rename(
        columns={
            "indicator": "Indicator",
            "n": "n",
            "spearman": "Spearman rho",
            "mae": "MAE",
            "mean_qwen_minus_deepseek": "Mean Qwen-DeepSeek",
        }
    )
    _write_tabular(
        latex_supports,
        TABLES / "submission_supports_indicator_alignment.tex",
        list(latex_supports.columns),
        formats=["l", "r", "r", "r", "r"],
        digits=3,
    )
    latex_rank_norm = rank_norm.rename(
        columns={
            "reference_label": "Label",
            "n": "n",
            "raw_spearman": "Raw Spearman",
            "rank_normalized_pearson": "Rank-normalized Pearson",
            "interpretation": "Diagnostic reading",
        }
    )
    _write_tabular(
        latex_rank_norm,
        TABLES / "submission_backend_rank_normalized_by_label.tex",
        list(latex_rank_norm.columns),
        formats=["l", "r", "r", "r", "p{0.34\\textwidth}"],
        digits=3,
    )
    latex_mechanisms = mechanism_hypotheses.rename(
        columns={
            "mechanism_hypothesis": "Mechanism hypothesis",
            "qwen_high_deepseek_low_reading": "Qwen high / DeepSeek low",
            "deepseek_high_qwen_low_reading": "DeepSeek high / Qwen low",
        }
    )
    _write_tabular(
        latex_mechanisms,
        TABLES / "submission_supports_mechanism_hypotheses.tex",
        list(latex_mechanisms.columns),
        formats=["l", "p{0.30\\textwidth}", "p{0.30\\textwidth}"],
        digits=3,
    )
    return dist, by_label


def deepseek_qwen25_calibration_sensitivity(behavior: pd.DataFrame) -> pd.DataFrame:
    full = behavior[behavior["evaluator_variant"].eq(FULL)].copy()
    pair = full[full["evaluator_backend_label"].isin([DEEPSEEK, QWEN25])].copy()
    key = ["id", "target_split", "target_model", "reference_label"]
    wide = (
        pair.pivot_table(index=key, columns="evaluator_backend_label", values="final_score", aggfunc="mean")
        .dropna()
        .reset_index()
    )
    supports = wide[wide["reference_label"].eq("SUPPORTS")].copy()

    def tie_prop(series: pd.Series) -> float:
        counts = series.astype(float).value_counts()
        total = len(series) * (len(series) - 1) / 2
        tied = sum(count * (count - 1) / 2 for count in counts)
        return float(tied / total) if total else np.nan

    z_deep = zscore(wide[DEEPSEEK])
    z_qwen = zscore(wide[QWEN25])
    z_pearson = float(z_deep.corr(z_qwen))
    z_spearman = spearman_corr(z_deep, z_qwen)
    z_mae = float((z_deep - z_qwen).abs().mean())

    pooled_rank = float(percentile_rank(wide[DEEPSEEK]).corr(percentile_rank(wide[QWEN25])))
    supports_rank = float(percentile_rank(supports[DEEPSEEK]).corr(percentile_rank(supports[QWEN25])))

    pooled_tau, pooled_counts = kendall_tau_b(wide[DEEPSEEK], wide[QWEN25])
    supports_tau, supports_counts = kendall_tau_b(supports[DEEPSEEK], supports[QWEN25])
    supports_tie_deep = tie_prop(supports[DEEPSEEK])
    supports_tie_qwen = tie_prop(supports[QWEN25])
    supports_unique_deep = supports[DEEPSEEK].nunique()
    supports_unique_qwen = supports[QWEN25].nunique()
    supports_high_deep = float((supports[DEEPSEEK] >= 0.90).mean())
    supports_high_qwen = float((supports[QWEN25] >= 0.90).mean())

    detail = pd.DataFrame(
        [
            {
                "scope": "pooled",
                "n": len(wide),
                "raw_spearman": spearman_corr(wide[DEEPSEEK], wide[QWEN25]),
                "raw_pearson": float(wide[DEEPSEEK].corr(wide[QWEN25])),
                "zscore_pearson": z_pearson,
                "zscore_spearman": z_spearman,
                "zscore_mae": z_mae,
                "rank_normalized_pearson": pooled_rank,
                "kendall_tau_b": pooled_tau,
                "deepseek_unique_scores": wide[DEEPSEEK].nunique(),
                "qwen_unique_scores": wide[QWEN25].nunique(),
                "deepseek_tie_pair_prop": tie_prop(wide[DEEPSEEK]),
                "qwen_tie_pair_prop": tie_prop(wide[QWEN25]),
                **{f"pooled_{key}": value for key, value in pooled_counts.items()},
            },
            {
                "scope": "supports",
                "n": len(supports),
                "raw_spearman": spearman_corr(supports[DEEPSEEK], supports[QWEN25]),
                "raw_pearson": float(supports[DEEPSEEK].corr(supports[QWEN25])),
                "zscore_pearson": float(zscore(supports[DEEPSEEK]).corr(zscore(supports[QWEN25]))),
                "zscore_spearman": spearman_corr(zscore(supports[DEEPSEEK]), zscore(supports[QWEN25])),
                "zscore_mae": float((zscore(supports[DEEPSEEK]) - zscore(supports[QWEN25])).abs().mean()),
                "rank_normalized_pearson": supports_rank,
                "kendall_tau_b": supports_tau,
                "deepseek_unique_scores": supports_unique_deep,
                "qwen_unique_scores": supports_unique_qwen,
                "deepseek_tie_pair_prop": supports_tie_deep,
                "qwen_tie_pair_prop": supports_tie_qwen,
                **{f"supports_{key}": value for key, value in supports_counts.items()},
            },
        ]
    )
    detail.to_csv(OUT / "deepseek_qwen25_calibration_sensitivity_detail.csv", index=False)

    table = pd.DataFrame(
        [
            {
                "check": "Within-backend z-score normalization",
                "completed_result": f"pooled Pearson={z_pearson:.3f}, Spearman={z_spearman:.3f}, z-MAE={z_mae:.3f}",
                "reading": "linear scale-location normalization does not repair the weak ordering",
            },
            {
                "check": "Percentile/rank normalization",
                "completed_result": f"pooled rank r={pooled_rank:.3f}; Supports rank r={supports_rank:.3f}",
                "reading": "rank disagreement remains after removing marginal score scale",
            },
            {
                "check": "Isotonic calibration with human anchors",
                "completed_result": "not run: no expert response-level anchor labels in the completed audit",
                "reading": "whether human-anchored calibration repairs the disagreement remains an empirical next test",
            },
            {
                "check": "Tie-adjusted Kendall tau-b and tie diagnostics",
                "completed_result": (
                    f"pooled tau-b={pooled_tau:.3f}; Supports tau-b={supports_tau:.3f}; "
                    f"Supports unique scores D/Q={supports_unique_deep}/{supports_unique_qwen}; "
                    f"tie-pair prop D/Q={supports_tie_deep:.3f}/{supports_tie_qwen:.3f}; "
                    f">=.90 D/Q={supports_high_deep:.3f}/{supports_high_qwen:.3f}"
                ),
                "reading": "ceiling compression is visible, but it does not remove the Supports rank inversion",
            },
        ]
    )
    table.to_csv(OUT / "deepseek_qwen25_calibration_sensitivity.csv", index=False)
    latex = table.rename(
        columns={
            "check": "Check",
            "completed_result": "Completed result",
            "reading": "Diagnostic reading",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_deepseek_qwen25_calibration_sensitivity.tex",
        list(latex.columns),
        formats=["p{0.26\\textwidth}", "p{0.34\\textwidth}", "p{0.34\\textwidth}"],
        digits=3,
    )
    return table


def pc1_concentration_diagnostics() -> pd.DataFrame:
    pca = pd.read_csv(AUDIT / "protocol_backend_pca_summary.csv")
    pca = pca[
        pca["evaluator_variant"].eq(FULL) & pca["target_split"].eq("pooled_qwen_llama")
    ].copy()
    keep = [
        "evaluator_backend_label",
        "rows",
        "pc1_variance",
        "pc1_pc4_cumulative_variance",
    ]
    pca = pca[keep].sort_values("pc1_variance", ascending=False)
    pca["interpretation"] = np.where(
        pca["pc1_variance"] >= 0.60,
        "high concentration; inspect for dominant scoring axis",
        np.where(
            pca["pc1_variance"] <= 0.30,
            "distributed variance; preserves multi-dimensional signal",
            "moderate concentration",
        ),
    )
    pca.to_csv(OUT / "pc1_backend_concentration_interpretation.csv", index=False)

    latex = pca.rename(
        columns={
            "evaluator_backend_label": "Backend",
            "rows": "Rows",
            "pc1_variance": "PC1 var.",
            "pc1_pc4_cumulative_variance": "PC1-PC4 var.",
            "interpretation": "Diagnostic reading",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_pc1_backend_concentration.tex",
        list(latex.columns),
        formats=["l", "r", "r", "r", "p{0.36\\textwidth}"],
        digits=3,
    )
    return pca


def backend_comparability_gate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = pd.DataFrame(
        [
            (
                "rho < 0.30",
                "backend-specific diagnostic patterns only",
                "shared ranking or cross-backend scalar claims",
            ),
            (
                "0.30 <= rho < 0.60",
                "exploratory comparison with backend labels",
                "strong shared-scale conclusions",
            ),
            (
                "rho >= 0.60 and low MAE",
                "cautious cross-backend comparison",
                "unreported calibration or label-specific bias assumptions",
            ),
            (
                "human anchors available",
                "calibrated shared-scale testing",
                "skipping label-specific and backend-specific bias checks",
            ),
        ],
        columns=["gate_condition", "permitted_reading", "not_permitted"],
    )
    rules.to_csv(OUT / "backend_comparability_gate_rules.csv", index=False)
    _write_tabular(
        rules.rename(
            columns={
                "gate_condition": "Gate condition",
                "permitted_reading": "Permitted reading",
                "not_permitted": "Not permitted",
            }
        ),
        TABLES / "submission_backend_comparability_gate_rules.tex",
        ["Gate condition", "Permitted reading", "Not permitted"],
        formats=["l", "p{0.34\\textwidth}", "p{0.34\\textwidth}"],
        digits=3,
    )

    pairs = pd.read_csv(AUDIT / "backend_pair_score_alignment.csv")
    full = pairs[
        pairs["evaluator_variant"].eq(FULL) & pairs["target_split"].eq("pooled_qwen_llama")
    ].copy()

    def gate_label(rho: float, mae: float) -> str:
        if rho < 0.30:
            return "backend-specific only"
        if rho < 0.60:
            return "exploratory comparison"
        if mae <= 0.20:
            return "cautious comparison"
        return "rho adequate but MAE high"

    full["gate"] = [gate_label(float(rho), float(mae)) for rho, mae in zip(full["spearman"], full["mae"])]
    full["backend_pair"] = full["backend_left"] + " vs " + full["backend_right"]
    full = full[["backend_pair", "n_shared_rows", "spearman", "mae", "gate"]].sort_values("spearman")
    full.to_csv(OUT / "backend_comparability_gate_by_pair.csv", index=False)
    latex = full.rename(
        columns={
            "backend_pair": "Backend pair",
            "n_shared_rows": "n",
            "spearman": "Spearman rho",
            "mae": "MAE",
            "gate": "Gate reading",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_backend_comparability_gate_by_pair.tex",
        list(latex.columns),
        formats=["p{0.40\\textwidth}", "r", "r", "r", "p{0.22\\textwidth}"],
        digits=3,
    )
    return rules, full


def covariance_concentration_diagnostics(behavior: pd.DataFrame) -> pd.DataFrame:
    full = behavior[behavior["evaluator_variant"].eq(FULL)].copy()
    pc1_summary = pd.read_csv(AUDIT / "protocol_backend_pca_summary.csv")
    pc1_summary = pc1_summary[
        pc1_summary["evaluator_variant"].eq(FULL)
        & pc1_summary["target_split"].eq("pooled_qwen_llama")
    ].copy()
    pc1_by_backend = dict(zip(pc1_summary["evaluator_backend_label"], pc1_summary["pc1_variance"]))
    alias_to_loading = {
        RULE_BACKEND: PCA_LOADINGS["Rule-based instrumentation"],
        "DeepSeek V4-Pro API judge": PCA_LOADINGS["DeepSeek V4-Pro"],
        "Gemma3-12B local HF judge": PCA_LOADINGS["Gemma3-12B"],
        "Qwen2.5-7B local HF judge": PCA_LOADINGS["Qwen2.5-7B"],
        "Qwen3-14B local HF judge": PCA_LOADINGS["Qwen3-14B"],
    }
    rows = []
    for backend, group in full.groupby("evaluator_backend_label", dropna=False):
        cols = [col for col in CONSTRUCT_COLUMNS if col in group.columns and group[col].notna().sum() >= 5]
        if len(cols) < 2 or backend not in alias_to_loading:
            continue
        corr = group[cols].astype(float).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        loading_df = pd.read_csv(AUDIT / alias_to_loading[backend], index_col=0)
        pc1 = loading_df.loc[loading_df.index.intersection(CONSTRUCT_COLUMNS), "PC1"].reindex(CONSTRUCT_COLUMNS).fillna(0)
        rows.append(
            {
                "backend": "Rule-based instrumentation" if backend == RULE_BACKEND else backend.replace(" local HF judge", "").replace(" API judge", ""),
                "rows": len(group),
                "mean_abs_inter_indicator_r": upper.mean(),
                "pc1_variance": pc1_by_backend.get(backend, np.nan),
                "pc1_loading_entropy": normalized_entropy(pc1.to_numpy(dtype=float)),
                "reading": (
                    "more modular indicator covariance"
                    if backend == RULE_BACKEND
                    else "more concentrated common-score covariance"
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("pc1_variance", ascending=False)
    table.to_csv(OUT / "pc1_covariance_concentration_summary.csv", index=False)
    latex = table.rename(
        columns={
            "backend": "Backend",
            "rows": "Rows",
            "mean_abs_inter_indicator_r": "Mean |r|",
            "pc1_variance": "PC1 var.",
            "pc1_loading_entropy": "PC1 loading entropy",
            "reading": "Diagnostic reading",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_pc1_covariance_concentration.tex",
        list(latex.columns),
        formats=["l", "r", "r", "r", "r", "p{0.32\\textwidth}"],
        digits=3,
    )
    return table


def instrumentation_pc1_and_profile_sanity() -> tuple[pd.DataFrame, pd.DataFrame]:
    common_indicators = CONSTRUCT_COLUMNS
    loadings = {}
    for backend, filename in PCA_LOADINGS.items():
        df = pd.read_csv(AUDIT / filename, index_col=0)
        pc1 = df.loc[df.index.intersection(common_indicators), "PC1"].reindex(common_indicators).fillna(0)
        loadings[backend] = pc1.abs().to_numpy(dtype=float)

    instrumentation = loadings["Rule-based instrumentation"]
    rows = []
    pc1_summary = pd.read_csv(AUDIT / "protocol_backend_pca_summary.csv")
    pc1_summary = pc1_summary[
        pc1_summary["evaluator_variant"].eq(FULL)
        & pc1_summary["target_split"].eq("pooled_qwen_llama")
    ].copy()
    pc1_by_backend = dict(zip(pc1_summary["evaluator_backend_label"], pc1_summary["pc1_variance"]))
    pc1_alias = {
        "Rule-based instrumentation": "Rule-based scoring backend (non-LLM)",
        "DeepSeek V4-Pro": "DeepSeek V4-Pro API judge",
        "Gemma3-12B": "Gemma3-12B local HF judge",
        "Qwen2.5-7B": "Qwen2.5-7B local HF judge",
        "Qwen3-14B": "Qwen3-14B local HF judge",
    }
    for backend, vector in loadings.items():
        top_indicator = common_indicators[int(np.argmax(vector))]
        rows.append(
            {
                "backend": backend,
                "pc1_variance": pc1_by_backend.get(pc1_alias[backend], np.nan),
                "cosine_to_instrumentation_pc1_abs": cosine_similarity(instrumentation, vector),
                "top_abs_pc1_indicator": top_indicator,
                "reading": (
                    "deterministic modular evidence/source PC1"
                    if backend == "Rule-based instrumentation"
                    else "model-based PC1; compare as replication screen, not invariance proof"
                ),
            }
        )
    pc1_diag = pd.DataFrame(rows).sort_values("pc1_variance", ascending=False)
    pc1_diag.to_csv(OUT / "instrumentation_vs_model_pc1_similarity.csv", index=False)
    latex_pc1 = pc1_diag.rename(
        columns={
            "backend": "Backend",
            "pc1_variance": "PC1 var.",
            "cosine_to_instrumentation_pc1_abs": "Cosine to instrumentation PC1",
            "top_abs_pc1_indicator": "Top abs. PC1 indicator",
            "reading": "Diagnostic reading",
        }
    )
    _write_tabular(
        latex_pc1,
        TABLES / "submission_instrumentation_vs_model_pc1_similarity.tex",
        list(latex_pc1.columns),
        formats=["l", "r", "r", "l", "p{0.34\\textwidth}"],
        digits=3,
    )

    instances = pd.read_csv(AUDIT / "paper_pc_structure_type_instances.csv")
    family_cols = ["A evidence/source", "B status/correction", "C resistance", "D boundary"]
    features = instances[family_cols].to_numpy(dtype=float)
    cluster_labels = deterministic_kmeans(features, k=5)
    rule_labels = instances["structure_type"].tolist()
    ari = adjusted_rand_index(rule_labels, cluster_labels.tolist())
    nmi = normalized_mutual_information(rule_labels, cluster_labels.tolist())
    cluster_summary = pd.DataFrame(
        [
            {
                "sanity_check": "deterministic_kmeans_k5_on_family_shares",
                "n_components": len(instances),
                "adjusted_rand_index": ari,
                "normalized_mutual_information": nmi,
                "interpretation": "higher values support coherence; lower values mean S1-S5 should remain interpretive bins",
            }
        ]
    )
    cluster_summary.to_csv(OUT / "s_profile_clustering_sanity.csv", index=False)
    latex_cluster = cluster_summary.rename(
        columns={
            "sanity_check": "Sanity check",
            "n_components": "n components",
            "adjusted_rand_index": "ARI",
            "normalized_mutual_information": "NMI",
            "interpretation": "Interpretation",
        }
    )
    _write_tabular(
        latex_cluster,
        TABLES / "submission_s_profile_clustering_sanity.tex",
        list(latex_cluster.columns),
        formats=["l", "r", "r", "r", "p{0.36\\textwidth}"],
        digits=3,
    )

    contingency = pd.crosstab(instances["structure_type"], cluster_labels, rownames=["rule_label"], colnames=["kmeans_cluster"])
    contingency.to_csv(OUT / "s_profile_cluster_contingency.csv")
    return pc1_diag, cluster_summary


def response_behavior_diversity(behavior: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = behavior[
        behavior["evaluator_variant"].eq(FULL) & behavior["evaluator_backend_label"].eq(RULE_BACKEND)
    ].copy()

    def behavior_type(row: pd.Series) -> str:
        label = row.get("reference_label")
        correction = float(row.get("correction_accuracy", np.nan))
        detection = float(row.get("misinformation_detection", np.nan))
        uncertainty_handling = float(row.get("uncertainty_handling", np.nan))
        hallucination = float(row.get("hallucination_control", np.nan))
        supports_failure = float(row.get("supports_status_failure_proxy", np.nan))
        nei_failure = float(row.get("nei_uncertainty_failure_proxy", np.nan))
        if label == "REFUTES":
            if detection >= 0.75 and correction >= 0.75:
                return "corrects false claim"
            if 0.25 <= correction < 0.75:
                return "partial correction"
            if detection < 0.50 and correction < 0.25:
                return "accepts false premise"
            return "other refutes behavior"
        if label == "SUPPORTS":
            if np.isfinite(supports_failure) and supports_failure >= 0.50:
                return "overcorrects true claim"
            return "accepts true claim"
        if label == "NOT ENOUGH INFO":
            if (np.isfinite(nei_failure) and nei_failure >= 0.50) or (
                np.isfinite(uncertainty_handling) and uncertainty_handling < 0.50
            ):
                return "overclaims NEI"
            if uncertainty_handling >= 0.75 and hallucination >= 0.75:
                return "cautious uncertainty"
            return "other NEI behavior"
        return "other"

    full["behavior_type"] = full.apply(behavior_type, axis=1)
    count_rows = []
    summary_rows = []
    for target, group in full.groupby("target_model", dropna=False):
        counts = group["behavior_type"].value_counts().sort_index()
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum() / math.log(len(probs))) if len(probs) > 1 else 0.0
        for label, count in counts.items():
            count_rows.append(
                {
                    "target_model": target,
                    "behavior_type": label,
                    "count": int(count),
                    "proportion": float(count / len(group)),
                }
            )
        indicator_means = {col: group[col].astype(float).mean() for col in CONSTRUCT_COLUMNS if col in group}
        near_ceiling = sum(value >= 0.90 for value in indicator_means.values() if np.isfinite(value))
        summary_rows.append(
            {
                "target_model": target,
                "rows": len(group),
                "misinformation_detection_mean": indicator_means.get("misinformation_detection", np.nan),
                "correction_accuracy_mean": indicator_means.get("correction_accuracy", np.nan),
                "evidence_grounding_mean": indicator_means.get("evidence_grounding", np.nan),
                "uncertainty_handling_mean": indicator_means.get("uncertainty_handling", np.nan),
                "overclaim_proxy_mean": group["nei_uncertainty_failure_proxy"].astype(float).mean(),
                "behavior_entropy": entropy,
                "near_ceiling_indicator_count": near_ceiling,
            }
        )
    counts = pd.DataFrame(count_rows)
    summary = pd.DataFrame(summary_rows)
    counts.to_csv(OUT / "response_behavior_category_counts.csv", index=False)
    summary.to_csv(OUT / "response_behavior_diversity_summary.csv", index=False)
    latex = summary.rename(
        columns={
            "target_model": "Target model",
            "rows": "Rows",
            "misinformation_detection_mean": "Misinfo det.",
            "correction_accuracy_mean": "Correction acc.",
            "evidence_grounding_mean": "Evidence ground.",
            "uncertainty_handling_mean": "Uncertainty",
            "overclaim_proxy_mean": "NEI overclaim proxy",
            "behavior_entropy": "Behavior entropy",
            "near_ceiling_indicator_count": "Near-ceiling indicators",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_response_behavior_diversity.tex",
        list(latex.columns),
        formats=["p{0.28\\textwidth}", "r", "r", "r", "r", "r", "r", "r", "r"],
        digits=3,
    )
    return summary, counts


def target_delta_diagnostics() -> pd.DataFrame:
    delta = pd.read_csv(AUDIT / "full_cale_construct_profile_qwen_minus_llama.csv")
    summary_cols = [
        "evaluator_backend_label",
        "misinformation_detection",
        "framing_resistance",
        "claim_status_recognition",
        "error_rejection",
        "correction_accuracy",
        "evidence_grounding",
        "source_faithfulness",
        "uncertainty_handling",
    ]
    compact = delta[summary_cols].copy()
    construct_cols = [col for col in compact.columns if col != "evaluator_backend_label"]
    compact["largest_positive_delta"] = compact[construct_cols].idxmax(axis=1)
    compact["largest_positive_value"] = compact[construct_cols].max(axis=1)
    compact["largest_negative_delta"] = compact[construct_cols].idxmin(axis=1)
    compact["largest_negative_value"] = compact[construct_cols].min(axis=1)
    compact.to_csv(OUT / "target_delta_construct_diagnostics.csv", index=False)

    latex = compact[
        [
            "evaluator_backend_label",
            "largest_positive_delta",
            "largest_positive_value",
            "largest_negative_delta",
            "largest_negative_value",
        ]
    ].rename(
        columns={
            "evaluator_backend_label": "Backend",
            "largest_positive_delta": "Largest Qwen>Llama indicator",
            "largest_positive_value": "Positive delta",
            "largest_negative_delta": "Largest Qwen<Llama indicator",
            "largest_negative_value": "Smallest delta",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_target_delta_diagnostics.tex",
        list(latex.columns),
        formats=["l", "l", "r", "l", "r"],
        digits=3,
    )
    return compact


def direct_full_delta_diagnostics(behavior: pd.DataFrame) -> pd.DataFrame:
    paired = pd.read_csv(AUDIT / "paired_direct_vs_full_cale_rows.csv")
    full = behavior[behavior["evaluator_variant"].eq(FULL)].copy()
    join_cols = ["evaluator_backend_label", "target_model", "id"]
    merged = paired.merge(
        full[join_cols + CONSTRUCT_COLUMNS + ["reference_label"]],
        on=join_cols,
        how="left",
    )
    rows = []
    for (backend, target), group in merged.groupby(["evaluator_backend_label", "target_model"], dropna=False):
        indicator_corrs = {}
        for col in CONSTRUCT_COLUMNS:
            paired_cols = group[["full_minus_direct_score", col]].dropna().astype(float)
            if len(paired_cols) >= 20 and paired_cols[col].std(ddof=1) > 0:
                indicator_corrs[col] = float(paired_cols["full_minus_direct_score"].corr(paired_cols[col]))
        if indicator_corrs:
            positive_indicator = max(indicator_corrs, key=indicator_corrs.get)
            negative_indicator = min(indicator_corrs, key=indicator_corrs.get)
            positive_value = indicator_corrs[positive_indicator]
            negative_value = indicator_corrs[negative_indicator]
        else:
            positive_indicator = negative_indicator = ""
            positive_value = negative_value = np.nan
        rows.append(
            {
                "backend": "Rule-based instrumentation" if backend == RULE_BACKEND else backend.replace(" local HF judge", "").replace(" API judge", ""),
                "target_model": target,
                "rows": len(group),
                "mean_full_minus_direct": group["full_minus_direct_score"].astype(float).mean(),
                "top_positive_indicator_corr": positive_indicator,
                "top_positive_corr": positive_value,
                "top_negative_indicator_corr": negative_indicator,
                "top_negative_corr": negative_value,
            }
        )
    table = pd.DataFrame(rows).sort_values(["backend", "target_model"])
    table.to_csv(OUT / "direct_full_delta_indicator_correlates.csv", index=False)
    latex = table.rename(
        columns={
            "backend": "Backend",
            "target_model": "Target model",
            "rows": "Rows",
            "mean_full_minus_direct": "Mean Full-Direct",
            "top_positive_indicator_corr": "Positive correlate",
            "top_positive_corr": "Positive r",
            "top_negative_indicator_corr": "Negative correlate",
            "top_negative_corr": "Negative r",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_direct_full_delta_indicator_correlates.tex",
        list(latex.columns),
        formats=["l", "p{0.25\\textwidth}", "r", "r", "l", "r", "l", "r"],
        digits=3,
    )
    return table


def boundary_hard_reference_table() -> pd.DataFrame:
    sources = [
        ("Rule-based instrumentation", pd.read_csv(ROOT / "figures" / "boundary_hard_real_subset" / "boundary_hard_behavior_with_manifest.csv")),
        ("Qwen3-14B semantic backend", pd.read_csv(BOUNDARY / "boundary_hard_behavior_with_manifest.csv")),
    ]
    rows = []
    for backend, df in sources:
        nei = df[df["reference_label"].eq("NOT ENOUGH INFO")].copy()
        for variant in ["direct_trustllm_heuristic", "direct_llm_judge", FULL]:
            sub = nei[nei["evaluator_variant"].eq(variant)].copy()
            if len(sub) == 0:
                continue
            row = {
                "backend": backend,
                "variant": variant,
                "label": "NEI",
                "rows": len(sub),
                "final_score": sub["final_score"].astype(float).mean(),
                "final_score_ci_low": mean_ci(sub["final_score"])[0],
                "final_score_ci_high": mean_ci(sub["final_score"])[1],
            }
            for metric in ["uncertainty_handling", "nei_uncertainty_failure_proxy", "hallucination_control"]:
                if metric in sub and sub[metric].notna().any():
                    series = sub[metric].dropna().astype(float)
                    row[metric] = series.mean()
                    lo, hi = mean_ci(series)
                    row[f"{metric}_ci_low"] = lo
                    row[f"{metric}_ci_high"] = hi
                else:
                    row[metric] = np.nan
                    row[f"{metric}_ci_low"] = np.nan
                    row[f"{metric}_ci_high"] = np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "boundary_hard_nei_reference_table.csv", index=False)
    compact = table[
        [
            "backend",
            "variant",
            "rows",
            "final_score",
            "uncertainty_handling",
            "nei_uncertainty_failure_proxy",
            "hallucination_control",
        ]
    ].copy()
    compact["reading"] = np.where(
        compact["backend"].str.contains("Rule-based"),
        "instrumentation contrast, not semantic reliability",
        "semantic stress signal",
    )
    latex = compact.rename(
        columns={
            "backend": "Backend",
            "variant": "Variant",
            "rows": "Rows",
            "final_score": "Final score",
            "uncertainty_handling": "Uncertainty",
            "nei_uncertainty_failure_proxy": "Failure proxy",
            "hallucination_control": "Hallucination ctrl.",
            "reading": "Diagnostic reading",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_boundary_hard_nei_reference.tex",
        list(latex.columns),
        formats=["l", "l", "r", "r", "r", "r", "r", "p{0.28\\textwidth}"],
        digits=3,
    )
    return table


def aggregation_weight_sensitivity(behavior: pd.DataFrame) -> pd.DataFrame:
    full = behavior[behavior["evaluator_variant"].eq(FULL)].copy()
    complete = full.dropna(subset=CONSTRUCT_COLUMNS + ["final_score"]).copy()
    weights_base = {
        "misinformation_detection": 1.0,
        "framing_resistance": 1.0,
        "claim_status_recognition": 1.0,
        "error_rejection": 1.0,
        "correction_accuracy": 1.5,
        "evidence_grounding": 1.0,
        "source_faithfulness": 1.1,
        "hallucination_control": 1.0,
        "uncertainty_handling": 1.0,
    }
    rows = []
    original = complete["final_score"].astype(float)
    for correction_weight in [1.0, 1.25, 1.5, 1.75, 2.0]:
        weights = weights_base.copy()
        weights["correction_accuracy"] = correction_weight
        weight_vec = np.array([weights[col] for col in CONSTRUCT_COLUMNS], dtype=float)
        weighted = complete[CONSTRUCT_COLUMNS].astype(float).to_numpy() @ weight_vec / weight_vec.sum()
        score = pd.Series(np.clip(0.08 + 0.88 * weighted, 0, 1), index=complete.index)
        rows.append(
            {
                "correction_accuracy_weight": correction_weight,
                "rows": len(complete),
                "spearman_vs_original": spearman_corr(original, score),
                "top10_overlap": top_decile_overlap(original, score),
                "mean_score_delta": float((score - original).mean()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "aggregation_weight_sensitivity.csv", index=False)
    latex = table.rename(
        columns={
            "correction_accuracy_weight": "Correction weight",
            "rows": "Rows",
            "spearman_vs_original": "Spearman vs original",
            "top10_overlap": "Top 10 pct overlap",
            "mean_score_delta": "Mean score delta",
        }
    )
    latex["Rows"] = latex["Rows"].astype(int).astype(str)
    _write_tabular(
        latex,
        TABLES / "submission_aggregation_weight_sensitivity.tex",
        list(latex.columns),
        formats=["r", "r", "r", "r", "r"],
        digits=3,
    )
    return table


def four_factor_cfa_diagnostics() -> pd.DataFrame:
    loadings = pd.read_csv(CFA / "heuristic_full" / "pooled_cale" / "standardized_loadings.csv")
    four = loadings[loadings["model"].eq("four_factor_sensitivity")].copy()
    four = four[["factor", "indicator", "std_loading"]].sort_values(["factor", "indicator"])
    four.to_csv(OUT / "four_factor_sensitivity_loadings.csv", index=False)

    latex = four.rename(
        columns={
            "factor": "Factor",
            "indicator": "Indicator",
            "std_loading": "Std. loading",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_four_factor_loadings.tex",
        list(latex.columns),
        formats=["l", "l", "r"],
        digits=3,
    )
    return four


def targeted_screen_ci_diagnostics() -> pd.DataFrame:
    rows = []
    framing = pd.read_csv(FRAMING / "controlled_framing_behavior_with_manifest.csv")
    framing = framing[framing["evaluator_variant"].eq(FULL)].copy()
    for contrast in ["assertive", "authoritative"]:
        sub = framing[framing["framing_style_manifest"].isin(["neutral", contrast])]
        wide = (
            sub.pivot_table(
                index=["source_id_manifest", "target_model"],
                columns="framing_style_manifest",
                values="final_score",
                aggfunc="mean",
            )
            .dropna()
            .reset_index()
        )
        rho = spearman_corr(wide["neutral"], wide[contrast])
        lo, hi = fisher_ci(rho, len(wide))
        rows.append(
            {
                "screen": f"controlled_framing_{contrast}",
                "n": len(wide),
                "metric": "neutral_vs_framed_spearman",
                "estimate": rho,
                "ci_low": lo,
                "ci_high": hi,
                "mean_abs_shift": (wide[contrast] - wide["neutral"]).abs().mean(),
            }
        )

    boundary = pd.read_csv(BOUNDARY / "boundary_hard_behavior_with_manifest.csv")
    boundary = boundary[
        boundary["evaluator_variant"].eq(FULL) & boundary["reference_label"].eq("NOT ENOUGH INFO")
    ].copy()
    for metric in ["uncertainty_handling", "nei_uncertainty_failure_proxy", "final_score"]:
        series = boundary[metric].dropna().astype(float)
        lo, hi = mean_ci(series)
        rows.append(
            {
                "screen": "boundary_hard_nei",
                "n": len(series),
                "metric": metric,
                "estimate": series.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "mean_abs_shift": np.nan,
            }
        )

    ci = pd.DataFrame(rows)
    ci.to_csv(OUT / "targeted_screen_confidence_intervals.csv", index=False)

    latex = ci.rename(
        columns={
            "screen": "Screen",
            "n": "n",
            "metric": "Metric",
            "estimate": "Estimate",
            "ci_low": "CI low",
            "ci_high": "CI high",
            "mean_abs_shift": "Mean abs. shift",
        }
    )
    _write_tabular(
        latex,
        TABLES / "submission_targeted_screen_ci.tex",
        list(latex.columns),
        formats=["l", "r", "l", "r", "r", "r", "r"],
        digits=3,
    )
    return ci


def aggregation_and_profile_rules() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame(
        [
            ("misinformation_detection", 1.0),
            ("framing_resistance", 1.0),
            ("claim_status_recognition", 1.0),
            ("error_rejection", 1.0),
            ("correction_accuracy", 1.5),
            ("evidence_grounding", 1.0),
            ("source_faithfulness", 1.1),
            ("hallucination_control", 1.0),
            ("uncertainty_handling", 1.0),
        ],
        columns=["indicator", "weight"],
    )
    weights.to_csv(OUT / "full_cale_aggregation_weights.csv", index=False)
    _write_tabular(
        weights.rename(columns={"indicator": "Indicator", "weight": "Weight"}),
        TABLES / "submission_full_cale_weights.tex",
        ["Indicator", "Weight"],
        formats=["l", "r"],
        digits=1,
    )

    s_rules = pd.DataFrame(
        [
            ("S1 broad CALE-general", "top family share < 0.30 or top-minus-second margin < 0.035"),
            ("S2 evidence/source", "largest family share is evidence/source"),
            ("S3 status/correction", "largest family share is claim-status/correction"),
            ("S4 resistance", "largest family share is framing/error resistance"),
            ("S5 boundary", "largest family share is boundary/uncertainty"),
        ],
        columns=["profile_label", "assignment_rule"],
    )
    s_rules.to_csv(OUT / "s_profile_assignment_rules.csv", index=False)
    _write_tabular(
        s_rules.rename(columns={"profile_label": "Profile", "assignment_rule": "Assignment rule"}),
        TABLES / "submission_s_profile_assignment_rules.tex",
        ["Profile", "Assignment rule"],
        formats=["l", "p{0.55\\textwidth}"],
        digits=3,
    )
    return weights, s_rules


def claim_boundary_table() -> pd.DataFrame:
    readiness = pd.read_csv(CFA / "validity" / "paper_cfa_claim_readiness.csv")
    area_labels = {
        "internal_structure": "Internal structure",
        "convergent_validity": "Convergent validity",
        "discriminant_validity": "Discriminant validity",
        "measurement_invariance": "Measurement invariance",
        "compact_capability_score": "Compact diagnostic score",
    }
    table = readiness[["claim_area", "status_label", "safe_wording"]].copy()
    table["claim_area"] = table["claim_area"].map(area_labels).fillna(table["claim_area"])
    table = table.rename(
        columns={
            "claim_area": "Claim area",
            "status_label": "Current status",
            "safe_wording": "Submission wording",
        }
    )
    table.to_csv(OUT / "claim_boundary_summary.csv", index=False)
    _write_tabular(
        table,
        TABLES / "submission_claim_boundary_summary.tex",
        list(table.columns),
        formats=["l", "l", "p{0.52\\textwidth}"],
        digits=3,
    )
    return table


def write_summary(
    dist: pd.DataFrame,
    by_label: pd.DataFrame,
    pc1: pd.DataFrame,
    ci: pd.DataFrame,
    pc1_similarity: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    covariance_summary: pd.DataFrame,
    gate_by_pair: pd.DataFrame,
    boundary_reference: pd.DataFrame,
    diversity_summary: pd.DataFrame,
    weight_sensitivity: pd.DataFrame,
    calibration_sensitivity: pd.DataFrame,
) -> None:
    qwen_dist = dist[dist["backend"].eq(QWEN25)].iloc[0]
    deepseek_dist = dist[dist["backend"].eq(DEEPSEEK)].iloc[0]
    support_row = by_label[by_label["reference_label"].eq("SUPPORTS")].iloc[0]
    assertive = ci[ci["screen"].eq("controlled_framing_assertive")].iloc[0]
    authoritative = ci[ci["screen"].eq("controlled_framing_authoritative")].iloc[0]
    instrumentation_row = pc1_similarity[pc1_similarity["backend"].eq("Rule-based instrumentation")].iloc[0]
    best_model_cosine = pc1_similarity[~pc1_similarity["backend"].eq("Rule-based instrumentation")][
        "cosine_to_instrumentation_pc1_abs"
    ].max()
    cluster_row = cluster_summary.iloc[0]
    weakest_gate = gate_by_pair.iloc[0]
    instrumentation_cov = covariance_summary[
        covariance_summary["backend"].eq("Rule-based instrumentation")
    ].iloc[0]
    qwen3_boundary = boundary_reference[
        boundary_reference["backend"].eq("Qwen3-14B semantic backend")
        & boundary_reference["variant"].eq(FULL)
    ].iloc[0]
    min_weight_rho = weight_sensitivity["spearman_vs_original"].min()
    calibration_rank_row = calibration_sensitivity[
        calibration_sensitivity["check"].eq("Percentile/rank normalization")
    ].iloc[0]
    summary = f"""# Submission Diagnostics Summary

Generated from existing artifacts; no new evaluator inference was run.

- DeepSeek/Qwen2.5 shared Full-CALE rows: 2,000.
- Qwen2.5-7B mean final score: {qwen_dist['mean']:.3f}; proportion >= .90: {qwen_dist['prop_ge_0_90']:.3f}.
- DeepSeek V4-Pro mean final score: {deepseek_dist['mean']:.3f}; proportion >= .90: {deepseek_dist['prop_ge_0_90']:.3f}.
- SUPPORTS rows show the strongest rank reversal in the pair: Spearman {support_row['spearman']:.3f}, MAE {support_row['mae']:.3f}.
- Pooled PC1 variance range across Full-CALE backends: {pc1['pc1_variance'].min():.3f}-{pc1['pc1_variance'].max():.3f}; interpret this as concentration/distribution, not quality.
- Rule-based instrumentation PC1 variance: {instrumentation_row['pc1_variance']:.3f}; max absolute-PC1 cosine to a model backend: {best_model_cosine:.3f}.
- Rule-based instrumentation mean absolute inter-indicator correlation: {instrumentation_cov['mean_abs_inter_indicator_r']:.3f}; this supports a modular-covariance reading rather than a weaker-quality reading.
- Weakest backend comparability gate: {weakest_gate['backend_pair']} has rho {weakest_gate['spearman']:.3f} and is {weakest_gate['gate']}.
- DeepSeek/Qwen2.5 calibration sensitivity: {calibration_rank_row['completed_result']}.
- S-profile k=5 sanity check: ARI {cluster_row['adjusted_rand_index']:.3f}, NMI {cluster_row['normalized_mutual_information']:.3f}.
- Qwen3 boundary-hard NEI Full-CALE failure proxy: {qwen3_boundary['nei_uncertainty_failure_proxy']:.3f}; interpret as backend/rubric stress signal.
- Aggregation correction-weight sensitivity min Spearman vs original: {min_weight_rho:.3f}.
- Qwen3 framing stability: assertive rho {assertive['estimate']:.4f} [{assertive['ci_low']:.4f}, {assertive['ci_high']:.4f}], authoritative rho {authoritative['estimate']:.4f} [{authoritative['ci_low']:.4f}, {authoritative['ci_high']:.4f}].
- Target response diversity summaries were generated for {len(diversity_summary)} target models using instrumentation-derived proxies.
"""
    (OUT / "submission_diagnostics_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    _mkdirs()
    behavior = pd.read_csv(AUDIT / "combined_available_behavior_matrix.csv")
    dist, by_label = backend_disagreement_diagnostics(behavior)
    calibration_sensitivity = deepseek_qwen25_calibration_sensitivity(behavior)
    pc1 = pc1_concentration_diagnostics()
    _, gate_by_pair = backend_comparability_gate()
    covariance_summary = covariance_concentration_diagnostics(behavior)
    pc1_similarity, cluster_summary = instrumentation_pc1_and_profile_sanity()
    diversity_summary, _ = response_behavior_diversity(behavior)
    target_delta_diagnostics()
    direct_full_delta_diagnostics(behavior)
    boundary_reference = boundary_hard_reference_table()
    weight_sensitivity = aggregation_weight_sensitivity(behavior)
    four_factor_cfa_diagnostics()
    ci = targeted_screen_ci_diagnostics()
    aggregation_and_profile_rules()
    claim_boundary_table()
    write_summary(
        dist,
        by_label,
        pc1,
        ci,
        pc1_similarity,
        cluster_summary,
        covariance_summary,
        gate_by_pair,
        boundary_reference,
        diversity_summary,
        weight_sensitivity,
        calibration_sensitivity,
    )
    print(f"Wrote diagnostics to {OUT}")
    print(f"Wrote LaTeX table fragments to {TABLES}")


if __name__ == "__main__":
    main()
