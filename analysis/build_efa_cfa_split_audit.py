#!/usr/bin/env python3
"""Run a split-sample EFA -> CFA readiness audit for CALE indicators.

This script keeps PCA as a descriptive concentration diagnostic and moves the
factor-model logic into an explicit discovery/confirmation workflow:

1. Discovery split: EFA-style principal-axis factoring plus parallel analysis.
2. Held-out split: CFA-style SEM comparison of the legacy theory model and the
   EFA-informed refinement.

The analysis uses the rule-based instrumentation backend only. The outputs
therefore diagnose the implemented CALE indicator schema, not evaluator-general
LLM latent structure or human-validated response quality.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from semopy import Model, calc_stats


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "figures" / "paper_tables"
OUT = ROOT / "figures" / "efa_cfa_split_audit"
MATRIX = ROOT / "figures" / "global_evaluator_audit" / "combined_available_behavior_matrix.csv"

RULE_BACKEND = "Rule-based scoring backend (non-LLM)"
CALE_VARIANTS = ["generic_cale", "attack_aware_cale", "full_attack_aware_cale"]

ITEMS = [
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

MODEL_SYNTAX = {
    "one_factor_baseline": """
cale_general =~ misinformation_detection + framing_resistance + claim_status_recognition + error_rejection + correction_accuracy + evidence_grounding + source_faithfulness + hallucination_control + uncertainty_handling
""",
    "legacy_three_factor": """
factual_handling =~ claim_status_recognition + correction_accuracy + evidence_grounding + source_faithfulness
adversarial_resistance =~ misinformation_detection + framing_resistance + error_rejection
boundary_control =~ hallucination_control + uncertainty_handling
factual_handling ~~ adversarial_resistance
factual_handling ~~ boundary_control
adversarial_resistance ~~ boundary_control
""",
    "efa_informed_four_factor": """
claim_correction =~ claim_status_recognition + correction_accuracy
evidence_faithfulness =~ evidence_grounding + source_faithfulness
adversarial_resistance =~ misinformation_detection + framing_resistance + error_rejection
boundary_control =~ hallucination_control + uncertainty_handling
claim_correction ~~ evidence_faithfulness
claim_correction ~~ adversarial_resistance
claim_correction ~~ boundary_control
evidence_faithfulness ~~ adversarial_resistance
evidence_faithfulness ~~ boundary_control
adversarial_resistance ~~ boundary_control
""",
}

MODEL_LABELS = {
    "one_factor_baseline": "One-factor baseline",
    "legacy_three_factor": "Legacy theory-guided three-factor",
    "efa_informed_four_factor": "EFA-informed four-factor refinement",
}

FACTOR_LABELS = {
    "evidence_source": "Evidence/source",
    "framing_error": "Framing/error",
    "claim_correction": "Claim/correction",
    "residual_detection_boundary": "Residual detection/boundary",
}

INDICATOR_LABELS = {
    "misinformation_detection": "Misinformation detection",
    "framing_resistance": "Framing resistance",
    "claim_status_recognition": "Claim-status recognition",
    "error_rejection": "Error rejection",
    "correction_accuracy": "Correction accuracy",
    "evidence_grounding": "Evidence grounding",
    "source_faithfulness": "Source faithfulness",
    "hallucination_control": "Hallucination control",
    "uncertainty_handling": "Uncertainty handling",
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
    if isinstance(value, (int, np.integer)):
        return str(int(value))
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


def stable_split_key(target_model: object, row_id: object) -> str:
    key = f"{target_model}::{row_id}"
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "efa_discovery" if value < 0.5 else "cfa_holdout"


def load_instrumentation_sample() -> pd.DataFrame:
    usecols = ["evaluator_backend_label", "evaluator_variant", "target_model", "id", *ITEMS]
    data = pd.read_csv(MATRIX, usecols=usecols)
    sample = data[
        data["evaluator_backend_label"].eq(RULE_BACKEND)
        & data["evaluator_variant"].isin(CALE_VARIANTS)
    ].copy()
    sample["response_key"] = sample["target_model"].astype(str) + "::" + sample["id"].astype(str)
    sample["split"] = [
        stable_split_key(target, row_id)
        for target, row_id in zip(sample["target_model"], sample["id"])
    ]
    return sample


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    complete = data[ITEMS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return complete.corr()


def parallel_analysis(data: pd.DataFrame, n_iter: int = 200, seed: int = 20260609) -> pd.DataFrame:
    complete = data[ITEMS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    observed_corr = complete.corr().to_numpy()
    observed = np.linalg.eigvalsh(observed_corr)[::-1]
    rng = np.random.default_rng(seed)
    random_eigs = []
    for _ in range(n_iter):
        random_data = rng.normal(size=complete.shape)
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigs.append(np.linalg.eigvalsh(random_corr)[::-1])
    random_eigs = np.asarray(random_eigs)
    table = pd.DataFrame(
        {
            "component": [f"F{i}" for i in range(1, len(ITEMS) + 1)],
            "observed_eigenvalue": observed,
            "parallel_p95": np.quantile(random_eigs, 0.95, axis=0),
            "retain_by_parallel": observed > np.quantile(random_eigs, 0.95, axis=0),
        }
    )
    return table


def principal_axis_loadings(corr: np.ndarray, n_factors: int) -> np.ndarray:
    try:
        inv_corr = np.linalg.inv(corr)
        communalities = 1.0 - 1.0 / np.diag(inv_corr)
    except np.linalg.LinAlgError:
        communalities = np.full(corr.shape[0], 0.70)
    communalities = np.clip(communalities, 0.02, 0.98)

    loadings = np.zeros((corr.shape[0], n_factors))
    for _ in range(300):
        reduced = corr.copy()
        np.fill_diagonal(reduced, communalities)
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))
        next_communalities = np.clip((loadings**2).sum(axis=1), 0.02, 0.98)
        if float(np.max(np.abs(next_communalities - communalities))) < 1e-6:
            break
        communalities = next_communalities
    return loadings


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    n_rows, n_cols = loadings.shape
    rotation = np.eye(n_cols)
    previous = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, singular_values, vh = np.linalg.svd(
            loadings.T
            @ (
                rotated**3
                - (gamma / n_rows) * rotated @ np.diag(np.diag(rotated.T @ rotated))
            )
        )
        rotation = u @ vh
        current = singular_values.sum()
        if previous and current / previous < 1 + tol:
            break
        previous = current
    return loadings @ rotation


def label_factor(column: pd.Series) -> str:
    abs_values = column.abs()
    top_indicator = str(abs_values.idxmax())
    if top_indicator in {"evidence_grounding", "source_faithfulness"}:
        return "evidence_source"
    if top_indicator in {"framing_resistance", "error_rejection"}:
        return "framing_error"
    if top_indicator in {"claim_status_recognition", "correction_accuracy"}:
        return "claim_correction"
    return "residual_detection_boundary"


def run_efa(discovery: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    corr = correlation_matrix(discovery).to_numpy()
    raw_loadings = principal_axis_loadings(corr, n_factors)
    rotated = varimax(raw_loadings)

    loading_df = pd.DataFrame(rotated, index=ITEMS, columns=[f"factor_{i}" for i in range(1, n_factors + 1)])
    labels = [label_factor(loading_df[col]) for col in loading_df.columns]

    # Orient signs and order factors by thesis-relevant label.
    for col in loading_df.columns:
        top = loading_df[col].abs().idxmax()
        if loading_df.loc[top, col] < 0:
            loading_df[col] *= -1

    order = ["evidence_source", "framing_error", "claim_correction", "residual_detection_boundary"]
    labeled_cols = []
    for label in order:
        if label in labels:
            col = loading_df.columns[labels.index(label)]
            labeled_cols.append((label, col))
    for label, col in zip(labels, loading_df.columns):
        if all(col != existing for _, existing in labeled_cols):
            labeled_cols.append((label, col))

    output = pd.DataFrame({"indicator": ITEMS, "indicator_label": [INDICATOR_LABELS[i] for i in ITEMS]})
    for label, col in labeled_cols:
        output[FACTOR_LABELS[label]] = loading_df[col].to_numpy()
    return output


def flatten_stats(stats_obj: pd.DataFrame, model_name: str, n_obs: int) -> dict[str, object]:
    row = stats_obj.iloc[0].to_dict()
    result: dict[str, object] = {"model": model_name, "model_label": MODEL_LABELS[model_name], "n_obs": n_obs}
    for key, value in row.items():
        clean_key = str(key).replace(" ", "_").replace("-", "_")
        try:
            result[clean_key] = float(value)
        except Exception:
            result[clean_key] = value
    return result


def fit_cfa_models(holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = holdout[ITEMS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fit_rows = []
    loading_rows = []
    correlation_rows = []

    for model_name, syntax in MODEL_SYNTAX.items():
        model = Model(syntax)
        model.fit(complete)
        fit_rows.append(flatten_stats(calc_stats(model), model_name, len(complete)))

        estimates = model.inspect(std_est=True).copy()
        std_col = next((c for c in ["Est. Std", "Est.Std", "Std. Estimate", "Std.Estimate"] if c in estimates.columns), None)
        observed = set(ITEMS)
        loadings = estimates[(estimates["op"].eq("~")) & estimates["lval"].isin(observed)].copy()
        if std_col:
            loadings = loadings.rename(columns={std_col: "std_loading"})
        for _, row in loadings.iterrows():
            loading_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "factor": row["rval"],
                    "indicator": row["lval"],
                    "indicator_label": INDICATOR_LABELS.get(row["lval"], row["lval"]),
                    "loading": row.get("Estimate", np.nan),
                    "std_loading": row.get("std_loading", np.nan),
                }
            )

        corrs = estimates[
            estimates["op"].eq("~~")
            & ~estimates["lval"].isin(observed)
            & ~estimates["rval"].isin(observed)
            & (estimates["lval"] != estimates["rval"])
        ].copy()
        if std_col:
            corrs = corrs.rename(columns={std_col: "std_factor_correlation"})
        for _, row in corrs.iterrows():
            correlation_rows.append(
                {
                    "model": model_name,
                    "factor_left": row["lval"],
                    "factor_right": row["rval"],
                    "factor_correlation": row.get("std_factor_correlation", np.nan),
                }
            )

    return pd.DataFrame(fit_rows), pd.DataFrame(loading_rows), pd.DataFrame(correlation_rows)


def build_outputs() -> None:
    _mkdirs()
    sample = load_instrumentation_sample()

    manifest = (
        sample.groupby("split")
        .agg(rows=("id", "size"), response_keys=("response_key", "nunique"))
        .reset_index()
    )
    manifest["backend_scope"] = RULE_BACKEND
    manifest["variants"] = ", ".join(CALE_VARIANTS)
    manifest.to_csv(OUT / "split_manifest.csv", index=False)

    discovery = sample[sample["split"].eq("efa_discovery")].copy()
    holdout = sample[sample["split"].eq("cfa_holdout")].copy()

    parallel = parallel_analysis(discovery)
    n_factors = int(parallel["retain_by_parallel"].sum())
    n_factors = max(1, min(n_factors, len(ITEMS)))
    parallel.to_csv(OUT / "efa_parallel_analysis.csv", index=False)

    efa_loadings = run_efa(discovery, n_factors)
    efa_loadings.to_csv(OUT / "efa_rotated_paf_loadings.csv", index=False)

    fit, loadings, corrs = fit_cfa_models(holdout)
    fit.to_csv(OUT / "heldout_cfa_fit_indices.csv", index=False)
    loadings.to_csv(OUT / "heldout_cfa_standardized_loadings.csv", index=False)
    corrs.to_csv(OUT / "heldout_cfa_factor_correlations.csv", index=False)

    parallel_tex = parallel.copy()
    parallel_tex["Retain"] = parallel_tex["retain_by_parallel"].map({True: "yes", False: "no"})
    parallel_tex = parallel_tex[["component", "observed_eigenvalue", "parallel_p95", "Retain"]]
    _write_tabular(
        parallel_tex.head(6),
        TABLES / "submission_efa_parallel_analysis.tex",
        ["Factor", "Observed eigenvalue", "Parallel 95%", "Retain"],
        ["l", "r", "r", "l"],
    )

    loadings_tex = efa_loadings.copy()
    for col in loadings_tex.columns:
        if col not in {"indicator", "indicator_label"}:
            loadings_tex[col] = loadings_tex[col].astype(float)
    loadings_tex = loadings_tex.drop(columns=["indicator"]).rename(columns={"indicator_label": "Indicator"})
    _write_tabular(
        loadings_tex,
        TABLES / "submission_efa_rotated_loadings.tex",
        loadings_tex.columns.tolist(),
        ["p{0.26\\textwidth}", *["r"] * (len(loadings_tex.columns) - 1)],
    )

    fit_tex = fit.copy()
    fit_tex = fit_tex[["model_label", "n_obs", "DoF", "CFI", "TLI", "RMSEA"]]
    fit_tex["DoF"] = fit_tex["DoF"].round().astype(int)
    _write_tabular(
        fit_tex,
        TABLES / "submission_efa_cfa_split_fit.tex",
        ["Model", "Holdout N", "DoF", "CFI", "TLI", "RMSEA"],
        ["p{0.40\\textwidth}", "r", "r", "r", "r", "r"],
    )

    focal = loadings[loadings["model"].eq("efa_informed_four_factor")].copy()
    focal["factor"] = focal["factor"].str.replace("_", " ", regex=False).str.title()
    focal = focal[["factor", "indicator_label", "std_loading"]].rename(
        columns={"factor": "Factor", "indicator_label": "Indicator", "std_loading": "Std. loading"}
    )
    _write_tabular(
        focal,
        TABLES / "submission_efa_cfa_heldout_loadings.tex",
        ["Factor", "Indicator", "Std. loading"],
        ["p{0.28\\textwidth}", "p{0.34\\textwidth}", "r"],
    )

    summary = {
        "efa_rows": int(manifest.loc[manifest["split"].eq("efa_discovery"), "rows"].iloc[0]),
        "cfa_rows": int(manifest.loc[manifest["split"].eq("cfa_holdout"), "rows"].iloc[0]),
        "retained_factors_parallel": n_factors,
        "legacy_three_factor_cfi": float(fit.loc[fit["model"].eq("legacy_three_factor"), "CFI"].iloc[0]),
        "legacy_three_factor_rmsea": float(fit.loc[fit["model"].eq("legacy_three_factor"), "RMSEA"].iloc[0]),
        "efa_four_factor_cfi": float(fit.loc[fit["model"].eq("efa_informed_four_factor"), "CFI"].iloc[0]),
        "efa_four_factor_rmsea": float(fit.loc[fit["model"].eq("efa_informed_four_factor"), "RMSEA"].iloc[0]),
    }
    pd.DataFrame([summary]).to_csv(OUT / "paper_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    build_outputs()
