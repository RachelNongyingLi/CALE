#!/usr/bin/env python3
"""Build a real-response boundary-control hard subset.

This script turns existing real case-selection CSVs into a JSONL dataset that
can be re-evaluated by `experiment.py`. It does not create new target responses;
it reuses the fixed target-model responses and keeps selection provenance.

The subset is diagnostic. It should not be mixed with the hand-authored
boundary-control fixture or used as a target-model leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RESPONSES = Path("outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl")
DEFAULT_SELECTION_DIR = Path("figures/real_case_selection")
DEFAULT_OUTPUT = Path("outputs/subsets/boundary_hard_real_subset.jsonl")
DEFAULT_MANIFEST = Path("outputs/subsets/boundary_hard_real_subset_manifest.csv")

SELECTED_FILES = [
    "selected_nei_boundary_failure.csv",
    "selected_boundary_indicator_mismatch.csv",
    "selected_backend_disagreement.csv",
    "selected_direct_vs_full_shift.csv",
    "selected_representative_factual_handling.csv",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_selected_cases(selection_dir: Path) -> pd.DataFrame:
    parts = []
    for filename in SELECTED_FILES:
        path = selection_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_selection_file"] = filename
        if "case_category" not in df.columns:
            df["case_category"] = filename.removeprefix("selected_").removesuffix(".csv")
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No selected case CSVs found under {selection_dir}")
    selected = pd.concat(parts, ignore_index=True, sort=False)
    required = {"id", "target_model", "case_category"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected case files are missing required columns: {missing}")
    return selected


def response_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        target_model = str(row.get("target_model") or row.get("model_name") or "")
        key = (target_model, str(row.get("id", "")))
        if key in lookup:
            raise ValueError(f"Duplicate response key found: {key}")
        normalized = dict(row)
        normalized["target_model"] = target_model
        normalized["model_name"] = str(normalized.get("model_name") or target_model)
        lookup[key] = normalized
    return lookup


def compact_selection_metadata(rows: pd.DataFrame) -> dict[str, Any]:
    categories = sorted(set(map(str, rows["case_category"].dropna())))
    reasons = sorted(set(map(str, rows.get("selection_reason", pd.Series(dtype=str)).dropna())))
    backend_models = sorted(set(map(str, rows.get("evaluator_backend_model", pd.Series(dtype=str)).dropna())))
    variants = sorted(set(map(str, rows.get("evaluator_variant", pd.Series(dtype=str)).dropna())))
    return {
        "case_categories": categories,
        "selection_reasons": reasons,
        "source_evaluator_backend_models": backend_models,
        "source_evaluator_variants": variants,
        "source_case_ids": sorted(set(map(str, rows.get("case_id", pd.Series(dtype=str)).dropna()))),
        "source_selection_files": sorted(set(map(str, rows["source_selection_file"].dropna()))),
    }


def build_subset(responses: list[dict[str, Any]], selected: pd.DataFrame, max_per_category: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_per_category is not None:
        selected = (
            selected.groupby("case_category", dropna=False, group_keys=False)
            .head(max_per_category)
            .copy()
        )

    lookup = response_lookup(responses)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in selected.iterrows():
        grouped[(str(row["target_model"]), str(row["id"]))].append(idx)

    output_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for (target_model, source_id), indices in grouped.items():
        source = lookup.get((target_model, source_id))
        if source is None:
            missing.append((target_model, source_id))
            continue
        selected_rows = selected.loc[indices]
        metadata = compact_selection_metadata(selected_rows)
        subset_id = f"{source_id}::boundary_hard"
        cloned = dict(source)
        cloned["source_id"] = source_id
        cloned["id"] = subset_id
        cloned["target_model"] = target_model
        cloned["model_name"] = str(cloned.get("model_name") or target_model)
        cloned["subset_name"] = "boundary_hard_real_subset"
        cloned["subset_type"] = "real_response_diagnostic_subset"
        cloned["subset_policy"] = "selected from fixed target responses using global evaluator-audit behavior signals"
        cloned["case_categories"] = metadata["case_categories"]
        cloned["selection_reason"] = " | ".join(metadata["selection_reasons"])
        cloned["source_selection_files"] = metadata["source_selection_files"]
        cloned["source_case_ids"] = metadata["source_case_ids"]
        cloned["source_evaluator_backend_models"] = metadata["source_evaluator_backend_models"]
        cloned["source_evaluator_variants"] = metadata["source_evaluator_variants"]
        attack_profile = dict(cloned.get("attack_profile") or {})
        attack_profile.update(
            {
                "boundary_hard_real_subset": True,
                "case_categories": metadata["case_categories"],
            }
        )
        cloned["attack_profile"] = attack_profile
        output_rows.append(cloned)
        manifest_rows.append(
            {
                "id": subset_id,
                "source_id": source_id,
                "target_model": target_model,
                "model_name": cloned["model_name"],
                "reference_label": cloned.get("reference_label", ""),
                "case_categories": "; ".join(metadata["case_categories"]),
                "selection_reason": cloned["selection_reason"],
                "source_selection_files": "; ".join(metadata["source_selection_files"]),
                "subset_name": cloned["subset_name"],
                "subset_type": cloned["subset_type"],
                "subset_policy": cloned["subset_policy"],
            }
        )

    if missing:
        sample = ", ".join(f"{target}:{source_id}" for target, source_id in missing[:5])
        raise ValueError(f"{len(missing)} selected cases did not match fixed responses; first missing keys: {sample}")
    return output_rows, manifest_rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source_id",
        "target_model",
        "model_name",
        "reference_label",
        "case_categories",
        "selection_reason",
        "source_selection_files",
        "subset_name",
        "subset_type",
        "subset_policy",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-per-category", type=int, default=None)
    args = parser.parse_args()

    responses = read_jsonl(args.responses)
    selected = read_selected_cases(args.selection_dir)
    output_rows, manifest_rows = build_subset(responses, selected, args.max_per_category)
    write_jsonl(args.output, output_rows)
    write_manifest(args.manifest, manifest_rows)

    manifest = pd.DataFrame(manifest_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    print(f"Wrote {len(manifest_rows)} manifest rows to {args.manifest}")
    if not manifest.empty:
        print("Reference labels:")
        print(manifest["reference_label"].value_counts(dropna=False).to_string())
        print("Target models:")
        print(manifest["target_model"].value_counts(dropna=False).to_string())
        print("Case categories:")
        exploded = manifest.assign(case_categories=manifest["case_categories"].str.split("; ")).explode("case_categories")
        print(exploded["case_categories"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
