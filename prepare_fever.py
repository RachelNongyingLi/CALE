#!/usr/bin/env python3
"""Prepare FEVER into a CALE-friendly resource JSONL.

This script converts raw FEVER JSONL files into a normalized resource format
that can later be consumed by `generate_responses.py`.

The output schema is intentionally lightweight:
  - id
  - resource
  - base_claim
  - reference_label
  - supporting_evidence
  - reference_evidence
  - verifiable
  - fever_evidence

If wiki pages are available, evidence sentence ids are resolved to evidence
text. If wiki pages are unavailable or incomplete, the script still emits a
resource file with empty evidence text (or raw FEVER evidence ids), so the
pipeline can be smoke-tested first.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def load_fever_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("label") not in VALID_LABELS:
                continue
            claim = (obj.get("claim") or "").strip()
            if not claim:
                continue
            rows.append(obj)
    return rows


def collect_needed_pages(rows: Iterable[dict[str, Any]]) -> set[str]:
    pages: set[str] = set()
    for row in rows:
        for evidence_group in row.get("evidence", []):
            for item in evidence_group:
                if len(item) >= 4:
                    pages.add(str(item[2]))
    return pages


def parse_lines_field(lines_field: Any) -> dict[int, str]:
    """Parse FEVER wiki `lines` payload into {sentence_id: sentence_text}."""
    sentence_map: dict[int, str] = {}
    if not lines_field:
        return sentence_map

    if isinstance(lines_field, list):
        for item in lines_field:
            if isinstance(item, dict):
                sent_id = item.get("line_num")
                sent_text = item.get("sentence") or item.get("text") or ""
                if isinstance(sent_id, int):
                    sentence_map[sent_id] = sent_text.strip()
        return sentence_map

    if not isinstance(lines_field, str):
        return sentence_map

    for raw_line in lines_field.split("\n"):
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t", 1)
        if len(parts) != 2:
            continue
        sent_id_raw, sent_text = parts
        try:
            sent_id = int(sent_id_raw)
        except ValueError:
            continue
        sentence_map[sent_id] = sent_text.strip()
    return sentence_map


def page_id_candidates(page_id: str) -> list[str]:
    candidates = [page_id]
    normalized = page_id.replace("_", " ")
    if normalized not in candidates:
        candidates.append(normalized)
    return candidates


def load_wiki_index_from_jsonl_handle(
    handle: Iterable[str],
    needed_pages: set[str],
    page_to_sentences: dict[str, dict[int, str]],
) -> None:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        page_id = str(obj.get("id", ""))
        if not page_id:
            continue

        matched_key = None
        for candidate in page_id_candidates(page_id):
            if candidate in needed_pages:
                matched_key = candidate
                break
        if matched_key is None:
            continue

        sentences = parse_lines_field(obj.get("lines"))
        if sentences:
            page_to_sentences[matched_key].update(sentences)


def load_wiki_index(wiki_source: Path | None, needed_pages: set[str]) -> dict[str, dict[int, str]]:
    page_to_sentences: dict[str, dict[int, str]] = defaultdict(dict)
    if wiki_source is None or not wiki_source.exists():
        return page_to_sentences

    if wiki_source.is_file() and wiki_source.suffix == ".zip":
        try:
            with zipfile.ZipFile(wiki_source) as archive:
                for member in archive.namelist():
                    if member.endswith("/"):
                        continue
                    with archive.open(member) as raw_handle:
                        text_handle = (
                            line.decode("utf-8", errors="ignore") for line in raw_handle
                        )
                        load_wiki_index_from_jsonl_handle(
                            text_handle,
                            needed_pages=needed_pages,
                            page_to_sentences=page_to_sentences,
                        )
        except zipfile.BadZipFile:
            return page_to_sentences
        return page_to_sentences

    if wiki_source.is_dir():
        for path in sorted(wiki_source.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".jsonl", ".json", ".txt"}:
                continue
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                load_wiki_index_from_jsonl_handle(
                    handle,
                    needed_pages=needed_pages,
                    page_to_sentences=page_to_sentences,
                )
    return page_to_sentences


def resolve_evidence_texts(
    fever_row: dict[str, Any],
    wiki_index: dict[str, dict[int, str]],
) -> tuple[list[str], list[dict[str, Any]]]:
    evidence_texts: list[str] = []
    evidence_records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for evidence_group in fever_row.get("evidence", []):
        for item in evidence_group:
            if len(item) < 4:
                continue
            page = str(item[2])
            sent_id = int(item[3])
            key = (page, sent_id)
            if key in seen:
                continue
            seen.add(key)

            sentence = ""
            page_map = wiki_index.get(page) or wiki_index.get(page.replace("_", " "))
            if page_map:
                sentence = page_map.get(sent_id, "").strip()
            if sentence:
                evidence_texts.append(sentence)
            evidence_records.append(
                {
                    "page": page,
                    "sentence_id": sent_id,
                    "text": sentence,
                }
            )
    return evidence_texts, evidence_records


def normalize_row(
    fever_row: dict[str, Any],
    wiki_index: dict[str, dict[int, str]],
) -> dict[str, Any]:
    evidence_texts, evidence_records = resolve_evidence_texts(fever_row, wiki_index)
    supporting_evidence = " ".join(evidence_texts).strip()
    if not supporting_evidence and evidence_records:
        supporting_evidence = json.dumps(evidence_records, ensure_ascii=False)

    return {
        "id": f"fever_{fever_row['id']}",
        "resource": "FEVER",
        "base_claim": fever_row["claim"].strip(),
        "reference_label": fever_row["label"],
        "supporting_evidence": supporting_evidence,
        "reference_evidence": evidence_texts,
        "verifiable": fever_row.get("verifiable", ""),
        "fever_evidence": evidence_records,
    }


def maybe_limit_by_label(rows: list[dict[str, Any]], max_per_label: int | None) -> list[dict[str, Any]]:
    if max_per_label is None:
        return rows
    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for row in rows:
        label = row["label"]
        if counts[label] >= max_per_label:
            continue
        counts[label] += 1
        selected.append(row)
    return selected


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FEVER for CALE experiments.")
    parser.add_argument("--input", required=True, help="Raw FEVER JSONL path.")
    parser.add_argument("--output", required=True, help="Prepared JSONL output path.")
    parser.add_argument(
        "--wiki-source",
        help="Optional FEVER wiki source: a wiki-pages zip or extracted directory.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        help="Optionally keep at most N examples per FEVER label.",
    )
    parser.add_argument(
        "--keep-nei",
        action="store_true",
        help="Keep NOT ENOUGH INFO items. If omitted, only SUPPORTS/REFUTES are kept.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    wiki_source = Path(args.wiki_source) if args.wiki_source else None

    rows = load_fever_rows(input_path)
    if not args.keep_nei:
        rows = [row for row in rows if row["label"] != "NOT ENOUGH INFO"]
    rows = maybe_limit_by_label(rows, args.max_per_label)

    needed_pages = collect_needed_pages(rows)
    wiki_index = load_wiki_index(wiki_source, needed_pages)
    prepared = [normalize_row(row, wiki_index) for row in rows]
    write_jsonl(prepared, output_path)

    with_evidence = sum(1 for row in prepared if row["reference_evidence"])
    print(
        f"Wrote {len(prepared)} FEVER resource rows to {output_path} "
        f"({with_evidence} with resolved evidence text)"
    )


if __name__ == "__main__":
    main()
