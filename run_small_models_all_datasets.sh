#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
DATA_ROOT="${CALE_DATA_ROOT:-${PROJECT_ROOT}/data}"
OUTPUT_DIR="${CALE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/small_models_all}"
MODEL_PRESET="${CALE_MODEL_PRESET:-open_small}"
RUN_MODE="${CALE_RUN_MODE:-full}"
FRAMING="${CALE_FRAMING:-neutral}"
LIMIT="${CALE_LIMIT:-20}"
INCLUDE_TRAIN="${CALE_INCLUDE_TRAIN:-0}"
RUN_STRESS_SUMMARY="${CALE_RUN_STRESS_SUMMARY:-0}"
LOG_DIR="${CALE_LOG_DIR:-${OUTPUT_DIR}/logs}"
DRY_RUN="${CALE_DRY_RUN:-0}"

status() {
  printf '[small-all] %s\n' "$1"
}

line() {
  printf '%s\n' '============================================================'
}

usage() {
  cat <<'USAGE'
Usage:
  bash run_small_models_all_datasets.sh

Runs the small-model preset over every prepared dataset found under:
  data/*/prepared/*.jsonl

Environment overrides:
  CALE_MODEL_PRESET=open_small
  CALE_RUN_MODE=full|smoke
  CALE_LIMIT=20
  CALE_FRAMING=neutral|assertive|authoritative|polite_misleading
  CALE_DATASETS="data/fever/prepared/dev_prepared.jsonl data/scifact/prepared/dev_prepared.jsonl"
  CALE_DATA_ROOT=data
  CALE_OUTPUT_DIR=outputs/small_models_all
  CALE_INCLUDE_TRAIN=1
  CALE_RUN_STRESS_SUMMARY=1
  CALE_LOG_DIR=outputs/small_models_all/logs
  CALE_DRY_RUN=1

Notes:
  By default, train_prepared.jsonl is skipped to avoid accidentally running huge
  training splits. Set CALE_INCLUDE_TRAIN=1 if you really want train files.
USAGE
}

dataset_slug() {
  local path="$1"
  local rel="$path"
  rel="${rel#${PROJECT_ROOT}/}"
  rel="${rel#./}"
  rel="${rel#data/}"
  rel="${rel%_prepared.jsonl}"
  rel="${rel%.jsonl}"
  rel="${rel//\/prepared\//_}"
  rel="${rel//\//_}"
  printf '%s' "$rel" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's#[^a-z0-9]+#_#g; s#^_+##; s#_+$##'
}

collect_datasets() {
  if [[ -n "${CALE_DATASETS:-}" ]]; then
    for dataset in $CALE_DATASETS; do
      printf '%s\n' "$dataset"
    done
    return
  fi

  if [[ ! -d "$DATA_ROOT" ]]; then
    return
  fi

  find "$DATA_ROOT" -path '*/prepared/*.jsonl' -type f | sort
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "$RUN_MODE" != "smoke" && "$RUN_MODE" != "full" ]]; then
  printf 'CALE_RUN_MODE must be smoke or full, got: %s\n' "$RUN_MODE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

DATASETS=()
while IFS= read -r dataset; do
  [[ -z "$dataset" ]] && continue
  if [[ "$dataset" != /* ]]; then
    dataset="${PROJECT_ROOT}/${dataset}"
  fi
  if [[ ! -f "$dataset" ]]; then
    status "Skipping missing dataset: $dataset"
    continue
  fi
  if [[ "$INCLUDE_TRAIN" != "1" && "$(basename "$dataset")" == train* ]]; then
    status "Skipping train split by default: $dataset"
    continue
  fi
  DATASETS+=("$dataset")
done < <(collect_datasets)

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  printf 'No prepared datasets found under %s\n' "$DATA_ROOT" >&2
  printf 'Expected files like data/fever/prepared/dev_prepared.jsonl.\n' >&2
  printf 'Run bash download_fever_data.sh first, or set CALE_DATASETS manually.\n' >&2
  exit 1
fi

START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
TOTAL_START_SECONDS="$SECONDS"

line
status "CALE small-model batch run"
line
status "Started: ${START_TIME}"
status "Project root: ${PROJECT_ROOT}"
status "Datasets found: ${#DATASETS[@]}"
status "Model preset: ${MODEL_PRESET}"
status "Run mode: ${RUN_MODE}"
status "Framing: ${FRAMING}"
status "Output dir: ${OUTPUT_DIR}"
status "Logs: ${LOG_DIR}"
status "Stress summary after eval: ${RUN_STRESS_SUMMARY}"
status "Dry run: ${DRY_RUN}"
if [[ "$RUN_MODE" == "smoke" ]]; then
  status "Smoke limit per model: ${LIMIT}"
fi
line

COMPLETED=()
FAILED=()

for index in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$index]}"
  DATASET_INDEX="$((index + 1))"
  SLUG="$(dataset_slug "$DATASET")"
  ROWS="$(wc -l < "$DATASET" | tr -d ' ')"
  LOG_PATH="${LOG_DIR}/${SLUG}_${FRAMING}_${RUN_MODE}_eval.log"

  line
  status "Dataset ${DATASET_INDEX}/${#DATASETS[@]}: ${SLUG}"
  status "Path: ${DATASET}"
  status "Rows: ${ROWS}"
  status "Eval log: ${LOG_PATH}"
  line

  if [[ "$DRY_RUN" == "1" ]]; then
    status "Dry run only. Would run eval for ${SLUG}."
    if [[ "$RUN_STRESS_SUMMARY" == "1" ]]; then
      status "Dry run only. Would also run stress summary for ${SLUG}."
    fi
    COMPLETED+=("$SLUG")
    continue
  fi

  DATASET_START_SECONDS="$SECONDS"
  if CALE_DATASET="$DATASET" \
    CALE_RUN_TAG_PREFIX="$SLUG" \
    CALE_OUTPUT_DIR="$OUTPUT_DIR" \
    CALE_MODEL_PRESET="$MODEL_PRESET" \
    CALE_RUN_MODE="$RUN_MODE" \
    CALE_FRAMING="$FRAMING" \
    CALE_LIMIT="$LIMIT" \
    CALE_SKIP_PREPARE=1 \
    CALE_RUN_STRESS=0 \
    CALE_SUMMARY_ONLY=0 \
    bash "${SCRIPT_DIR}/run_pipeline.sh" 2>&1 | tee "$LOG_PATH"; then
    status "Eval completed for ${SLUG} in $((SECONDS - DATASET_START_SECONDS))s."
  else
    status "Eval failed for ${SLUG}. See log: ${LOG_PATH}"
    FAILED+=("${SLUG}:eval")
    continue
  fi

  if [[ "$RUN_STRESS_SUMMARY" == "1" ]]; then
    STRESS_LOG_PATH="${LOG_DIR}/${SLUG}_${FRAMING}_${RUN_MODE}_stress_summary.log"
    line
    status "Stress summary for ${SLUG}"
    status "Stress log: ${STRESS_LOG_PATH}"
    line
    if CALE_DATASET="$DATASET" \
      CALE_RUN_TAG_PREFIX="$SLUG" \
      CALE_OUTPUT_DIR="$OUTPUT_DIR" \
      CALE_MODEL_PRESET="$MODEL_PRESET" \
      CALE_RUN_MODE="$RUN_MODE" \
      CALE_FRAMING="$FRAMING" \
      CALE_LIMIT="$LIMIT" \
      CALE_SKIP_PREPARE=1 \
      CALE_SKIP_GENERATION=1 \
      CALE_RUN_STRESS=1 \
      CALE_SUMMARY_ONLY=1 \
      bash "${SCRIPT_DIR}/run_pipeline.sh" 2>&1 | tee "$STRESS_LOG_PATH"; then
      status "Stress summary completed for ${SLUG}."
    else
      status "Stress summary failed for ${SLUG}. See log: ${STRESS_LOG_PATH}"
      FAILED+=("${SLUG}:stress")
      continue
    fi
  fi

  COMPLETED+=("$SLUG")
done

line
status "Batch finished in $((SECONDS - TOTAL_START_SECONDS))s."
status "Completed datasets: ${#COMPLETED[@]}"
for slug in "${COMPLETED[@]}"; do
  printf '  [ok] %s\n' "$slug"
done
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  status "Failed steps: ${#FAILED[@]}"
  for item in "${FAILED[@]}"; do
    printf '  [failed] %s\n' "$item"
  done
  exit 1
fi
status "All requested datasets completed."
status "Reports are under: ${OUTPUT_DIR}"
line
