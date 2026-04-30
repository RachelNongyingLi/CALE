#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/fever"
PREPARED_DIR="${DATA_DIR}/prepared"

TRAIN_URL="https://fever.ai/download/fever/train.jsonl"
DEV_URL="https://fever.ai/download/fever/shared_task_dev.jsonl"
WIKI_URL="https://fever.ai/download/fever/wiki-pages.zip"

DO_PREPARE=1

status() {
  printf '[fever-data] %s\n' "$1"
}

download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    status "Skipping existing file: $output"
    return
  fi
  status "Downloading $(basename "$output")"
  curl -L "$url" -o "$output"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-only)
      DO_PREPARE=0
      shift
      ;;
    --data-dir)
      DATA_DIR="$2"
      PREPARED_DIR="${DATA_DIR}/prepared"
      shift 2
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      printf 'Usage: %s [--download-only] [--data-dir PATH]\n' "$0" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$DATA_DIR" "$PREPARED_DIR"

download_if_missing "$TRAIN_URL" "${DATA_DIR}/train.jsonl"
download_if_missing "$DEV_URL" "${DATA_DIR}/shared_task_dev.jsonl"
download_if_missing "$WIKI_URL" "${DATA_DIR}/wiki-pages.zip"

if [[ "$DO_PREPARE" -eq 1 ]]; then
  status "Preparing CALE-ready FEVER files"
  python3 "${SCRIPT_DIR}/prepare_fever.py" \
    --input "${DATA_DIR}/train.jsonl" \
    --output "${PREPARED_DIR}/train_prepared.jsonl" \
    --wiki-source "${DATA_DIR}/wiki-pages.zip"

  python3 "${SCRIPT_DIR}/prepare_fever.py" \
    --input "${DATA_DIR}/shared_task_dev.jsonl" \
    --output "${PREPARED_DIR}/dev_prepared.jsonl" \
    --wiki-source "${DATA_DIR}/wiki-pages.zip"
fi

status "Done."
printf 'Raw data directory: %s\n' "$DATA_DIR"
if [[ "$DO_PREPARE" -eq 1 ]]; then
  printf 'Prepared train file: %s\n' "${PREPARED_DIR}/train_prepared.jsonl"
  printf 'Prepared dev file: %s\n' "${PREPARED_DIR}/dev_prepared.jsonl"
fi
