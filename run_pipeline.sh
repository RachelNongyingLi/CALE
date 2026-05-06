#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DATASET="${CALE_DATASET:-${PROJECT_ROOT}/data/fever/prepared/dev_prepared.jsonl}"
OUTPUT_DIR="${CALE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
RUN_MODE="${CALE_RUN_MODE:-smoke}"
LIMIT="${CALE_LIMIT:-20}"
FRAMING="${CALE_FRAMING:-neutral}"
DEVICE_MAP="${CALE_DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${CALE_MAX_NEW_TOKENS:-160}"
TEMPERATURE="${CALE_TEMPERATURE:-0.0}"
MODEL_PRESET="${CALE_MODEL_PRESET:-open_small}"
SKIP_PREPARE="${CALE_SKIP_PREPARE:-0}"
RUN_STRESS="${CALE_RUN_STRESS:-0}"
SUMMARY_ONLY="${CALE_SUMMARY_ONLY:-0}"

if [[ -n "${CALE_MODELS:-}" ]]; then
  MODELS="$CALE_MODELS"
  MODEL_PRESET="custom"
else
  case "$MODEL_PRESET" in
    open_small)
      MODELS="Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-1B-Instruct"
      ;;
    open_tiny)
      MODELS="Qwen/Qwen2.5-0.5B-Instruct meta-llama/Llama-3.2-1B-Instruct"
      ;;
    open_larger)
      MODELS="Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-3B-Instruct"
      ;;
    open_three_family)
      MODELS="Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-1B-Instruct google/gemma-2-2b-it"
      ;;
    qwen_only)
      MODELS="Qwen/Qwen2.5-1.5B-Instruct"
      ;;
    llama_only)
      MODELS="meta-llama/Llama-3.2-1B-Instruct"
      ;;
    *)
      printf 'Unknown CALE_MODEL_PRESET: %s\n' "$MODEL_PRESET" >&2
      printf 'Use one of: open_small, open_tiny, open_larger, open_three_family, qwen_only, llama_only.\n' >&2
      exit 1
      ;;
  esac
fi

status() {
  printf '[cale-pipeline] %s\n' "$1"
}

usage() {
  cat <<'USAGE'
Usage:
  bash run_pipeline.sh

Environment overrides:
  CALE_RUN_MODE=smoke|full
  CALE_LIMIT=20
  CALE_MODEL_PRESET=open_small|open_tiny|open_larger|open_three_family|qwen_only|llama_only
  CALE_MODELS="Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-1B-Instruct"
  CALE_DATASET=/path/to/dev_prepared.jsonl
  CALE_OUTPUT_DIR=/path/to/outputs
  CALE_FRAMING=neutral|assertive|authoritative|polite_misleading
  CALE_DEVICE_MAP=auto
  CALE_MAX_NEW_TOKENS=160
  CALE_TEMPERATURE=0.0
  CALE_SKIP_PREPARE=1
  CALE_RUN_STRESS=1
  CALE_SUMMARY_ONLY=1

Notes:
  CALE_MODEL_PRESET is ignored when CALE_MODELS is set.
  meta-llama/Llama-3.2-1B-Instruct usually requires accepting the model license
  on Hugging Face and setting HF_TOKEN before running.
USAGE
}

model_slug() {
  local raw="$1"
  case "$raw" in
    "Qwen/Qwen2.5-1.5B-Instruct")
      printf 'qwen25_15b'
      return
      ;;
    "Qwen/Qwen2.5-0.5B-Instruct")
      printf 'qwen25_05b'
      return
      ;;
    "meta-llama/Llama-3.2-1B-Instruct")
      printf 'llama32_1b'
      return
      ;;
    "meta-llama/Llama-3.2-3B-Instruct")
      printf 'llama32_3b'
      return
      ;;
    "google/gemma-2-2b-it")
      printf 'gemma2_2b'
      return
      ;;
  esac
  printf '%s' "$raw" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's#meta-llama/llama-#llama#g; s#qwen/qwen#qwen#g; s#/#_#g; s#[^a-z0-9]+#_#g; s#^_+##; s#_+$##'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "$RUN_MODE" != "smoke" && "$RUN_MODE" != "full" ]]; then
  printf 'CALE_RUN_MODE must be smoke or full, got: %s\n' "$RUN_MODE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$SKIP_PREPARE" != "1" && ! -f "$DATASET" ]]; then
  status "Prepared dataset is missing; running download_fever_data.sh first."
  bash "${SCRIPT_DIR}/download_fever_data.sh"
fi

if [[ ! -f "$DATASET" ]]; then
  printf 'Prepared dataset not found: %s\n' "$DATASET" >&2
  printf 'Run bash download_fever_data.sh or set CALE_DATASET.\n' >&2
  exit 1
fi

MODEL_SLUGS=()
read -r -a MODEL_ARRAY <<< "$MODELS"
for model in "${MODEL_ARRAY[@]}"; do
  MODEL_SLUGS+=("$(model_slug "$model")")
done
JOINED_SLUGS="$(IFS=_; printf '%s' "${MODEL_SLUGS[*]}")"
RUN_TAG="fever_dev_${JOINED_SLUGS}_${FRAMING}_${RUN_MODE}"

RESPONSES_PATH="${OUTPUT_DIR}/${RUN_TAG}.jsonl"
REPORT_PATH="${OUTPUT_DIR}/${RUN_TAG}_report.json"

status "Project root: ${PROJECT_ROOT}"
status "Dataset: ${DATASET}"
status "Model preset: ${MODEL_PRESET}"
status "Models: ${MODELS}"
status "Run mode: ${RUN_MODE} | framing=${FRAMING} | limit=${LIMIT} | stress=${RUN_STRESS} | summary_only=${SUMMARY_ONLY}"
status "Response output: ${RESPONSES_PATH}"
status "Report output: ${REPORT_PATH}"

if [[ "$MODELS" == *"meta-llama/"* && -z "${HF_TOKEN:-}" ]]; then
  status "HF_TOKEN is not set. Meta Llama downloads may fail unless the model is already cached."
fi
if [[ "$MODELS" == *"google/gemma"* && -z "${HF_TOKEN:-}" ]]; then
  status "HF_TOKEN is not set. Gemma downloads may fail unless the model is already cached or publicly accessible."
fi

GEN_ARGS=(
  python "${SCRIPT_DIR}/generate_responses.py"
  --dataset "$DATASET"
  --models "${MODEL_ARRAY[@]}"
  --output "$RESPONSES_PATH"
  --framing "$FRAMING"
  --device-map "$DEVICE_MAP"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  GEN_ARGS+=(--limit "$LIMIT")
fi

status "Generating model responses."
"${GEN_ARGS[@]}"

if [[ ! -s "$RESPONSES_PATH" ]]; then
  printf 'Response file is empty: %s\n' "$RESPONSES_PATH" >&2
  exit 1
fi

status "Generated rows: $(wc -l < "$RESPONSES_PATH" | tr -d ' ')"

EXP_ARGS=(
  python "${SCRIPT_DIR}/experiment.py"
  --dataset "$RESPONSES_PATH"
  --output "$REPORT_PATH"
  --pretty
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  EXP_ARGS+=(--limit "$(( LIMIT * ${#MODEL_ARRAY[@]} ))")
fi

if [[ "$RUN_STRESS" == "1" ]]; then
  EXP_ARGS+=(--stress)
fi

if [[ "$SUMMARY_ONLY" == "1" ]]; then
  EXP_ARGS+=(--summary-only)
fi

status "Running CALE experiment."
"${EXP_ARGS[@]}"

status "Done."
printf 'Responses JSONL: %s\n' "$RESPONSES_PATH"
printf 'Report JSON: %s\n' "$REPORT_PATH"
