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
BATCH_SIZE="${CALE_BATCH_SIZE:-1}"
MODEL_PRESET="${CALE_MODEL_PRESET:-open_small}"
SKIP_PREPARE="${CALE_SKIP_PREPARE:-0}"
RUN_STRESS="${CALE_RUN_STRESS:-0}"
SUMMARY_ONLY="${CALE_SUMMARY_ONLY:-0}"
SKIP_GENERATION="${CALE_SKIP_GENERATION:-0}"
RUN_TAG_PREFIX="${CALE_RUN_TAG_PREFIX:-fever_dev}"
RESUME="${CALE_RESUME:-0}"

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
  CALE_BATCH_SIZE=8
  CALE_SKIP_PREPARE=1
  CALE_RUN_STRESS=1
  CALE_SUMMARY_ONLY=1
  CALE_SKIP_GENERATION=1
  CALE_RUN_TAG_PREFIX=fever_dev
  CALE_RESUME=1

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
RUN_TAG="${RUN_TAG_PREFIX}_${JOINED_SLUGS}_${FRAMING}_${RUN_MODE}"
REPORT_KIND="eval"
if [[ "$RUN_STRESS" == "1" && "$SUMMARY_ONLY" == "1" ]]; then
  REPORT_KIND="stress_summary"
elif [[ "$RUN_STRESS" == "1" ]]; then
  REPORT_KIND="stress_full"
elif [[ "$SUMMARY_ONLY" == "1" ]]; then
  REPORT_KIND="eval_summary"
fi

RESPONSES_PATH="${OUTPUT_DIR}/${RUN_TAG}.jsonl"
REPORT_PATH="${OUTPUT_DIR}/${RUN_TAG}_${REPORT_KIND}_report.json"

status "Project root: ${PROJECT_ROOT}"
status "Dataset: ${DATASET}"
status "Model preset: ${MODEL_PRESET}"
status "Models: ${MODELS}"
status "Run tag prefix: ${RUN_TAG_PREFIX}"
status "Run mode: ${RUN_MODE} | framing=${FRAMING} | limit=${LIMIT} | batch_size=${BATCH_SIZE} | stress=${RUN_STRESS} | summary_only=${SUMMARY_ONLY} | skip_generation=${SKIP_GENERATION} | resume=${RESUME}"
status "Response output: ${RESPONSES_PATH}"
status "Report output: ${REPORT_PATH}"
status "Visualization input should be the report JSON, not the responses JSONL."

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
  --batch-size "$BATCH_SIZE"
)

if [[ "$RESUME" == "1" ]]; then
  GEN_ARGS+=(--resume)
fi

if [[ "$RUN_MODE" == "smoke" ]]; then
  GEN_ARGS+=(--limit "$LIMIT")
fi

if [[ "$SKIP_GENERATION" == "1" ]]; then
  status "Skipping response generation and reusing existing response JSONL."
else
  status "Generating model responses."
  "${GEN_ARGS[@]}"
fi

if [[ ! -s "$RESPONSES_PATH" ]]; then
  printf 'Response file is empty: %s\n' "$RESPONSES_PATH" >&2
  printf 'If you used CALE_SKIP_GENERATION=1, run once without it first.\n' >&2
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
printf '\nGenerated files:\n'
printf '  Responses JSONL: %s\n' "$RESPONSES_PATH"
printf '  Report JSON:     %s\n' "$REPORT_PATH"
printf '\nUse this in visualize_results.ipynb:\n'
printf '  RESULTS_PATH = Path("%s")\n' "$REPORT_PATH"
