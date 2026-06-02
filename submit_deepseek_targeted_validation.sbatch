#!/bin/bash
# Run DeepSeek V4-Pro API judging for the targeted validation subsets.
#
# Usage on Galvani:
#   export DEEPSEEK_API_KEY="..."
#   sbatch --export=ALL,DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" submit_deepseek_targeted_validation.sbatch
#
# This job does not need a GPU. Do not write the API key into this file.

#SBATCH --job-name=cale-ds-targeted
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /mnt/lustre/home/kelava/koh927/thesis/CALE
mkdir -p outputs/slurm outputs/subsets figures/controlled_framing_n300_deepseek figures/boundary_hard_real_subset_deepseek

echo "[slurm] Job started at $(date)"
echo "[slurm] Running on $(hostname)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jupyterenv

export TOKENIZERS_PARALLELISM=false

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[slurm] ERROR: DEEPSEEK_API_KEY is not set." >&2
  exit 2
fi

LIMIT_ARGS=()
if [[ -n "${CALE_TARGETED_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${CALE_TARGETED_LIMIT}")
  echo "[slurm] Applying CALE_TARGETED_LIMIT=${CALE_TARGETED_LIMIT}"
fi

python experiment.py \
  --dataset outputs/subsets/controlled_framing_reuse_response_n300.jsonl \
  --judge deepseek \
  --model deepseek-v4-pro \
  --variants direct_llm_judge full_attack_aware_cale \
  --repeats 1 \
  "${LIMIT_ARGS[@]}" \
  --output outputs/subsets/controlled_framing_reuse_response_n300_deepseek_eval_report.json \
  --behavior-matrix-output outputs/subsets/controlled_framing_reuse_response_n300_deepseek_behavior_matrix.csv \
  --summary-only \
  --pretty

python analyze_controlled_framing.py \
  --behavior-matrix outputs/subsets/controlled_framing_reuse_response_n300_deepseek_behavior_matrix.csv \
  --manifest outputs/subsets/controlled_framing_reuse_response_n300_manifest.csv \
  --output-dir figures/controlled_framing_n300_deepseek

python experiment.py \
  --dataset outputs/subsets/boundary_hard_real_subset.jsonl \
  --judge deepseek \
  --model deepseek-v4-pro \
  --variants direct_llm_judge full_attack_aware_cale \
  --repeats 1 \
  "${LIMIT_ARGS[@]}" \
  --output outputs/subsets/boundary_hard_real_subset_deepseek_eval_report.json \
  --behavior-matrix-output outputs/subsets/boundary_hard_real_subset_deepseek_behavior_matrix.csv \
  --summary-only \
  --pretty

python analyze_boundary_hard_subset.py \
  --behavior-matrix outputs/subsets/boundary_hard_real_subset_deepseek_behavior_matrix.csv \
  --manifest outputs/subsets/boundary_hard_real_subset_manifest.csv \
  --output-dir figures/boundary_hard_real_subset_deepseek

echo "[slurm] Job finished at $(date)"
