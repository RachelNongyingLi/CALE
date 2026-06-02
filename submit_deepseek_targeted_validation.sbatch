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

PYTHON_BIN="python"
CONDA_SH=""
for candidate in \
  "${HOME}/.conda/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/home/kelava/koh927/.conda/etc/profile.d/conda.sh"; do
  if [[ -f "${candidate}" ]]; then
    CONDA_SH="${candidate}"
    break
  fi
done

if [[ -n "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate jupyterenv
elif [[ -x "${HOME}/.conda/envs/jupyterenv/bin/python" ]]; then
  PYTHON_BIN="${HOME}/.conda/envs/jupyterenv/bin/python"
elif [[ -x "/home/kelava/koh927/.conda/envs/jupyterenv/bin/python" ]]; then
  PYTHON_BIN="/home/kelava/koh927/.conda/envs/jupyterenv/bin/python"
else
  echo "[slurm] WARNING: could not find conda init; falling back to python on PATH." >&2
fi

echo "[slurm] Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

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

"${PYTHON_BIN}" experiment.py \
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

"${PYTHON_BIN}" analyze_controlled_framing.py \
  --behavior-matrix outputs/subsets/controlled_framing_reuse_response_n300_deepseek_behavior_matrix.csv \
  --manifest outputs/subsets/controlled_framing_reuse_response_n300_manifest.csv \
  --output-dir figures/controlled_framing_n300_deepseek

"${PYTHON_BIN}" experiment.py \
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

"${PYTHON_BIN}" analyze_boundary_hard_subset.py \
  --behavior-matrix outputs/subsets/boundary_hard_real_subset_deepseek_behavior_matrix.csv \
  --manifest outputs/subsets/boundary_hard_real_subset_manifest.csv \
  --output-dir figures/boundary_hard_real_subset_deepseek

echo "[slurm] Job finished at $(date)"
