#!/bin/bash
# Run local Hugging Face evaluator judging for targeted validation subsets.
#
# This is the generic HF runner. Prefer the fixed wrappers when possible:
#   submit_qwen25_7b_targeted_validation.sbatch
#   submit_gemma3_12b_targeted_validation.sbatch
#
# Smoke test through a wrapper:
#   sbatch --export=ALL,CALE_TARGETED_LIMIT=10 submit_qwen25_7b_targeted_validation.sbatch
#
# Direct generic usage:
#   sbatch --export=ALL,CALE_TARGETED_LIMIT=10,CALE_HF_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct",CALE_HF_JUDGE_TAG="qwen25_7b" submit_hf_targeted_validation.sbatch
#
# Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN in the shell if the model requires access.
# Do not write tokens into this file.

#SBATCH --job-name=cale-hf-targeted
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /mnt/lustre/home/kelava/koh927/thesis/CALE
mkdir -p outputs/slurm outputs/subsets

echo "[slurm] Job started at $(date)"
echo "[slurm] Running on $(hostname)"
echo "[slurm] CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"

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
export CALE_JUDGE_MAX_NEW_TOKENS="${CALE_JUDGE_MAX_NEW_TOKENS:-1200}"
export CALE_JUDGE_DTYPE="${CALE_JUDGE_DTYPE:-float16}"
export CALE_JUDGE_DEVICE_MAP="${CALE_JUDGE_DEVICE_MAP:-auto}"

JUDGE_MODEL="${CALE_HF_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_TAG="${CALE_HF_JUDGE_TAG:-qwen25_7b}"

LIMIT_ARGS=()
if [[ -n "${CALE_TARGETED_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${CALE_TARGETED_LIMIT}")
  echo "[slurm] Applying CALE_TARGETED_LIMIT=${CALE_TARGETED_LIMIT}"
fi

mkdir -p "figures/controlled_framing_n300_${JUDGE_TAG}" "figures/boundary_hard_real_subset_${JUDGE_TAG}"

echo "[slurm] HF judge model: ${JUDGE_MODEL}"
echo "[slurm] HF judge tag: ${JUDGE_TAG}"

"${PYTHON_BIN}" experiment.py \
  --dataset outputs/subsets/controlled_framing_reuse_response_n300.jsonl \
  --judge hf \
  --model "${JUDGE_MODEL}" \
  --variants direct_llm_judge full_attack_aware_cale \
  --repeats 1 \
  "${LIMIT_ARGS[@]}" \
  --output "outputs/subsets/controlled_framing_reuse_response_n300_${JUDGE_TAG}_eval_report.json" \
  --behavior-matrix-output "outputs/subsets/controlled_framing_reuse_response_n300_${JUDGE_TAG}_behavior_matrix.csv" \
  --summary-only \
  --pretty

"${PYTHON_BIN}" analyze_controlled_framing.py \
  --behavior-matrix "outputs/subsets/controlled_framing_reuse_response_n300_${JUDGE_TAG}_behavior_matrix.csv" \
  --manifest outputs/subsets/controlled_framing_reuse_response_n300_manifest.csv \
  --output-dir "figures/controlled_framing_n300_${JUDGE_TAG}"

"${PYTHON_BIN}" experiment.py \
  --dataset outputs/subsets/boundary_hard_real_subset.jsonl \
  --judge hf \
  --model "${JUDGE_MODEL}" \
  --variants direct_llm_judge full_attack_aware_cale \
  --repeats 1 \
  "${LIMIT_ARGS[@]}" \
  --output "outputs/subsets/boundary_hard_real_subset_${JUDGE_TAG}_eval_report.json" \
  --behavior-matrix-output "outputs/subsets/boundary_hard_real_subset_${JUDGE_TAG}_behavior_matrix.csv" \
  --summary-only \
  --pretty

"${PYTHON_BIN}" analyze_boundary_hard_subset.py \
  --behavior-matrix "outputs/subsets/boundary_hard_real_subset_${JUDGE_TAG}_behavior_matrix.csv" \
  --manifest outputs/subsets/boundary_hard_real_subset_manifest.csv \
  --output-dir "figures/boundary_hard_real_subset_${JUDGE_TAG}"

echo "[slurm] Job finished at $(date)"
