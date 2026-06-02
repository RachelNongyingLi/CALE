#!/bin/bash
# Fixed wrapper for Qwen2.5-7B targeted validation.
#
# Smoke test:
#   sbatch --export=ALL,CALE_TARGETED_LIMIT=10 submit_qwen25_7b_targeted_validation.sbatch
#
# Full run:
#   sbatch submit_qwen25_7b_targeted_validation.sbatch

#SBATCH --job-name=cale-qwen25-targeted
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

export CALE_HF_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export CALE_HF_JUDGE_TAG="qwen25_7b"
export CALE_JUDGE_DTYPE="${CALE_JUDGE_DTYPE:-bfloat16}"
export CALE_JUDGE_MAX_NEW_TOKENS="${CALE_JUDGE_MAX_NEW_TOKENS:-1800}"
export CALE_JUDGE_JSON_RETRIES="${CALE_JUDGE_JSON_RETRIES:-3}"
export CALE_JUDGE_DEVICE_MAP="${CALE_JUDGE_DEVICE_MAP:-auto}"

bash submit_hf_targeted_validation.sbatch
