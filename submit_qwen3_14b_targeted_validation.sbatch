#!/bin/bash
# Fixed wrapper for Qwen3-14B targeted validation.
#
# Smoke test:
#   sbatch --export=ALL,CALE_TARGETED_LIMIT=10 submit_qwen3_14b_targeted_validation.sbatch
#
# Full run:
#   sbatch submit_qwen3_14b_targeted_validation.sbatch
#
# This wrapper assumes the current llm_judge.py HF backend is installed on the
# server; that backend disables Qwen3 thinking in the chat template.

#SBATCH --job-name=cale-qwen3-targeted
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

export CALE_HF_JUDGE_MODEL="Qwen/Qwen3-14B"
export CALE_HF_JUDGE_TAG="qwen3_14b"
export CALE_JUDGE_DTYPE="${CALE_JUDGE_DTYPE:-bfloat16}"
export CALE_JUDGE_MAX_NEW_TOKENS="${CALE_JUDGE_MAX_NEW_TOKENS:-1800}"
export CALE_JUDGE_JSON_RETRIES="${CALE_JUDGE_JSON_RETRIES:-3}"
export CALE_JUDGE_DEVICE_MAP="${CALE_JUDGE_DEVICE_MAP:-auto}"

bash submit_hf_targeted_validation.sbatch
