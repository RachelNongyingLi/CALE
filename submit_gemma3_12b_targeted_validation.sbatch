#!/bin/bash
# Fixed wrapper for Gemma3-12B targeted validation.
#
# Smoke test:
#   sbatch --export=ALL,CALE_TARGETED_LIMIT=10 submit_gemma3_12b_targeted_validation.sbatch
#
# Full run:
#   sbatch submit_gemma3_12b_targeted_validation.sbatch
#
# If model access requires it, set HF_TOKEN/HUGGINGFACE_HUB_TOKEN in the shell
# or pass it with --export. Do not write tokens into this file.

#SBATCH --job-name=cale-gemma-targeted
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

export CALE_HF_JUDGE_MODEL="google/gemma-3-12b-it"
export CALE_HF_JUDGE_TAG="gemma3_12b"
export CALE_JUDGE_DTYPE="${CALE_JUDGE_DTYPE:-bfloat16}"
export CALE_JUDGE_MAX_NEW_TOKENS="${CALE_JUDGE_MAX_NEW_TOKENS:-1800}"
export CALE_JUDGE_JSON_RETRIES="${CALE_JUDGE_JSON_RETRIES:-3}"
export CALE_JUDGE_DEVICE_MAP="${CALE_JUDGE_DEVICE_MAP:-auto}"
export CALE_JUDGE_MIN_NEW_TOKENS="${CALE_JUDGE_MIN_NEW_TOKENS:-0}"
export CALE_JUDGE_DEBUG_EMPTY="${CALE_JUDGE_DEBUG_EMPTY:-1}"
export CALE_JUDGE_SUPPRESS_PAD_TOKEN="${CALE_JUDGE_SUPPRESS_PAD_TOKEN:-1}"

bash submit_hf_targeted_validation.sbatch
