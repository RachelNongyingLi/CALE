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

bash submit_hf_targeted_validation.sbatch
