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

bash submit_hf_targeted_validation.sbatch
