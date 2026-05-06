# CALE Experiment Pipeline

This folder contains the code for running CALE experiments on adversarial factuality correction.

## Files

- `prepare_fever.py`: converts raw FEVER files into CALE-ready claim resources.
- `generate_responses.py`: generates target model responses with `stub` or Hugging Face models such as `Qwen/Qwen2.5-7B-Instruct`.
- `experiment.py`: runs direct-judge and CALE evaluator variants, with optional stress tests.
- `cale_demo.py`: core CALE schema, heuristic judge, scoring, calibration, and aggregation.
- `llm_judge.py`: direct and structured judge backends.
- `perturbations.py`: stress-test perturbation definitions.
- `download_fever_data.sh`: downloads raw FEVER data and optionally prepares it.
- `environment.yml`: conda environment for notebooks and pipeline dependencies except CUDA PyTorch.

## Environment

Create the conda environment:

```bash
cd /thesis/CALE
conda env create -f environment.yml
conda activate jupyterenv
```

Install a CUDA PyTorch build on the GPU server. Choose the command that matches the server CUDA version. For CUDA 12.1, for example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check that the server sees the GPU:

```bash
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## Prepare FEVER

If raw FEVER is already present under `data/fever/`, prepare the dev split:

```bash
python prepare_fever.py \
  --input "data/fever/shared_task_dev.jsonl" \
  --output "data/fever/prepared/dev_prepared.jsonl" \
  --wiki-source "data/fever/wiki-pages.zip" \
  --keep-nei
```

To download and prepare from scratch:

```bash
bash download_fever_data.sh
```

## Generate Qwen Responses

Smoke test first:

```bash
python generate_responses.py \
  --dataset "data/fever/prepared/dev_prepared.jsonl" \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --output "outputs/fever_dev_qwen25_7b_smoke.jsonl" \
  --limit 20 \
  --framing neutral \
  --device-map auto
```

Full dev generation:

```bash
python generate_responses.py \
  --dataset "data/fever/prepared/dev_prepared.jsonl" \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --output "outputs/fever_dev_qwen25_7b_neutral.jsonl" \
  --framing neutral \
  --device-map auto
```

You can repeat generation with `--framing assertive` or `--framing authoritative` to create matched adversarial framing sets.

## Run CALE Experiments

Smoke test:

```bash
python experiment.py \
  --dataset "outputs/fever_dev_qwen25_7b_smoke.jsonl" \
  --limit 20 \
  --summary-only \
  --output "outputs/fever_dev_qwen25_7b_smoke_report.json" \
  --pretty
```

Full internal evaluation with stress tests:

```bash
python experiment.py \
  --dataset "outputs/fever_dev_qwen25_7b_neutral.jsonl" \
  --stress \
  --summary-only \
  --output "outputs/fever_dev_qwen25_7b_neutral_stress_summary.json" \
  --pretty
```

Use `--summary-only` for large runs. Without it, the report includes every prediction and every stress-test row, which can easily exceed 1GB.
