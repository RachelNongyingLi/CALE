# CALE Experiment Pipeline

This folder contains the code for running CALE experiments on adversarial factuality correction.

## Files

- `prepare_fever.py`: converts raw FEVER files into CALE-ready claim resources.
- `generate_responses.py`: generates target model responses with `stub` or Hugging Face models such as `Qwen/Qwen2.5-1.5B-Instruct`.
- `experiment.py`: runs direct-judge and CALE evaluator variants, with optional stress tests.
- `cale_demo.py`: core CALE schema, heuristic judge, scoring, calibration, and aggregation.
- `llm_judge.py`: direct and structured judge backends.
- `perturbations.py`: stress-test perturbation definitions.
- `download_fever_data.sh`: downloads raw FEVER data and optionally prepares it.
- `run_pipeline.sh`: one-command FEVER generation and CALE evaluation pipeline.
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

## Recommended One-Command Pipeline

The default pipeline compares a Qwen-family model with a Llama-family model that should fit a single RTX 2080 Ti:

```text
Qwen/Qwen2.5-1.5B-Instruct
meta-llama/Llama-3.2-1B-Instruct
```

The Meta Llama model may require accepting the Hugging Face license and setting `HF_TOKEN`:

```bash
export HF_TOKEN="your_huggingface_token"
```

Run a 20-example smoke test:

```bash
bash run_pipeline.sh
```

The default preset is:

```bash
CALE_MODEL_PRESET=open_small bash run_pipeline.sh
```

Useful model presets:

```text
open_small        Qwen2.5-1.5B + Llama3.2-1B
open_tiny         Qwen2.5-0.5B + Llama3.2-1B
open_larger       Qwen2.5-1.5B + Llama3.2-3B
open_three_family Qwen2.5-1.5B + Llama3.2-1B + Gemma2-2B
qwen_only         Qwen2.5-1.5B
llama_only        Llama3.2-1B
```

Run the full FEVER dev split with stress-test summaries:

```bash
CALE_RUN_MODE=full \
CALE_RUN_STRESS=1 \
CALE_SUMMARY_ONLY=1 \
bash run_pipeline.sh
```

If the Llama model is not accessible yet, run only Qwen:

```bash
CALE_MODEL_PRESET=qwen_only bash run_pipeline.sh
```

If the 1B Llama model is fast and you want a stronger Llama-family comparison, try:

```bash
CALE_MODEL_PRESET=open_larger bash run_pipeline.sh
```

## Manual Generation

Smoke test first:

```bash
python generate_responses.py \
  --dataset "data/fever/prepared/dev_prepared.jsonl" \
  --models "Qwen/Qwen2.5-1.5B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" \
  --output "outputs/fever_dev_qwen25_15b_llama32_1b_neutral_smoke.jsonl" \
  --limit 20 \
  --framing neutral \
  --device-map auto
```

Full dev generation:

```bash
python generate_responses.py \
  --dataset "data/fever/prepared/dev_prepared.jsonl" \
  --models "Qwen/Qwen2.5-1.5B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" \
  --output "outputs/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl" \
  --framing neutral \
  --device-map auto
```

You can repeat generation with `--framing assertive` or `--framing authoritative` to create matched adversarial framing sets.

## Run CALE Experiments

Smoke test:

```bash
python experiment.py \
  --dataset "outputs/fever_dev_qwen25_15b_llama32_1b_neutral_smoke.jsonl" \
  --limit 40 \
  --output "outputs/fever_dev_qwen25_15b_llama32_1b_neutral_smoke_report.json" \
  --pretty
```

This smoke report keeps row-level `predictions`, which is useful for checking the visualization notebook.

Full internal evaluation with stress tests:

```bash
python experiment.py \
  --dataset "outputs/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl" \
  --stress \
  --summary-only \
  --output "outputs/fever_dev_qwen25_15b_llama32_1b_neutral_full_report.json" \
  --pretty
```

Use `--summary-only` for large runs. Without it, the report includes every prediction and every stress-test row, which can easily exceed 1GB.

## Visualize Results

Open `visualize_results.ipynb` and set `RESULTS_PATH` to the report JSON from `experiment.py`, not the generated response JSONL. The notebook defaults to:

```text
outputs/fever_dev_qwen25_15b_llama32_1b_neutral_smoke_report.json
```
