# CALE Code Track: Server Workflow And Handoff Notes

This file is the quickest way to orient a future collaborator, notebook session, or coding agent working on the code and experiment track.

For the top-level thesis split between paper writing and code work, see `../WORKSPACE_OVERVIEW.md`.

## Core Reality

The repository contains the code, but the main experiments are usually run on Galvani, not on the local laptop.

- Edit code and notebooks locally, then transfer code by the cluster-approved method. Do not run `rsync`, IDE backends, Claude, Python, notebooks, or other user workloads on login nodes.
- Run heavy Hugging Face generation and full FEVER experiments through Slurm on Galvani.
- Treat server-side `outputs/` as the source of truth for fresh experiment artifacts.
- Use Jupyter mainly for inspection, visualization, and debugging, not as the main long-running experiment process.
- Never run computation, notebooks, IDE backends, Claude, `rsync`, or automated polling on login nodes. Use login nodes only for short interactive actions such as SSH entry, `sbatch`, `scancel`, and occasional one-shot Slurm status checks.

## Common Pattern Behind Our Commands

Most commands we used follow the same structure:

1. Confirm the server project root.
2. Confirm the prepared dataset path and row count.
3. Choose a model preset or explicit model list.
4. Run generation into a response JSONL.
5. Run `experiment.py` over that JSONL into a report JSON.
6. Use the report JSON, not the response JSONL, as notebook input.
7. Save figures and tables under `figures/<report_name>/`.

The recurring distinction is important:

- `prepare_fever.py` creates CALE-ready dataset resources.
- `generate_responses.py` creates target-model responses.
- `experiment.py` creates evaluator metrics and report JSON files.
- `visualize_results.ipynb` or `visualize_fever_small_models.ipynb` consumes report JSON files.

If something fails, identify which stage failed first. Do not debug visualization by pointing it at a response JSONL, and do not debug evaluation before confirming generation produced the expected number of rows.

## Canonical Server Project Layout

Main server project directory used in current Galvani runs:

```text
/mnt/lustre/home/kelava/koh927/thesis/CALE
```

Some older notes or screenshots may also abbreviate this as:

```text
/thesis/CALE
```

Important subdirectories:

- `data/`: raw and prepared datasets
- `data/fever/prepared/dev_prepared.jsonl`: current primary FEVER dev resource
- `outputs/`: generated responses, evaluation reports, logs, and run artifacts
- `outputs/small_models_all/`: current small-model full-run output directory
- `outputs/small_models_all/logs/`: pipeline logs from the all-datasets runner
- `outputs/slurm/`: Slurm stdout/stderr logs for batch jobs
- `figures/`: exported plots or paper figures

Important scripts on the server:

- `prepare_fever.py`
- `generate_responses.py`
- `experiment.py`
- `download_fever_data.sh`
- `run_pipeline.sh`
- `run_small_models_all_datasets.sh`
- `visualize_results.ipynb`
- `visualize_fever_small_models.ipynb`

## Required Workflow

### 1. Enter The Server Project

```bash
cd /mnt/lustre/home/kelava/koh927/thesis/CALE
```

### 2. Check Whether Responses Already Exist

Before using A100 time, check whether the GPU-heavy response JSONL already
exists. Generation is the only A100-heavy stage; `experiment.py`,
`visualize_behavior_matrix.py`, and `analyze_behavior_matrix.py` are CPU-friendly.

The current reusable neutral full FEVER response file is:

```bash
ls -lh outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl
wc -l outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl
```

Expected rows:

```text
39996
```

This is `19998` FEVER dev items times two target models. If this file exists,
do not regenerate neutral full responses on A100; run `experiment.py` directly
on CPU and export the report plus behavior matrix.

After the behavior matrix exists, run target-specific robustness analysis on
CPU:

```bash
python run_target_specific_behavior_analysis.py \
  --input outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full_eval_behavior_matrix.csv \
  --output-dir figures/behavior_target_specific_neutral_full
```

This repeats the behavior profile and CALE-only PCA summaries for
`pooled_all_targets`, `target_qwen25_15b_only`, and `target_llama32_1b_only`.

### 3. Check Data Readiness

```bash
ls -lh data/fever/prepared/dev_prepared.jsonl
wc -l data/fever/prepared/dev_prepared.jsonl
```

Expected current FEVER dev rows:

```text
19998
```

If the prepared file is missing, run:

```bash
bash download_fever_data.sh
```

The download script downloads raw FEVER data and prepares train/dev files. It should not delete the raw FEVER files.

### 4. Run A Smoke Test Before Full Runs

Use smoke mode to verify model access, GPU visibility, output names, and JSON format:

```bash
CALE_RUN_MODE=smoke \
CALE_LIMIT=20 \
CALE_MODEL_PRESET=open_small \
CALE_BATCH_SIZE=8 \
CALE_RESUME=1 \
bash run_pipeline.sh
```

For quick Qwen-only debugging when Llama access is uncertain:

```bash
CALE_RUN_MODE=smoke \
CALE_LIMIT=20 \
CALE_MODEL_PRESET=qwen_only \
CALE_BATCH_SIZE=8 \
bash run_pipeline.sh
```

### 5. Submit Long Runs Through Slurm

Preferred full small-model run on A100:

```bash
sbatch submit_small_models_a100.sbatch
```

If launching manually inside an allocated job, the core command is:

```bash
CALE_RUN_MODE=full \
CALE_MODEL_PRESET=open_small \
CALE_BATCH_SIZE=64 \
CALE_RESUME=1 \
CALE_OUTPUT_DIR=outputs/small_models_all \
bash run_small_models_all_datasets.sh
```

Current `open_small` preset:

```text
Qwen/Qwen2.5-1.5B-Instruct
meta-llama/Llama-3.2-1B-Instruct
```

Expected FEVER dev output rows:

```text
19998 items x 2 models = 39996 response JSONL rows
```

### 6. Check Progress Without Polling Login Nodes

Do not leave `watch`, `tail -f`, repeated `squeue`, repeated `grep`, or custom polling loops running on a login node. Use occasional one-shot checks, or inspect logs from a compute allocation/Jupyter session that is already running on a compute node.

One-shot Slurm queue check:

```bash
squeue -u $USER
```

One-shot Slurm log peek:

```bash
tail -n 80 outputs/slurm/cale-small-<JOB_ID>.out
```

One-shot pipeline log peek:

```bash
tail -n 80 outputs/small_models_all/logs/fever_dev_neutral_full_eval.log
```

Check generation progress:

```bash
wc -l outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl
ls -lh outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl
```

Check GPU use:

```bash
nvidia-smi
```

Run `nvidia-smi` inside the allocated compute job or compute-node Jupyter session, not on a login node.

Healthy logs should show:

- `cuda_available=True`
- visible CUDA device name
- actual model device such as `cuda:0`
- batch size
- expected output rows
- elapsed time, ETA, and generation rate

### 7. Use The Correct Visualization Input

Use the report JSON:

```python
RESULTS_PATH = Path("outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full_eval_report.json")
```

Do not use:

```text
outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl
```

That file is the response JSONL from generation, not the final evaluation report.

For the current FEVER small-model result, use:

```text
visualize_fever_small_models.ipynb
```

It creates:

- model comparison figures
- evaluator-variant comparison figures
- model x evaluator heatmaps
- FEVER reference-label behavior plots
- NEI overclaim analysis
- Source Faithfulness and construct-subscore plots
- model-disagreement tables
- paper-table CSV exports

## Resource Policy And Practical Choices

Use Slurm for GPU work.

- A100 is best for full generation runs and larger batch sizes.
- 2080 Ti is best for smoke tests, small-model debugging, and short notebook sessions.
- CPU Jupyter is best for result inspection, notebook editing, and visualization that does not need GPU.

Do not rely on an SSH foreground process for long runs. If a command is running in the foreground and the laptop sleeps or disconnects, the job may stop. Use `sbatch` for long experiments.

If a foreground generation was interrupted, resume with:

```bash
CALE_RESUME=1 bash run_small_models_all_datasets.sh
```

## Jupyter Usage

Local launcher scripts live one directory above this code folder:

- `../start_galvani_jupyter.command`: opens one unified launcher menu
- `../stop_galvani_jupyter.command`: stops the saved Jupyter Slurm session

The unified launcher first asks which resource profile you want:

- `2080 visualization`: notebook inspection, quick debugging, and general plotting
- `2080 experiment`: notebook sessions that need to launch or monitor actual model runs on 2080
- `A100`: heavier notebook work or cases where 2080 VRAM is not enough
- `CPU`: visualization, table inspection, and notebook editing without GPU

It then asks for a walltime with A/B/C/D choices and recommended defaults.

Default resource bundles are:

`2080 visualization`

```text
1 GPU, 2 CPU, 12G memory, 2 hours
```

`2080 experiment`

```text
1 GPU, 8 CPU, 48G memory, 8 hours
```

`A100`

```text
1 GPU, 8 CPU, 64G memory, 4 hours
```

`CPU`

```text
0 GPU, 4 CPU, 16G memory, 2 hours
```

For scripting, you can still skip the menu by setting `GALVANI_PROFILE_KIND` and optionally `GALVANI_WALLTIME` before launching `start_galvani_jupyter.sh`.

To comply with the login-node policy, the launcher submits the Slurm job and exits. It does not poll the login node while waiting for Slurm or Jupyter readiness. After a manual one-shot `squeue` check shows the job is `RUNNING`, run:

```bash
../connect_galvani_jupyter.command
```

If the connector says the token is not ready yet, wait a little and run it once again. Do not wrap it in `watch` or another polling loop.

If Jupyter is pending with `Reason=Resources`, this is usually not a script bug. It means Slurm cannot currently satisfy the requested resource bundle. Cancel unneeded pending Jupyter jobs with:

```bash
scancel <JOB_ID>
```

## Hugging Face Access

Llama and Gemma models may require accepting the model license and setting a Hugging Face token.

Set the token on the server shell before submitting jobs:

```bash
export HF_TOKEN="your_token"
```

Do not paste tokens into chat or commit them to files. If a token was exposed, revoke or rotate it.

## Output Conventions

Typical full FEVER small-model output paths:

```text
outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl
outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full_eval_report.json
```

Typical output pattern:

- response generations: `outputs/**/*.jsonl`
- CALE evaluation reports: `outputs/**/*_report.json`
- Slurm logs: `outputs/slurm/*.out` and `outputs/slurm/*.err`
- pipeline logs: `outputs/small_models_all/logs/*.log`
- notebook figures: `figures/<report_stem>/`

## Decision Rules

Use this checklist before changing commands:

- If the dataset path ends in `prepared/*.jsonl` and lacks `candidate_response`, run `generate_responses.py` first.
- If the file is a response JSONL with `candidate_response`, run `experiment.py`.
- If the file is `*_report.json`, use it in visualization notebooks.
- If a run will take more than a few minutes, submit it with Slurm.
- If only making plots, prefer CPU Jupyter or a local notebook.
- If GPU memory is low, reduce `CALE_BATCH_SIZE`.
- If the process was interrupted, use `CALE_RESUME=1`.
- If Llama/Gemma download fails, check `HF_TOKEN` and model license access.

## Recommended Handoff Fields

When finishing a run, append or update the following information somewhere convenient in this file or in a dated lab note:

- date
- server name or login target
- dataset used
- model preset or exact model names
- key environment variables
- command launched
- Slurm job id, if applicable
- output directory
- response JSONL path
- final report JSON path
- figure directory
- whether the run completed successfully
- next recommended step

## Suggested Status Template

Copy this block when recording a new run:

```md
## Run Status

- Date:
- Server: Galvani
- Working directory on server: /mnt/lustre/home/kelava/koh927/thesis/CALE
- Dataset:
- Models / preset:
- Command:
- Slurm job id:
- Main outputs:
- Response JSONL:
- Latest report JSON:
- Figures exported:
- Current state: planned | running | finished | failed
- Notes:
- Next step:
```

## Current Known Successful Run

- Date: 2026-05-07
- Server: Galvani A100
- Dataset: FEVER dev, `data/fever/prepared/dev_prepared.jsonl`
- Rows: 19998
- Models / preset: `open_small`
- Models: `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`
- Framing: `neutral`
- Batch size: 64
- Response rows: 39996
- Response JSONL: `outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full.jsonl`
- Report JSON: `outputs/small_models_all/fever_dev_qwen25_15b_llama32_1b_neutral_full_eval_report.json`
- Recommended visualization: `visualize_fever_small_models.ipynb`

## Recommended One-Sentence Summary For Future Agents

> The CALE codebase is edited locally, but full experiments run on Galvani through Slurm; always verify server-side `outputs/`, use response JSONL only for `experiment.py`, and use `*_report.json` files for visualization notebooks.
