# CALE Thesis Environment

This project has three dependency layers:

## 1. Minimal local experiment pipeline

These scripts use only the Python standard library:

- `cale_demo.py`
- `experiment.py`
- `perturbations.py`
- `prepare_fever.py`

If you only want to run the CALE heuristic pipeline, Python itself is enough.

## 2. Jupyter and visualization

These are needed to work comfortably in notebooks and open `visualize_results.ipynb`:

- `jupyterlab`
- `notebook`
- `ipykernel`
- `pandas`
- `matplotlib`

## 3. Optional model and API backends

These are only needed for specific scripts or modes:

- `openai`
  - required for `llm_judge.py` when using `--judge openai`
- `transformers`, `accelerate`
  - required for `generate_responses.py` when using Hugging Face models
- `torch`
  - required for `generate_responses.py` with local model generation
  - install this separately on the cluster so you can choose the CUDA build that matches the server

## Recommended setup on Galvani

Create the environment:

```bash
conda env create -f "CALE code/environment.yml"
conda activate jupyterenv
```

Register the notebook kernel:

```bash
python -m ipykernel install --user --name jupyterenv --display-name "Python (jupyterenv)"
```

Install PyTorch separately if you want local model generation:

```bash
python -m pip install torch
```

If you only need CALE experiments plus notebooks, you can skip PyTorch.

## FEVER data download

This repository includes a helper script that downloads the raw FEVER files and
optionally prepares CALE-ready JSONL files:

```bash
bash "CALE code/download_fever_data.sh"
```

That script downloads:

- `train.jsonl`
- `shared_task_dev.jsonl`
- `wiki-pages.zip`

into `data/fever/`, then writes prepared files into `data/fever/prepared/`.

If you only want the raw downloads, use:

```bash
bash "CALE code/download_fever_data.sh" --download-only
```
