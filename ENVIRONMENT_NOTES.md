# CALE Environment Notes

This project has three dependency layers.

## 1. Minimal Local Experiment Pipeline

The heuristic evaluator path mainly uses the Python standard library:

- `cale/cale_demo.py`
- `cale/experiment.py`
- `cale/perturbations.py`
- `examples/prepare_fever.py`

For small heuristic-only checks, Python itself is usually enough.

## 2. Notebooks and Analysis

These packages are useful for the notebooks and analysis scripts:

- `jupyterlab`
- `notebook`
- `ipykernel`
- `pandas`
- `numpy`
- `matplotlib`

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate jupyterenv
```

Register the notebook kernel:

```bash
python -m ipykernel install --user --name jupyterenv --display-name "Python (jupyterenv)"
```

## 3. Optional Model and API Backends

These packages are only needed for specific scripts or modes:

- `openai`: required for `cale/llm_judge.py` when using `--judge openai`
- `transformers` and `accelerate`: required for Hugging Face generation or
  Hugging Face evaluator backends
- `torch`: required for local model generation; install this separately so the
  build matches your CPU, CUDA, or MPS environment

Example CUDA install:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If you only need existing behavior matrices, notebooks, and heuristic analysis,
you can skip the PyTorch install.

## FEVER Data Download

Download and prepare FEVER resources:

```bash
bash workflows/download_fever_data.sh
```

The script downloads:

- `train.jsonl`
- `shared_task_dev.jsonl`
- `wiki-pages.zip`

into `data/fever/`, then writes prepared files into `data/fever/prepared/`.

If you only want the raw downloads, use:

```bash
bash workflows/download_fever_data.sh --download-only
```
