# Conda Environment Setup

## Step 1 — One-time channel configuration (per machine)

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

## Step 2 — Create and activate the environment

```bash
conda env create -f conda/environment.yml
conda activate foundry
```

This creates the `foundry` environment with Python 3.14. Then install the pip dependencies:

```bash
pip install -r conda/requirements.txt
```

This installs the core data science and utility packages (numpy, pandas, scikit-learn, matplotlib, etc.).

To update an existing environment after dependency changes:

```bash
conda env update -f conda/environment.yml --prune
pip install -r conda/requirements.txt
```

## Step 3 — Install PyTorch (separately, platform-dependent)

PyTorch is not included in `requirements.txt` because the correct version depends on your platform and CUDA version. Install it manually after activating the environment.

**Windows / Linux with CUDA 12.8:**
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

**CPU only:**
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

> `torchaudio` is not used in this project and does not need to be installed.
