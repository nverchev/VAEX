# VAEX

Implementation of [**Hierarchical Variational Autoencoders for Visual Counterfactuals**](https://ieeexplore.ieee.org/document/9506692), ICIP 2021.

This project provides a modular, type-safe implementation of VAEX using the [drytorch](https://github.com/nverchev/drytorch) engine for ML experimentation.
---

## Overview

Visual counterfactual explanations answer the question: *"How would this image change if its attribute classification were different?"*

This repository implements the **VAEX** model framework:

- **Hierarchical Bottom-Up / Top-Down VAE (`BVAE`)** — a multi-scale VAE featuring resolution-aware residual layers, adaptive instance normalization (ADAIN), and free-bits Gaussian KL divergence.
- **Conditional Attribute Manipulation (`VAEX`)** — explicit conditioning on input attribute maps ($fc_0$) and top-latent attribute maps ($fc_1$), enabling controllable generation across scaling values $s \in [0, 1]$.
- **Decoupled Loss Objectives** — Gaussian Negative Log-Likelihood with running per-pixel variance tracking (`train_var`), free-bits KLD, and prior conditioning.
- **Visual Counterfactual Generation** — automated generation and rendering of counterfactual image grids across target classes (e.g., `Female` vs `Male`) and scaling steps.
- **Anomaly Detection** — evaluation of VAE reconstruction loss on classifier misclassifications versus correctly classified samples.

---

## Dataset — CelebA

The CelebA dataset contains over 200,000 celebrity images, each with 40 attribute annotations.

### Download

The dataset directory is expected at `<DATASET_DIR>/celeba_torchvision/`. The project will automatically attempt to download the dataset via `torchvision.datasets.CelebA`.

If the automatic Google Drive download fails due to a quota error, download the files manually:

| File | Google Drive link | Destination |
|------|-------------------|-------------|
| `img_align_celeba.zip` | [download](https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) | extract to `<DATASET_DIR>/celeba_torchvision/img_align_celeba/` |
| `list_attr_celeba.txt` | [download](https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U) | place at `<DATASET_DIR>/celeba_torchvision/list_attr_celeba.txt` |

### Configure the data path

Set the environment variable `DATASET_DIR` in a `.env` file at the project root:

```dotenv
DATASET_DIR=/path/to/your/datasets
ROOT_EXP_DIR=./experiments
METADATA_DIR=./dataset_metadata
```

Alternatively, pass `data.dataset.data_dir` CLI overrides to customize the dataset path for specific runs:
```bash
uv run train_vae.py data.dataset.data_dir=/path/to/custom/dataset
```

---

## Installation

The project requires **Python 3.13** and uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install core dependencies
uv sync

# Optional extras
uv sync --extra track_all   # TensorBoard, W&B, SQLAlchemy
uv sync --extra finetune    # Optuna hyperparameter tuning
```

---

## Usage

### Training

- **Train the CelebA attribute classifier:**
  ```bash
  uv run train_classifier.py
  ```

- **Train the Hierarchical VAEX model:**
  ```bash
  uv run train_vae.py
  ```

*Note: Hyperparameter tuning entry points are available via `tune_classifier.py` and `tune_vae.py`.*

### Visualisation and Evaluation

- **Generate visual counterfactuals for test samples:**
  ```bash
  uv run generate_counterfactuals.py
  ```

- **Reconstruct dataset samples:**
  ```bash
  uv run reconstruct_samples.py
  ```

- **Evaluate quantitative metrics and anomaly detection:**
  ```bash
  uv run evaluate_counterfactuals.py
  ```

- **Generate random samples:**
  ```bash
  uv run generate_random_samples.py
  ```

- **Inspect dataset statistics and model parameters:**
  ```bash
  uv run explore_dataset.py
  uv run profile_model.py
  ```

---

## Project structure

```
VAEX/
├── configs/                      # Hydra YAML configuration hierarchy
│   ├── experiment/               # Subexperiments (classification, autoencoder, data, user)
│   └── tuning/                   # Optuna search space definitions
├── reference/                    # Original reference implementation (VAEX.ipynb)
├── src/
│   ├── config/                   # Pydantic specs, options, and drytorch experiment wiring
│   ├── data/                     # CelebA dataset loading, structures
│   ├── module/                   # Architectures (CelebaClassifier, BVAE, VAEX) and VAE layers
│   ├── train/                    # Decoupled loss functions, metrics, learning schemas
│   └── utils/                    # Utilities
├── train_classifier.py           # Classifier training entrypoint
├── train_vae.py                  # VAE / VAEX training entrypoint
├── reconstruct_samples.py        # Sample reconstruction entrypoint
├── generate_counterfactuals.py   # Visual counterfactual grid generator
├── evaluate_counterfactuals.py   # Quantitative evaluation & anomaly detection
└── pyproject.toml
```

---

## Citation

```bibtex
@inproceedings{9506692,
  author       = {{Vercheval, Nicolas and Pizurica, Aleksandra}},
  booktitle    = {{2021 IEEE International Conference on Image Processing (ICIP)}},
  title        = {{Hierarchical Variational Autoencoders for Visual Counterfactuals}},
  year         = {{2021}},
  pages        = {{3498-3502}},
  doi          = {{10.1109/ICIP42928.2021.9506692}}
}
```
