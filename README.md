# VAEX

Implementation of [**Hierarchical Variational Autoencoders for Visual Counterfactuals**](https://arxiv.org/abs/2104.14869).

This project provides a modular, type-safe implementation of VAEX using the [drytorch](https://github.com/nverchev/drytorch) engine for ML experimentation.

---

## Overview

Visual counterfactual explanations answer the question: *"How would this image change if its attribute classification were different?"*

This repository implements the **VAEX** model framework:
- **Hierarchical Bottom-Up / Top-Down VAE (`BVAE`)**: A multi-scale VAE featuring resolution-aware residual layers, adaptive instance normalization (ADAIN), and free-bits Gaussian KL divergence.
- **Conditional Attribute Manipulation (`VAEX`)**: Explicit conditioning on input attribute maps ($fc_0$) and top-latent attribute maps ($fc_1$), enabling controllable generation across scaling values $s \in [0, 1]$.
- **Decoupled Loss Objectives**: Gaussian Negative Log-Likelihood with running per-pixel variance tracking (`train_var`), free-bits KLD, and prior conditioning.
- **Visual Counterfactual Generation**: Automated generation and rendering of counterfactual image grids across target classes (e.g. `Female` vs `Male`) and scaling steps.
- **Anomaly Detection**: Evaluation of VAE reconstruction loss on classifier misclassifications versus correctly classified samples.

---

## CelebA Dataset Setup

### Environment Configuration

Set your root dataset directory `DATASET_DIR` in `.env`:

```ini
DATASET_DIR=/path/to/your/datasets
ROOT_EXP_DIR=./experiments
METADATA_DIR=./dataset_metadata
```

The dataset directory is expected at `<DATASET_DIR>/celeba_torchvision/`. If it does not exist, the project
automatically creates it and attempts to download the dataset via `torchvision.datasets.CelebA(download=True)`.

### Expected Directory Layout

```
<DATASET_DIR>/celeba_torchvision/
├── img_align_celeba/       # CelebA aligned facial images (000001.jpg, ...)
└── list_attr_celeba.txt    # Facial attribute labels
```

### Manual Download

The automatic download uses Google Drive, which may fail with a quota error:

```
Too many users have viewed or downloaded this file recently.
```

In that case, download the following files directly from Google Drive and place them as shown:

| File | Google Drive link | Destination |
|------|-------------------|-------------|
| `img_align_celeba.zip` | [download](https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) | extract to `<DATASET_DIR>/celeba_torchvision/img_align_celeba/` |
| `list_attr_celeba.txt` | [download](https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U) | place at `<DATASET_DIR>/celeba_torchvision/list_attr_celeba.txt` |

### CLI Overrides

Pass `data.dataset.data_dir` CLI overrides to customize the dataset path for specific runs:

```bash
uv run train_vae.py data.dataset.data_dir=/path/to/custom/dataset
```

---

## Installation

The project requires **Python 3.13** and uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install core dependencies
uv sync

# Optional extras for tracking and tuning
uv sync --extra track_all   # TensorBoard, W&B, SQLAlchemy
uv sync --extra finetune    # Optuna hyperparameter tuning
```

---

## Usage Commands

### 1. Classifier Training & Tuning

Train the 5-stage convolutional CelebA attribute classifier:

```bash
uv run train_classifier.py
```

Hyperparameter tuning with Optuna:

```bash
uv run tune_classifier.py
```

### 2. VAE / VAEX Model Training & Tuning

Train the Hierarchical VAEX model:

```bash
uv run train_vae.py
```

Hyperparameter tuning for VAE architecture:

```bash
uv run tune_vae.py
```

### 3. Sample Reconstruction & Visualization

Reconstruct selected dataset samples using the VAE / VAEX model:

```bash
uv run reconstruct_samples.py
```

### 4. Visual Counterfactual Generation

Generate visual counterfactual image grids for test samples:

```bash
uv run generate_counterfactuals.py
```

Generated image panels are saved under `outputs/v0.1.0/images/`.

### 5. Quantitative Evaluation & Anomaly Detection

Evaluate classifier accuracy, counterfactual metrics, and VAE reconstruction loss on misclassifications:

```bash
uv run evaluate_counterfactuals.py
```

### 6. Dataset & Model Inspection

Inspect dataset statistics and attributes:

```bash
uv run explore_dataset.py
```

Profile model parameters and FLOPs:

```bash
uv run profile_model.py
```

---

## Architecture & Project Structure

```
VAEX/
├── configs/                # Hydra YAML configuration hierarchy
│   ├── experiment/         # Subexperiments (classification, autoencoder, data, user)
│   └── tuning/             # Optuna search space definitions
├── reference/              # Original reference implementation (VAEX.ipynb)
├── src/
│   ├── config/             # Pydantic specs, options, and drytorch experiment wiring
│   ├── data/               # CelebA dataset loading, structures (Inputs, Outputs, Targets)
│   ├── module/             # Architectures (CelebaClassifier, BVAE, VAEX) and VAE layers
│   ├── train/              # Decoupled loss functions, metrics, learning schemas, and model wrappers
│   └── utils/              # Parallel worker spawning and profiling utilities
├── train_classifier.py     # Classifier training entrypoint
├── train_vae.py            # VAE / VAEX training entrypoint
├── reconstruct_samples.py  # Sample reconstruction entrypoint
├── generate_counterfactuals.py # Visual counterfactual grid generator
├── evaluate_counterfactuals.py # Quantitative evaluation & anomaly detection
└── explore_dataset.py      # Dataset inspection tool
```
