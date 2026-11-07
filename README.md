# VAEX for hierarchical counterfactual explanations

Implementation of [**Hierarchical Variational Autoencoders for Visual Counterfactuals**](https://ieeexplore.ieee.org/document/9506692), ICIP 2021.

This version recreates the original code in a modern style using [DRYTorch](https://github.com/nverchev/drytorch). The [results](#results) compare the new model with the article.

---

## Overview

Visual counterfactual explanations answer the question: *"How would this image change if its attribute classification were different?"*

### What is VAEX?

**VAEX** (**V**ariational **A**utoencoder **EX**planations) is a framework for generating visual counterfactual explanations for pre-trained black-box classifiers. Rather than modifying images using gradient-based adversarial perturbations (which produce unnatural high-frequency artifacts) or unconditional generative models (which struggle to preserve subject identity), VAEX integrates the classifier's predicted probabilities directly into the generative process of a Variational Autoencoder.

By conditioning the autoencoder on the classifier's predictions, VAEX learns the image distribution relative to the decision boundaries. Setting this condition to a counterfactual target class produces semantically plausible image edits that flip the classifier's decision while preserving the subject's identity.

### Application: Hierarchical VAE on CelebA (`Female` / `Male`)

The framework is implemented as a deep **Hierarchical VAE (HVAE)** with multi-scale residual stages connected through bidirectional inference and generative paths. Applied to facial attribute classification on **CelebA**, it benchmarks on the binary **`Male`** attribute (`Female` vs. `Male`).

### Key Contributions

- **Posterior Relaxation**: Resolves the tension between identity preservation and counterfactual transformation. High-level semantics are adjusted at the top latent, while the posterior shift of the lower latent `level` stages below the top is weighted by `relaxation ** (6 - level)`: the coarse latents follow the condition and the finest keep the texture of the image, allowing a smooth transition from reconstruction to prior sampling.
- **Free-Bits KL Divergence**: Reserves a minimum information capacity per latent coordinate to prevent inactive stages and posterior collapse across the hierarchy. Each stage has its own free bits (`autoencoder.objective.freebits`, from the top latent down), and the KL divergence of each stage is logged as `KLD_stage_<i>`. `Active_stage_<i>` logs the fraction of the coordinates of each stage whose KL divergence exceeds 0.01 nats; it falls to zero when a stage collapses.
- **Prior Conditioning**: The condition is the softmax of the classifier's log-probabilities divided by a temperature (`autoencoder.model.temperature`). A linear layer maps it to the prior mean and log-variance of the conditioned channels of the top latent (`autoencoder.model.z_cond_dim`); the other channels, the free ones, have a learned prior that ignores the condition. Every posterior is a shift from its prior.
- **Information Eraser**: An adversary predicts the probabilities of the classifier from the posterior of the free channels of the top latent. A gradient reversal (`autoencoder.objective.reversal_strength`) makes the free channels hide the class, so that a counterfactual changes the class through the conditioned channels only. The adversarial loss (`Adv`) rises to full weight over `autoencoder.train.adv_warmup_epochs`.
- **Cyclical KL Annealing**: The weight of the KL divergence rises along a cosine from 0.01 to 1 in each of `autoencoder.train.annealing_cycles` cycles over the first `autoencoder.train.annealing_period` epochs, then stays at 1.
- **Gaussian NLL**: Scores the reconstruction with a fixed variance (`autoencoder.objective.sigma_2`), which sets the balance with the KL divergence.
- **Sampling**: In evaluation mode, the standard deviation of each latent is multiplied by its factor in `autoencoder.model.sampling_std`, from the top latent down. Counterfactuals mix the classifier's probabilities with the target class, weighted by `evaluation.strength`.

---

## Dataset — CelebA

The CelebA dataset contains over 200,000 celebrity images, each with 40 attribute annotations.

### Download

The dataset directory is expected at `<DATA_DIR>/celeba_torchvision/`. The project will automatically attempt to download the dataset via `torchvision.datasets.CelebA`.

If the automatic Google Drive download fails due to a quota error, download the files manually:

| File | Google Drive link | Destination |
|------|-------------------|-------------|
| `img_align_celeba.zip` | [download](https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) | extract to `<DATA_DIR>/celeba_torchvision/img_align_celeba/` |
| `list_attr_celeba.txt` | [download](https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U) | place at `<DATA_DIR>/celeba_torchvision/list_attr_celeba.txt` |

### Configure the data path

Set the paths in a `.env` file at the project root:

```dotenv
DATA_DIR=/path/to/your/datasets
ROOT_EXP_DIR=./experiments
```

An environment variable takes precedence over the `.env` file, for example for a single run:
```bash
DATA_DIR=/path/to/custom/datasets uv run train_vae.py
```

---

## Project structure

```
VAEX/
├── configs/                      # Hydra YAML configuration hierarchy
│   └── experiment/               # Phases (classification, autoencoder), data and user settings
├── src/
│   ├── config/                   # Pydantic specs, options, and DRYTorch experiment wiring
│   ├── data/                     # CelebA dataset loading, structures
│   ├── module/                   # Architectures (CelebaClassifier, VAEX) and VAE layers
│   ├── trackers/                 # Trackers of the images the scripts publish, and of the learning curves
│   ├── lib/                      # DRYTorch components: losses, metrics, learning schemas, models
│   └── utils/                    # Counterfactual sweeps, FID, figures, tuning, distributed training
├── tests/                        # Unit tests
├── train_classifier.py           # Classifier training entrypoint
├── train_vae.py                  # VAE / VAEX training entrypoint
├── reconstruct_samples.py        # Sample reconstruction entrypoint
├── generate_counterfactuals.py   # Visual counterfactual grid generator
├── generate_misclassified_counterfactuals.py  # The same grid, for samples the classifier gets wrong
├── evaluate_counterfactuals.py   # Reconstruction metrics, counterfactual success and FID
├── generate_random_samples.py    # Random samples from the priors
├── explore_dataset.py            # Samples of the preprocessed dataset
├── plot_confidence.py            # Distribution of the probability of the predicted class
├── profile_vae.py                # Profiling of the VAE
├── tune_vae.py                   # Tuning of the VAE
├── plot_tuning_results.py        # Plots of a study
└── pyproject.toml
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
uv sync --all-extras --dev  # everything, with the development tools
```

`uv sync` installs the default PyTorch build. To match your machine, check your CUDA version with `nvcc --version`, then add the matching index to `~/.config/uv/uv.toml` before syncing. Replace `cu130` with your version, for example `cu126` for CUDA 12.6, or with `cpu` if you have no GPU:

```toml
index-strategy = "unsafe-best-match"

[[index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu130"
```

---

## Usage

### Training

- **Train the CelebA attribute classifier and test its accuracy:**
  ```bash
  uv run train_classifier.py
  uv run evaluate_classifier.py
  ```

- **Train the Hierarchical VAEX model:**
  ```bash
  uv run train_vae.py
  uv run train_vae.py autoencoder.model.architecture.inference_block=mobile   # VAEXcat
  ```

  Run every script with the same overrides, apart from `user.*`: the VAE finds the
  classifier through the name of the run, which the overrides form.

- **Reuse a trained classifier:** to train the VAE with new overrides, clone the experiment
  of the classifier under the name the overrides form, instead of training it again:
  ```bash
  uv run clone_experiment.py autoencoder.train.n_epochs=100
  uv run train_vae.py autoencoder.train.n_epochs=100
  ```
  `--from NAME` clones another experiment than the current variation. Clone one where only
  the classifier was trained: a VAE trained there would share its run.

### Visualisation and Evaluation

- **Generate visual counterfactuals for test samples, or for samples the classifier gets
  wrong:**
  ```bash
  uv run generate_counterfactuals.py
  uv run generate_misclassified_counterfactuals.py
  ```

- **Reconstruct dataset samples:**
  ```bash
  uv run reconstruct_samples.py
  ```

- **Evaluate the reconstructions (NLL, D_KL, MSE), the classifier on them, the percentage of
  successful counterfactuals for each relaxation $r$, also of the misclassified images turned
  towards their label, and the FID of reconstructions and counterfactuals:**
  ```bash
  uv run evaluate_counterfactuals.py
  ```

- **Generate random samples:**
  ```bash
  uv run generate_random_samples.py
  ```

- **Plot the distribution of the probability of the predicted class**, the condition of VAEX;
  `user.plot.use_train=true` uses the training images:
  ```bash
  uv run plot_confidence.py final=true
  ```

- **Inspect dataset statistics and model parameters:**
  ```bash
  uv run explore_dataset.py
  uv run profile_vae.py
  ```

The scripts that draw images publish them as events: the `matplotlib` tracker saves them
under the `images` folder of the run, and TensorBoard and W&B show them. Each sample has its
own folder, which every script adds to:

```
test/sample_5/original.png
test/sample_5/reconstruction.png
test/sample_5/difference.png       # between the original and the reconstruction
test/sample_5/counterfactuals.png  # a row per class, a column per relaxation
```

After training, the tracker also saves the learning curves under `plots`. Set
`user.trackers.matplotlib=false` to skip them, or `user.plot.show=true` to also display them.
Running a script under a debugger skips every tracker, so it writes no files.

Training keeps an exponential moving average of the weights of the VAE
(`autoencoder.model.ema_decay`), which the scripts evaluate. The trials of `tune_vae.py` train without
the average.

### Tuning

`tune_vae.py` searches the settings of the VAE with Optuna. Each trial trains for `n_epochs` (10) and scores the validation loss, so train the classifier first with the same overrides. The search spaces are in `configs/tuning/tune`, chosen with `tune=`:

```bash
uv run tune_vae.py                         # learning rate, weight decay, batch size
uv run tune_vae.py tune=architecture       # hidden dimensions, depths, activation, batch normalization
uv run plot_tuning_results.py tune=learn   # plots of a study
```

Studies are stored in `optimization_db/` and continue when run again. `sigma_2` is not tuned, because it changes the scale of the loss that compares the trials.

### Final pipeline

The final models train on training and validation and are evaluated on the test partition,
as in the [results](#results):

```bash
uv run train_classifier.py         final=true
uv run train_vae.py                final=true
uv run evaluate_classifier.py      final=true
uv run evaluate_counterfactuals.py final=true
uv run generate_counterfactuals.py final=true
```

---

## Differences from the original implementation

The model, the loss and the counterfactuals compute what the original implementation
computes. Technical differences:

- **Training loop**: DRYTorch runs the loops, checkpoints and logs; the validation runs
  every epoch, as in the original implementation.
- **Images**: they are preprocessed once and cached (center crop 148, bicubic resize to 129),
  instead of on every load.
- **Weight normalization**: the current PyTorch parametrization replaces the deprecated
  `torch.nn.utils.weight_norm`; the two compute the same weights.
- **FID**: as in the article, the reconstructions and the counterfactuals (towards the class the
  classifier does not predict) of 2048 images are compared with 2048 other images of the same
  partition (`evaluation.fid_samples`). The counterfactuals use $r = 0.6$
  (`evaluation.fid_relaxation`); the article used $r = 0$, so set
  `evaluation.fid_relaxation=0` to compare with it. The distance is the one of
  `torchmetrics`, which uses the Inception weights of the original FID.
- **Not ported**: the other conditioning variants of the original implementation (`C`, `X`,
  `Prior`, `CPrior`) and the anomaly detection.

## Results

| | this version | article |
|---|---|---|
| classifier accuracy | 97.7 % | |
| NLL | −102 575 | |
| $D_{KL}$ | 725.9 | |
| MSE | $9.81 \times 10^{-4}$ | |
| FID, reconstructions | 53.0 | 48.1 |
| FID, counterfactuals ($r = 0$) | 63.8 | 66.0 |

Successful counterfactuals, i.e. counterfactuals the classifier assigns to the target class,
for each relaxation $r$:

| $r$ | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|---|
| this version | 99.3 % | 98.8 % | 96.9 % | 89.1 % | 60.7 % | 14.4 % |
| article | 100 % | 99.8 % | 99.4 % | 96.1 % | 71.4 % | 13.9 % |

The model trained for the article has a sharper trade-off between sample quality and
counterfactual success: its counterfactuals stay successful up to $r = 0.8$ and then fail
abruptly, while the success of this version decreases more gradually.
This version reconstructs slightly less sharply (FID 53.0 against 48.1), but its
counterfactuals look more realistic (FID 63.8 against 66.0), and overall it offers a better
trade-off between image quality and counterfactual success from a visual point of view.

## Citation

```bibtex
@inproceedings{9506692,
  author       = {Vercheval, Nicolas and Pizurica, Aleksandra},
  booktitle    = {2021 IEEE International Conference on Image Processing (ICIP)},
  title        = {Hierarchical Variational Autoencoders for Visual Counterfactuals},
  year         = {2021},
  pages        = {3498-3502},
  doi          = {10.1109/ICIP42928.2021.9506692}
}
```
