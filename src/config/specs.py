"""Specification for the configuration files.

Root config hierarchy::

    AllConfig
    ├── classification: ExperimentConfig
    │   ├── name
    │   ├── model: ModelConfig
    │   │   └── architecture: LinearArchitectureConfig
    │   └── train: TrainingConfig
    │       ├── learn: LearningConfig
    │       │   └── scheduler: SchedulerConfig
    │       └── early_stopping: EarlyStoppingConfig
    ├── data: DataConfig
    └── user: UserSettings
        ├── trackers: TrackerList
        ├── path: PathSpecs          ← populated from .env
        └── hydra: HydraSettings     ← populated at runtime

For projects with multiple subexperiments, add one ExperimentConfig per subexperiment directly to AllConfig.
"""

import dataclasses
import pathlib
from functools import cached_property

from typing import Any, Annotated, Self

import torch

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from src.config.environment import EnvSettings, VERSION, get_current_branch
from src.config.torch import get_activation_cls, get_norm_cls, get_optim_cls, set_seed, DEFAULT_ACT, DEFAULT_NORM
from src.config.options import (
    GradOp,
    ClipCriterion,
    Schedulers,
    Models,
    Datasets,
    Conditioning,
    InferenceBlock,
)


PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]


@dataclass(kw_only=True)
class BaseArchitectureConfig:
    """Base specification for an architecture with activation and normalisation.

    Attributes:
        norm_name: The name of the normalisation layer class (e.g. 'BatchNorm2d').
            If empty, defaults to no normalisation or a standard default.
        act_name: The name of the activation function class (e.g. 'ReLU', 'SiLU').
            If empty, defaults to a standard activation (e.g. SiLU).
    """

    norm_name: str = ''
    act_name: str = ''

    def __post_init__(self) -> None:
        """Resolve activation and normalisation classes from their string names."""
        self.act_cls = get_activation_cls(self.act_name) if self.act_name else DEFAULT_ACT
        self.norm_cls = get_norm_cls(self.norm_name) if self.norm_name else DEFAULT_NORM
        return


@dataclass(kw_only=True)
class LinearArchitectureConfig(BaseArchitectureConfig):
    """Feedforward (linear) architecture parameters.

    Attributes:
        hidden_dims: A tuple defining the number of neurons in each hidden layer.
            For example, (256, 128) creates two hidden layers.
        dropout_rate: The probability of an element to be zeroed out during training.
            Must be between 0.0 and 1.0.
    """

    hidden_dims: tuple[StrictlyPositiveInt, ...] = (256, 128)
    dropout_rate: PositiveFloat = 0.0


@dataclass(kw_only=True)
class VAEArchitectureConfig(BaseArchitectureConfig):
    """Hierarchical VAE architecture parameters.

    Attributes:
        n_resolutions: The number of distinct spatial resolutions in the hierarchy.
        h_dims: The number of feature channels at each resolution stage.
    """

    n_resolutions: StrictlyPositiveInt = 5
    h_dims: tuple[StrictlyPositiveInt, ...] = (16, 32, 64, 128, 256, 512)
    k_dims: tuple[StrictlyPositiveInt, ...] = (3, 3, 3, 3, 3)
    strides: tuple[StrictlyPositiveInt, ...] = (2, 2, 2, 2, 2)
    paddings: tuple[PositiveInt, ...] = (1, 1, 1, 1, 1)
    depths: tuple[PositiveInt, ...] = (1, 1, 1, 1, 1)
    conditioning: Conditioning = Conditioning.ExplicitAndLearned
    inference_block: InferenceBlock = InferenceBlock.ADAIN


@dataclass(kw_only=True)
class ObjectiveConfig:
    """Specification for Autoencoder / VAE loss objective and metrics parameters.

    Attributes:
        sigma_2: Variance of the Gaussian NLL reconstruction loss.
        posterior_weight_multiplier: Ratio between the posterior weights of consecutive latent stages.
        cf_strength: Interpolation weight from original condition (0.0) to one-hot target class (1.0).
        sampling_variance: Per-stage multiplier on the sampling standard deviation at inference.
        freebits: Free-bits floor per latent stage.
        recalibrate_bn: Whether to re-estimate BatchNorm running statistics after training.
        reversal_strength: Multiplier for the gradient reversed in the InformationEraser layer.
    """

    sigma_2: float = 1.0
    posterior_weight_multiplier: float = 0.5
    cf_strength: float = 1.0
    sampling_variance: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    freebits: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0)
    recalibrate_bn: bool = True
    reversal_strength: float = 0.5



@dataclass
class ModelConfig:
    """Model class selector and non-architectural settings.

    Attributes:
        name: Model architecture variant to instantiate.
        architecture: VAEArchitectureConfig
        z_dims: Per-stage latent dimensions; must match objective.freebits in length.
        z_cond_dim: Conditioned channels per latent stage; the rest of the stage is free.
        ema_decay: Exponential moving average decay rate for model weights.
        temperature: Temperature scaling factor for logits in conditional models (e.g. VAEX).
    """

    name: Models = Models.VAEX
    architecture: VAEArchitectureConfig = dataclasses.field(default_factory=VAEArchitectureConfig)
    z_dims: tuple[StrictlyPositiveInt, ...] = (1, 2, 4, 8, 16, 32)
    z_cond_dim: PositiveInt = 1
    ema_decay: float = 0.999
    temperature: float = 5.0

    @model_validator(mode='after')
    def _check_root_keeps_a_free_part(self) -> Self:
        if self.z_cond_dim >= self.z_dims[-1]:
            msg = 'z_cond_dim ({}) must be smaller than the root z_dims[-1] ({}).'
            raise ValueError(msg.format(self.z_cond_dim, self.z_dims[-1]))

        return self


@dataclass
class DatasetConfig:
    """Dataset identity and paths.

    Attributes:
        name: Name of the dataset to use.
        target_attr: The specific attribute to use for conditional learning.
        classes: The names of the classes/unique values expected for the target attribute.
        n_train_samples: Expected number of training samples.
        n_val_samples: Expected number of validation samples.
        n_test_samples: Expected number of test samples.
    """

    name: Datasets = Datasets.CelebaMale
    target_attr: str = 'Male'
    classes: list[str] = dataclasses.field(default_factory=lambda: ['Female', 'Male'])
    n_train_samples: StrictlyPositiveInt = 162770
    n_val_samples: StrictlyPositiveInt = 19867
    n_test_samples: StrictlyPositiveInt = 19962


@dataclass
class DataConfig:
    """Full data configuration.

    Attributes:
        dataset: The dataset-specific configuration and splits.
        x_dim: Spatial dimension of the input images (e.g., 129 for 129x129).
        crop_size: Center crop size before resizing to x_dim.
    """

    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    x_dim: StrictlyPositiveInt = 129
    crop_size: StrictlyPositiveInt = 148
    n_channels: StrictlyPositiveInt = 3


@dataclass
class SchedulerConfig:
    """Specification for the learning-rate scheduler.

    Attributes:
        function: Which scheduler to use (see :class:`~src.config.options.Schedulers`).
        restart_interval: Epochs between cosine restarts (0 = no restart).
        restart_fraction: Fraction of base LR applied at each restart.
        warmup_steps: Linear warm-up epochs at the start of training.
        settings: Additional keyword arguments forwarded to the scheduler
            constructor (e.g. ``decay_steps``, ``min_decay``).
    """

    function: Schedulers = Schedulers.Cosine
    restart_interval: PositiveInt = 0
    restart_fraction: PositiveFloat = 1.0
    warmup_steps: PositiveInt = 0
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class LearningConfig:
    """Specification for the optimiser and learning schedule.

    Attributes:
        optimizer_name: Name of a PyTorch optimiser, e.g. ``'AdamW'``.
        learning_rate: Global learning rate (or per-parameter group base LR).
        scheduler: Learning-rate scheduler configuration.
        grad_op: Optional gradient pre-processing operation applied before
            each optimiser step (see :class:`~src.config.options.GradOp`).
        clip_criterion: Criterion for histogram-based clippers.
        opt_settings: Extra keyword arguments forwarded to the optimiser
            constructor (e.g. ``weight_decay``).
    """

    optimizer_name: str = 'AdamW'
    learning_rate: PositiveFloat = 1e-3
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    grad_op: GradOp | None = None
    clip_criterion: ClipCriterion | None = ClipCriterion.EMA
    opt_settings: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve optimizer class from name."""
        self.optimizer_cls = get_optim_cls(self.optimizer_name)
        return

    @model_validator(mode='after')
    def _check_clip_criterion(self) -> Self:
        if self.clip_criterion is None and self.grad_op in (GradOp.HistClipper, GradOp.ParamHistClipper):
            msg = 'Must specify clip_criterion when using HistClipper or ParamHistClipper.'
            raise ValueError(msg)

        return self


@dataclass
class EarlyStoppingConfig:
    """Specification for early stopping.

    Attributes:
        active: Whether to enable early stopping.
        window: Number of recent validation metrics to average.
        patience: Epochs to wait for improvement before stopping.
    """

    active: bool = False
    window: int = 1
    patience: int = 10


@dataclass
class TrainingConfig:
    """Specification for the training loop.

    Attributes:
        batch_size: Total batch size across all devices.
        learn: Optimiser and scheduler configuration.
        n_epochs: Maximum number of training epochs.
        early_stopping: Early stopping configuration.
        annealing_period: Epochs over which the KL weight is annealed. Past it the weight
            stays at one.
        annealing_cycles: Number of times the KL weight ramps back to one within that
            period, so that reconstruction pressure returns periodically instead of once.
    """

    batch_size: StrictlyPositiveInt = 32
    learn: LearningConfig = dataclasses.field(default_factory=LearningConfig)
    n_epochs: StrictlyPositiveInt = 10
    early_stopping: EarlyStoppingConfig = dataclasses.field(default_factory=EarlyStoppingConfig)
    _n_subprocesses: PositiveInt = 0
    annealing_period: int = 90
    annealing_cycles: StrictlyPositiveInt = 2
    adv_warmup_epochs: int = 3

    @model_validator(mode='after')
    def _check_batch_divisibility(self) -> Self:
        if self._n_subprocesses and self.batch_size % self._n_subprocesses != 0:
            msg = 'Global batch size {} not divisible by number of devices {}.'
            raise ValueError(msg.format(self.batch_size, self._n_subprocesses))

        return self

    @property
    def batch_size_per_device(self) -> int:
        """Batch size assigned to each individual device."""
        if self._n_subprocesses == 0:
            return self.batch_size

        return self.batch_size // self._n_subprocesses


@dataclass
class BaseExperimentConfig:
    """Base specification for an experiment task.

    Attributes:
        name: Subexperiment identifier.
        train: Training loop configuration.
    """

    name: str
    train: TrainingConfig


@dataclass
class ClassificationExperimentConfig(BaseExperimentConfig):
    """Specification for a classification experiment task.

    Attributes:
        model: Classifier model architecture configuration.
    """

    model: LinearArchitectureConfig = dataclasses.field(default_factory=LinearArchitectureConfig)


@dataclass
class AutoencoderExperimentConfig(BaseExperimentConfig):
    """Specification for an autoencoder / VAE experiment task.

    Attributes:
        model: VAE model configuration.
        objective: VAE loss objective and metrics configuration.
    """

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    objective: ObjectiveConfig = dataclasses.field(default_factory=ObjectiveConfig)

    @model_validator(mode='after')
    def _check_objective_tuples_match_z_dims(self) -> Self:
        if len(self.objective.sampling_variance) != len(self.model.z_dims):
            msg = 'sampling_variance (len {}) in objective must match z_dims (len {}) in model.'
            raise ValueError(msg.format(len(self.objective.sampling_variance), len(self.model.z_dims)))

        if len(self.objective.freebits) != len(self.model.z_dims):
            msg = 'freebits (len {}) in objective must match z_dims (len {}) in model.'
            raise ValueError(msg.format(len(self.objective.freebits), len(self.model.z_dims)))

        return self


@dataclass
class TrackerList:
    """Flags controlling which experiment trackers are active.

    Attributes:
        hydra: Saves a copy of the resolved config to the Hydra output dir.
        csv: Appends metrics to a CSV file in the experiment directory.
        tensorboard: Streams metrics to a local TensorBoard server.
        wandb: Uploads metrics to Weights & Biases (requires ``track_all`` extra).
        sqlalchemy: Persists metrics in a SQLite database (requires ``track_all`` extra).
    """

    hydra: bool
    csv: bool
    tensorboard: bool
    wandb: bool
    sqlalchemy: bool


@dataclass
class PlottingOptions:
    """Options for plotting and sample visualization."""

    sample_indices: list[int] = dataclasses.field(default_factory=list)
    use_train: bool = False
    n_delta: int = 5


@dataclass
class GenerateOptions:
    """Options for sample generation."""

    batch_size: PositiveInt = 16


@dataclasses.dataclass
class HydraSettings:
    """Subset of Hydra runtime settings, populated automatically at startup.

    Attributes:
        output_dir: Directory where Hydra stores logs and config dumps.
        job_logging: Hydra job-logging DictConfig used to configure Python logging.
    """

    output_dir: pathlib.Path = dataclasses.field(default_factory=pathlib.Path, init=False)
    job_logging: Any = dataclasses.field(default=None, init=False)


@dataclass
class PathSpecs:
    """Filesystem path specifications populated from environment variables.

    Attributes:
        root_exp_dir: Root directory under which all experiment artefacts are stored.
        data_dir: Dataset directory path loaded from environment variables.
        metadata_dir: Directory containing dataset metadata.

    Note:
        Paths are loaded from the ``.env`` file via :class:`~src.config.environment.EnvSettings`.
    """

    _env = EnvSettings()
    root_exp_dir: pathlib.Path = _env.root_exp_dir
    data_dir: pathlib.Path = _env.dataset_dir
    metadata_dir: pathlib.Path = _env.metadata_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """Version-scoped experiment directory: ``<root_exp_dir>/v<VERSION>``."""
        return self.root_exp_dir / f'v{VERSION}'


@dataclass
class UserSettings:
    """User-specific runtime preferences.

    Attributes:
        cpu: Force all computations onto CPU (useful for debugging).
        n_workers: DataLoader worker processes. ``0`` runs loading in the main process.
        n_subprocesses: Number of DDP processes. ``0`` disables distributed training.
        trackers: Active experiment-tracking backends.
        seed: Global random seed for reproducibility. ``None`` leaves it unset.
        checkpoint_every: Save a model checkpoint every *n* epochs.
        checkpoint_every: Save a model checkpoint every *n* epochs.
        load: Checkpoint epoch to resume from:``0`` skip; ``-1`` loads last checkpoint.
        use_recalibrated_bn: Whether generation reads the model whose batch normalization
            was re-estimated on the generative path. ``False`` reads the trained model
            with the statistics training left, which is what a decoding path close to the
            training one wants. Lives here rather than in the objective so that changing
            it does not rename the run.
    """

    cpu: bool
    n_workers: PositiveInt
    n_subprocesses: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    load: int = 0
    use_recalibrated_bn: bool = True
    plot: PlottingOptions = dataclasses.field(default_factory=PlottingOptions)
    generate: GenerateOptions = dataclasses.field(default_factory=GenerateOptions)
    hydra: HydraSettings = dataclasses.field(default_factory=HydraSettings)
    path: PathSpecs = dataclasses.field(default_factory=PathSpecs)

    def __post_init__(self) -> None:
        """Apply the random seed immediately after the config is constructed."""
        if self.seed is not None:
            set_seed(self.seed)

        return

    @property
    def device(self) -> torch.device | None:
        """Compute device: ``cpu`` when forced, ``None`` to use the default accelerator."""
        return torch.device('cpu') if self.cpu else None


@dataclass
class AllConfig:
    """Root configuration object for all experiment settings.

    Attributes:
        classification: Classification subexperiment configuration.
        user: User-specific options and runtime preferences.
        data: Data pre-processing and dataset configuration.
        final: If True, uses validation data for training and test data for evaluation.
        variation: Variation name for the experiment (defaults to git branch if None).
        tags: List of tags describing active configuration overrides.
    """

    classification: ClassificationExperimentConfig
    autoencoder: AutoencoderExperimentConfig
    user: UserSettings
    data: DataConfig
    final: bool

    variation: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = dataclasses.field(default=f'v{VERSION}', init=False)

    @cached_property
    def git_branch(self) -> str:
        """Current git branch (used when variation is unset)."""
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Set 'variation' in defaults.yaml."
            raise ValueError(msg) from ve

    @property
    def resolved_variation(self) -> str:
        """variation if set, otherwise the current git branch."""
        return self.variation or self.git_branch

    @property
    def name(self) -> str:
        """Full experiment name embedded into tracker runs."""
        suffix = '_final' if self.final else ''
        return self.resolved_variation[:249] + suffix

    @property
    def project(self) -> str:
        """Tracker project name"""
        return f'DrytorchTemplate{self.version}'
