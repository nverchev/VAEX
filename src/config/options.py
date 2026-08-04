"""Options and choices for configuration settings."""

import enum


class Datasets(enum.StrEnum):
    """Dataset choices."""

    CelebaMale = enum.auto()
    """CelebA cropped facial attribute dataset specifically for Male attribute."""


class Models(enum.StrEnum):
    """Autoencoder VAE model choices."""

    CVAE = enum.auto()
    """Conditional Variational Autoencoder (Conditions on the attribute)."""
    VAEX = enum.auto()
    """Explainable Variational Autoencoder (Conditions on the classifier's attribute evalution)."""


class Conditioning(enum.StrEnum):
    """Conditioning variants."""

    Learned = 'Learned'
    """Conditioning via learned representations."""
    Explicit = 'Explicit'
    """Explicit conditioning using known attributes or logits."""
    ExplicitAndLearned = 'ExplicitAndLearned'
    """Combined explicit and learned conditioning."""


class InferenceBlock(enum.StrEnum):
    """Inference block architecture variant."""

    ADAIN = enum.auto()
    """Adaptive Instance Normalization based inference block."""
    Mobile = enum.auto()
    """Mobile inverted bottleneck based inference block with concatenation."""


class GradOp(enum.StrEnum):
    """Gradient operation names."""

    GradParamNormalizer = enum.auto()
    """Normalize gradients by parameter norm."""
    GradZScoreNormalizer = enum.auto()
    """Standardize gradients using running Z-score statistics."""
    GradNormClipper = enum.auto()
    """Clip global gradient norm."""
    GradValueClipper = enum.auto()
    """Clip elementwise gradient values."""
    HistClipper = enum.auto()
    """Histogram-based global gradient clipper."""
    ParamHistClipper = enum.auto()
    """Histogram-based per-parameter gradient clipper."""


class ClipCriterion(enum.StrEnum):
    """Clipping criterion names for histogram-based gradient clippers."""

    ZStat = enum.auto()
    """Z-score statistical clipping threshold."""
    EMA = enum.auto()
    """Exponential moving average threshold."""


class Schedulers(enum.StrEnum):
    """Learning-rate scheduler names."""

    Constant = enum.auto()
    """Fixed constant learning rate."""
    Cosine = enum.auto()
    """Cosine annealing learning rate schedule."""
    Exponential = enum.auto()
    """Exponential decay learning rate schedule."""
    Polynomial = enum.auto()
    """Polynomial decay learning rate schedule."""
    Step = enum.auto()
    """Step-wise learning rate decay schedule."""
