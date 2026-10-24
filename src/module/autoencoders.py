"""Hierarchical VAE conditioned on the probabilities of a classifier (VAEX)."""

import abc
import itertools

from typing import Final, cast, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import InferenceBlock, Models
from src.config.torch import ActClass, NormClass
from src.data.structures import Encoding, Inputs, Outputs
from src.module.layers import (
    ConvBlock,
    FromLatentConv,
    NormFactory,
    ResBlock,
    ToLatentBlock,
    ToLatentConvADAIN,
    ToLatentConvMobile,
    get_batch_norm,
    soft_clamp,
)

__all__ = [
    'VAEX',
    'BaseAutoencoder',
    'BaseVAEX',
    'LatentStage',
    'get_autoencoder',
]

POSTERIOR_LOGVAR_BOUND: Final = 4.0
"""Symmetric soft bound on the log-variance from the inference path."""
PRIOR_LOGVAR_MAX: Final = 2.0
"""Soft upper bound on the log-variance from the top-down prior."""
PRIOR_LOGVAR_MIN: Final = -4.0
"""Soft lower bound, low enough that a stage the model has given up adds almost no noise."""


class LatentStage(nn.Module):
    """Stage of the decoder: a change of resolution, then a latent drawn at the new one.

    Attributes:
        decode: residual block changing the resolution.
        generate: prior of the latent, restyled by the latent above.
        merge: convolution joining the decoder and the encoder features.
        inference: shift of the posterior from the prior.
        from_latent: upsampling of the latent added to the features.
    """

    decode: Final[ResBlock]
    generate: Final[ToLatentBlock]
    merge: Final[ConvBlock]
    inference: Final[ToLatentBlock]
    from_latent: Final[FromLatentConv]

    def __init__(
        self,
        decode: ResBlock,
        generate: ToLatentBlock,
        merge: ConvBlock,
        inference: ToLatentBlock,
        from_latent: FromLatentConv,
    ) -> None:
        """Initialize.

        Args:
            decode: residual block changing the resolution.
            generate: prior of the latent, restyled by the latent above.
            merge: convolution joining the decoder and the encoder features.
            inference: shift of the posterior from the prior.
            from_latent: upsampling of the latent added to the features.
        """
        super().__init__()
        self.decode = decode
        self.generate = generate
        self.merge = merge
        self.inference = inference
        self.from_latent = from_latent
        return


class BaseAutoencoder(nn.Module, abc.ABC):
    """Abstract base for all autoencoder model architectures.

    Attributes:
        x_dim: Resolution of input images.
        n_channels: Number of input channels.
        h_dims: Feature channels at each resolution stage.
        k_dims: Kernel sizes of the resolution changes.
        strides: Strides of the resolution changes.
        paddings: Paddings of the resolution changes.
        depths: Residual layers before each resolution change.
        z_dims: Latent dimensions per stage.
        act_cls: Activation class of the layers.
        norm_cls: Normalization class of the layers.
        bn_momentum: Momentum for batch normalization.
        bn_eps: Epsilon for batch normalization.
        eval_std_divisor: Standard deviation divisor in evaluation mode.
    """

    x_dim: Final[int]
    n_channels: Final[int]
    h_dims: Final[tuple[int, ...]]
    k_dims: Final[tuple[int, ...]]
    strides: Final[tuple[int, ...]]
    paddings: Final[tuple[int, ...]]
    depths: Final[tuple[int, ...]]
    z_dims: Final[tuple[int, ...]]
    act_cls: Final[ActClass]
    norm_cls: Final[NormClass]
    bn_momentum: Final[float]
    bn_eps: Final[float]
    eval_std_divisor: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_model = cfg.autoencoder.model
        arch = cfg_model.architecture
        self.x_dim = cfg.data.x_dim
        self.n_channels = cfg.data.n_channels
        self.h_dims = arch.h_dims
        self.k_dims = arch.k_dims
        self.strides = arch.strides
        self.paddings = arch.paddings
        self.depths = arch.depths
        self.z_dims = cfg_model.z_dims
        self.act_cls = arch.act_cls
        self.norm_cls = arch.norm_cls
        self.bn_momentum = arch.bn_momentum
        self.bn_eps = arch.bn_eps
        self.eval_std_divisor = cfg_model.eval_std_divisor
        return

    @abc.abstractmethod
    def forward(self, inputs: Inputs, /) -> Outputs:
        """Run forward pass."""

    @abc.abstractmethod
    def reconstruct(self, inputs: Inputs, /, relaxation: float = 1.0) -> Outputs:
        """Reconstruct the inputs under the condition of the classifier."""


class BaseVAEX(BaseAutoencoder, abc.ABC):
    """Abstract base for explainable VAEs conditioned on classifier probabilities.

    Attributes:
        n_classes: Number of target classes.
        inference_block: Variant of the inference block.
        centering_steps: Number of centering steps for classifier probabilities.
        prior_scale: Prior mean scale of top latent conditioning channel.
    """

    n_classes: Final[int]
    inference_block: Final[InferenceBlock]
    centering_steps: Final[int]
    prior_scale: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_model = cfg.autoencoder.model
        arch = cfg_model.architecture
        self.n_classes = len(cfg.data.dataset.classes)
        self.inference_block = arch.inference_block
        self.centering_steps = cfg_model.centering_steps
        self.prior_scale = cfg.autoencoder.objective.prior_scale
        return

    @abc.abstractmethod
    def generate_counterfactual(self, image: torch.Tensor, target: int, relaxation: float) -> Outputs:
        """Generate counterfactual reconstructions."""

    @abc.abstractmethod
    def condition_from(self, inputs: Inputs) -> torch.Tensor:
        """Get the condition from inputs holding the log-probabilities of the classifier."""

    @abc.abstractmethod
    def generate(self, condition: torch.Tensor, /) -> torch.Tensor:
        """Generate images under a condition, sampling every latent from its prior."""

    @abc.abstractmethod
    def get_random_condition(self, batch_size: int) -> torch.Tensor:
        """Get the condition of random probabilities of the classes for each sample."""

    @abc.abstractmethod
    def get_target_condition(self, target: int, batch_size: int) -> torch.Tensor:
        """Get the condition of a class, a probability of one for it."""


class VAEX(BaseVAEX):
    """Hierarchical VAE conditioned on the probability that a classifier assigns to a sample.

    The condition enters as an extra channel of the image and as an extra channel of the top
    latent, whose first channel has the condition, scaled, as prior mean. Counterfactuals set
    condition and channel to a target, and relax the posterior of the lower latents.

    Attributes:
        top_channel_dim: Size of a channel of the top latent, flattened.
        condition_to_image: Map from the condition to an extra image channel.
        condition_to_latent: Map from the condition to an extra channel of the top latent.
        initial_convolution: Map from the image and its condition to the first features.
        encoding: Residual blocks of the encoder.
        final_inference: Posterior of the top latent.
        init_decoder: Learned features the top latent is added to.
        from_latent: Upsampling of the top latent and its condition.
        decoding: Stages of the decoder, from the top.
        final_convolution: Map from the last features to the image.
    """

    top_channel_dim: Final[int]
    condition_to_image: Final[nn.Sequential]
    condition_to_latent: Final[nn.Sequential]
    initial_convolution: Final[nn.Conv2d]
    encoding: Final[nn.ModuleList]
    final_inference: Final[ToLatentBlock]
    init_decoder: Final[nn.Parameter]
    from_latent: Final[FromLatentConv]
    decoding: Final[nn.ModuleList]
    final_convolution: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        norm = get_batch_norm(self.bn_momentum, self.bn_eps, self.norm_cls)
        to_latent_cls = ToLatentConvMobile if self.inference_block == InferenceBlock.MOBILE else ToLatentConvADAIN
        resolutions = self._get_resolutions()
        top_res = resolutions[-1]
        top_z_dim = self.z_dims[-1]
        self.top_channel_dim = ((top_res + 1) // 2) ** 2
        condition_dim = self.n_classes - 1
        self.condition_to_image = nn.Sequential(
            nn.Linear(condition_dim, self.x_dim**2),
            nn.ELU(),
            nn.Unflatten(1, (1, self.x_dim, self.x_dim)),
        )
        self.condition_to_latent = nn.Sequential(nn.Linear(condition_dim, self.top_channel_dim), nn.ELU())
        self.initial_convolution = nn.Conv2d(self.n_channels + 1, self.h_dims[0], 1, bias=False)
        self.encoding = self._get_encoding(norm)
        self.final_inference = to_latent_cls(self.h_dims[-1], top_z_dim, top_res, act_cls=self.act_cls, norm=norm)
        self.init_decoder = nn.Parameter(torch.ones(1, self.h_dims[-1], top_res, top_res))
        self.from_latent = FromLatentConv(self.h_dims[-1], top_z_dim + 1, top_res, act_cls=self.act_cls, norm=norm)
        self.decoding = self._get_decoding(norm, to_latent_cls, resolutions)
        self.final_convolution = nn.Sequential(
            ConvBlock(self.h_dims[0], self.n_channels, act_cls=self.act_cls, norm=norm),
            nn.Sigmoid(),
        )
        return

    @property
    def device(self) -> torch.device:
        """The device of the parameters."""
        return self.init_decoder.device

    @override
    def forward(self, inputs: Inputs, /) -> Outputs:
        """Reconstruct the images under the condition of the classifier."""
        return self.reconstruct(inputs)

    @override
    def extra_repr(self) -> str:
        return f'centering_steps={self.centering_steps}, eval_std_divisor={self.eval_std_divisor}'

    @override
    def condition_from(self, inputs: Inputs) -> torch.Tensor:
        """Get the centered probabilities of all classes but the last.

        Each centering applies ``(sqrt(p) - sqrt(1 - p) + 1) / 2``, which keeps 0, 1/2 and 1.

        Args:
            inputs: inputs holding the log-probabilities of the classifier.

        Returns:
            The condition, of shape ``(B, n_classes - 1)``.

        Raises:
            ValueError: if the inputs have no log-probabilities.
        """
        if inputs.log_probs.numel() == 0:
            raise ValueError('The inputs need the log-probabilities of the classifier.')

        return self._center(inputs.log_probs.exp())

    @override
    def generate_counterfactual(self, image: torch.Tensor, target: int, relaxation: float) -> Outputs:
        """Generate counterfactuals of the images for a target class.

        The target conditions both the encoder and the decoder, and sets the conditioned channel
        of the top latent; ``relaxation`` scales the posterior shifts of the lower latents.

        Args:
            image: the images.
            target: the class the counterfactuals aim at.
            relaxation: weight of the posterior, from 1 (reconstruction) to 0 (prior only).

        Returns:
            The counterfactuals in ``recon``.
        """
        condition = self.get_target_condition(target, len(image))
        encoding = self.encode(image, condition)
        z = encoding.z.clone()
        z[:, : condition.size(1) * self.top_channel_dim] = self._get_top_shift(condition)
        outputs = self.decode(z, condition, encoding.hidden, relaxation=relaxation)
        return outputs

    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        hidden: tuple[torch.Tensor, ...] | None,
        relaxation: float = 1.0,
    ) -> Outputs:
        """Decode the top latent under a condition.

        Args:
            z: sample of the top latent, flattened.
            condition: the condition.
            hidden: features of the encoder, from the coarsest; ``None`` samples the latents from
                their priors.
            relaxation: weight of the posterior shift of the first stage, of its mean and of its
                log-variance, squared at each stage; ``0`` samples the lower latents from their priors.

        Returns:
            The reconstruction, the prior mean of the top latent and, with the encoder features,
            the posterior shifts and the prior log-variances of the stages.
        """
        mus: list[torch.Tensor] = []
        log_vars: list[torch.Tensor] = []
        prior_log_vars: list[torch.Tensor] = []
        prior_mu = self._get_prior_mu(condition, z)
        z = torch.cat([z, self.condition_to_latent(condition)], dim=1)
        x = self.from_latent(z, self.init_decoder.expand(z.size(0), -1, -1, -1))
        for i, stage in enumerate(cast(list[LatentStage], list(self.decoding))):
            x = stage.decode(x)
            p_mu, p_log_var = stage.generate(x, z)
            p_log_var = soft_clamp(p_log_var, PRIOR_LOGVAR_MIN, PRIOR_LOGVAR_MAX)
            if hidden is None:
                z = self.sample(p_mu, p_log_var)
            else:
                d_mu, d_log_var = stage.inference(stage.merge(torch.cat([x, hidden[i]], dim=1)))
                d_log_var = soft_clamp(d_log_var, -POSTERIOR_LOGVAR_BOUND, POSTERIOR_LOGVAR_BOUND)
                z = self.sample(relaxation * d_mu + p_mu, relaxation * d_log_var + p_log_var)
                mus.append(d_mu)
                log_vars.append(d_log_var)
                prior_log_vars.append(p_log_var)

            x = stage.from_latent(z, x)
            relaxation *= relaxation

        recon = self.final_convolution(x)
        return Outputs(recon, prior_mu, mu=mus, log_var=log_vars, prior_log_var=prior_log_vars)

    def encode(self, image: torch.Tensor, condition: torch.Tensor) -> Encoding:
        """Encode the images under a condition.

        Args:
            image: the images.
            condition: the condition.

        Returns:
            The top latent with its posterior, and the features at each resolution.
        """
        x = self.initial_convolution(torch.cat([image, self.condition_to_image(condition)], dim=1))
        hidden: list[torch.Tensor] = []
        for encode in self.encoding:
            hidden.append(x)
            x = encode(x)

        mu, log_var = self.final_inference(x)
        log_var = soft_clamp(log_var, -POSTERIOR_LOGVAR_BOUND, POSTERIOR_LOGVAR_BOUND)
        return Encoding(z=self.sample(mu, log_var), mu=mu, log_var=log_var, hidden=tuple(reversed(hidden)))

    @override
    def generate(self, condition: torch.Tensor, /) -> torch.Tensor:
        """Generate images under a condition, sampling every latent from its prior.

        Args:
            condition: the condition of each image.

        Returns:
            The generated images.
        """
        top_dim = self.z_dims[-1] * self.top_channel_dim
        standard = torch.zeros(len(condition), top_dim, device=self.device)
        z = self.sample(standard, standard)  # the prior of the top latent is standard
        z[:, : condition.size(1) * self.top_channel_dim] = self._get_top_shift(condition)
        return self.decode(z, condition, None).recon

    @override
    def get_random_condition(self, batch_size: int) -> torch.Tensor:
        """Get the condition of random probabilities of the classes for each sample.

        The probabilities are uniform over all distributions on the classes, so that a sample
        can be of a class, of another, or in between, as ``(0.5, 0.5)``.

        Args:
            batch_size: number of samples.

        Returns:
            The condition, of shape ``(B, n_classes - 1)``, on the device of the parameters.
        """
        concentration = torch.ones(self.n_classes, device=self.device)
        probabilities = torch.distributions.Dirichlet(concentration).sample((batch_size,))
        return self._center(probabilities)

    @override
    def get_target_condition(self, target: int, batch_size: int) -> torch.Tensor:
        """Get the condition of a class, a probability of one for it.

        Args:
            target: the class.
            batch_size: number of samples.

        Returns:
            The condition, one-hot on all classes but the last, on the device of the parameters.
        """
        condition = torch.zeros(batch_size, self.n_classes - 1, device=self.device)
        if target < self.n_classes - 1:
            condition[:, target] = 1

        return condition

    @override
    def reconstruct(self, inputs: Inputs, /, relaxation: float = 1.0) -> Outputs:
        """Reconstruct the images under the condition of the classifier.

        Args:
            inputs: the images and the log-probabilities of the classifier.
            relaxation: weight of the posterior of the lower latents; ``1`` reconstructs the
                images, ``0`` gives their simple cut, the decoding with no posterior shift.

        Returns:
            The reconstructions, with the posteriors and the priors.
        """
        condition = self.condition_from(inputs)
        encoding = self.encode(inputs.image, condition)
        outputs = self.decode(encoding.z, condition, encoding.hidden, relaxation=relaxation)
        outputs.mu.insert(0, encoding.mu)
        outputs.log_var.insert(0, encoding.log_var)
        return outputs

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample a Gaussian, with a smaller standard deviation in evaluation mode.

        Args:
            mu: the mean.
            log_var: the log-variance.

        Returns:
            The sample.
        """
        std = torch.exp(0.5 * log_var)
        if not self.training:
            std = std / self.eval_std_divisor

        return mu + torch.randn_like(std) * std

    def _center(self, probabilities: torch.Tensor) -> torch.Tensor:
        # the condition holds all classes but the last, which the others imply
        condition = probabilities[:, :-1]
        for _ in range(self.centering_steps):
            condition = (condition.sqrt() - (1 - condition).sqrt() + 1) / 2

        return condition

    def _get_prior_mu(self, condition: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        prior_mu = torch.zeros_like(z)
        prior_mu[:, : condition.size(1) * self.top_channel_dim] = self._get_top_shift(condition)
        return prior_mu

    def _get_encoding(self, norm: NormFactory) -> nn.ModuleList:
        encoding = nn.ModuleList()
        widths = itertools.pairwise(self.h_dims)
        zipped = zip(widths, self.k_dims, self.strides, self.paddings, self.depths, strict=True)
        for (h_in, h_out), k_dim, stride, padding, depth in zipped:
            encoding.append(
                ResBlock(
                    h_in,
                    h_out,
                    act_cls=self.act_cls,
                    norm=norm,
                    kernel_size=k_dim,
                    stride=stride,
                    padding=padding,
                    depth=depth,
                )
            )

        return encoding

    def _get_resolutions(self) -> list[int]:
        resolutions = [self.x_dim]
        for k_dim, stride, padding in zip(self.k_dims, self.strides, self.paddings, strict=True):
            resolutions.append((resolutions[-1] + 2 * padding - k_dim) // stride + 1)

        return resolutions

    def _get_decoding(
        self,
        norm: NormFactory,
        to_latent_cls: type[ToLatentConvADAIN] | type[ToLatentConvMobile],
        resolutions: list[int],
    ) -> nn.ModuleList:
        *stage_z_dims, top_z_dim = self.z_dims
        decoding = nn.ModuleList()
        aux_dim = top_z_dim + 1
        zipped_stages = zip(
            itertools.pairwise(reversed(self.h_dims)),
            reversed(self.k_dims),
            reversed(self.strides),
            reversed(self.paddings),
            reversed(stage_z_dims),
            reversed(resolutions[:-1]),
            reversed(self.depths),
            strict=True,
        )
        for (h_in, h_out), k_dim, stride, padding, z_dim, res, depth in zipped_stages:
            settings = (k_dim, stride, padding, z_dim, res, depth, aux_dim)
            stage = self._get_stage(h_in, h_out, *settings, norm, to_latent_cls)
            decoding.append(stage)
            aux_dim = z_dim

        return decoding

    def _get_stage(
        self,
        h_in: int,
        h_out: int,
        k_dim: int,
        stride: int,
        padding: int,
        z_dim: int,
        res: int,
        depth: int,
        aux_dim: int,
        norm: NormFactory,
        to_latent_cls: type[ToLatentConvADAIN] | type[ToLatentConvMobile],
    ) -> LatentStage:
        return LatentStage(
            decode=ResBlock(
                h_in,
                h_out,
                act_cls=self.act_cls,
                norm=norm,
                kernel_size=k_dim,
                stride=stride,
                padding=padding,
                deconv=True,
                depth=depth,
            ),
            generate=to_latent_cls(h_out, z_dim, res, aux_dim, act_cls=self.act_cls, norm=norm),
            merge=ConvBlock(2 * h_out, h_out, act_cls=self.act_cls, norm=norm),
            inference=to_latent_cls(h_out, z_dim, res, act_cls=self.act_cls, norm=norm),
            from_latent=FromLatentConv(h_out, z_dim, res, act_cls=self.act_cls, norm=norm),
        )

    def _get_top_shift(self, condition: torch.Tensor) -> torch.Tensor:
        return self.prior_scale * condition.repeat_interleave(self.top_channel_dim, dim=1)


def get_autoencoder() -> BaseVAEX:
    """Get autoencoder architecture based on experiment config.

    Returns:
        The autoencoder module.
    """
    cfg = Experiment.get_config()
    model_name = cfg.autoencoder.model.name
    model_map: dict[Models, type[BaseVAEX]] = {
        Models.VAEX: VAEX,
    }
    return model_map[model_name]()
