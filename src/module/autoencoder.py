"""Hierarchical VAE and VAEX autoencoder architectures."""

import abc
import itertools

from typing import override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import InferenceBlock, Models
from src.data.structures import Inputs, Outputs
from src.module.layers import (
    ConvBlock,
    FromLatentConv,
    LinearLayer,
    NullOp,
    ResBlock,
    ToLatentBlock,
    ToLatentConvADAIN,
    ToLatentConvMobile,
    InformationEraser,
    reverse_zip,
    soft_clamp,
)


PRIOR_LOGVAR_BOUND = 2.0
POSTERIOR_LOGVAR_BOUND = 4.0


class Abstract_VAE(nn.Module, abc.ABC):
    """Abstract base interface for hierarchical VAE architectures."""

    n_channels: int
    dim_target: int
    x_dim: int
    z_dims: tuple[int, ...]
    sampling_variance: tuple[float, ...]
    posterior_weight_multiplier: float
    cf_strength: float
    z_cond_dim: int
    reversal_strength: float

    def __init__(self) -> None:
        super().__init__()
        exp_cfg = Experiment.get_config()
        self.n_channels = exp_cfg.data.n_channels
        self.dim_target = len(exp_cfg.data.dataset.classes)
        self.x_dim = exp_cfg.data.x_dim
        self.z_dims = exp_cfg.autoencoder.model.z_dims
        self.sampling_variance = exp_cfg.autoencoder.objective.sampling_variance
        self.posterior_weight_multiplier = exp_cfg.autoencoder.objective.posterior_weight_multiplier
        self.cf_strength = exp_cfg.autoencoder.objective.cf_strength
        self.z_cond_dim = exp_cfg.autoencoder.model.z_cond_dim
        self.reversal_strength = exp_cfg.autoencoder.objective.reversal_strength
        return

    @property
    def full_posterior_weights(self) -> tuple[float, ...]:
        """Per-stage weights that keep the entire posterior, as used for reconstruction."""
        return (1.0,) * len(self.z_dims)

    @property
    def posterior_weights(self) -> tuple[float, ...]:
        """Per-stage weights decaying geometrically, as configured for generation."""
        return self.posterior_weights_from(self.posterior_weight_multiplier)

    def posterior_weights_from(self, multiplier: float) -> tuple[float, ...]:
        """Build per-stage posterior weights growing geometrically towards the pixels.

        These weight the conditioned half of each stage only; the free half always keeps
        its posterior, so identity is preserved without the schedule having to protect it.
        Every stage is cut, the finest least and the root most, which is what lets the
        condition rewrite the semantics while the texture still follows the source.
        ``1.0`` keeps the whole posterior everywhere and ``0.0`` removes the conditioned
        half entirely.

        Args:
            multiplier: ratio between the weights of consecutive stages.
        """
        return tuple(multiplier ** (len(self.z_dims) - level) for level in range(len(self.z_dims)))

    @property
    @abc.abstractmethod
    def final_z_dim(self) -> int:
        """Final latent dimension size."""

    @property
    @abc.abstractmethod
    def final_res(self) -> int:
        """Final spatial resolution."""

    @abc.abstractmethod
    def condition_from(self, inputs: Inputs) -> torch.Tensor:
        """Derive the (batch, dim_target) conditioning vector for the given inputs."""

    @abc.abstractmethod
    def sample_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Sample a random condition tensor."""

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> Outputs:
        """Encode input into latent space."""

    @abc.abstractmethod
    def decode(
        self,
        out: Outputs,
        condition: torch.Tensor,
        posterior_weights: tuple[float, ...],
    ) -> Outputs:
        """Decode latent space into reconstruction.

        Args:
            out: encoder output, or an empty instance to decode from the priors alone.
            condition: conditioning vector parameterising the root prior.
            posterior_weights: per-stage weight on the posterior offsets, one entry per
                latent stage. ``1.0`` keeps the stage posterior, ``0.0`` falls back to
                the stage prior.
        """

    def forward(self, inputs: Inputs, **kwargs) -> Outputs:
        x = inputs.image
        condition = self.condition_from(inputs)
        out = self.encode(x)
        out = self.decode(out, condition, self.full_posterior_weights)
        out.image = x
        out.condition = condition
        out.logits = inputs.log_probs if inputs.log_probs.numel() > 0 else condition.clamp_min(1e-8).log()
        return out

    def generate_counterfactual(
        self,
        inputs: Inputs,
        target_dim: int = 0,
        posterior_weights: tuple[float, ...] | None = None,
    ) -> Outputs:
        """Generate a counterfactual sample for the given inputs and target class.

        Args:
            inputs: batch to intervene on.
            target_dim: class the root prior is conditioned on.
            posterior_weights: per-stage weight on the posterior offsets; defaults to the
                decaying weights configured for generation.
        """
        if not 0 <= target_dim < self.dim_target:
            raise ValueError(f'target_dim {target_dim} outside [0, {self.dim_target})')

        if posterior_weights is None:
            posterior_weights = self.posterior_weights
            posterior_weights = tuple((*posterior_weights[:-1], 100.0))

        x = inputs.image
        orig_condition = self.condition_from(inputs)
        target = torch.zeros((x.size(0), self.dim_target), device=x.device, dtype=x.dtype)
        target[:, target_dim] = 1.0
        cf_condition = (1.0 - self.cf_strength) * orig_condition + self.cf_strength * target
        out = self.encode(x)
        out = self.decode(out, cf_condition, posterior_weights)
        out.image = x
        out.condition = cf_condition
        return out

    @torch.inference_mode()
    def generate_sample(
        self,
        batch_size: int = 1,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate random sample for given batch size and condition."""
        if condition is None:
            condition = self.sample_condition(batch_size)

        # No encoder output, so every stage falls back to its prior and the weights are
        # inert; the root prior is what makes the sample class-controlled.
        out = Outputs()
        out.hidden = []
        return self.decode(
            out, condition.to(next(self.parameters()).device), self.full_posterior_weights
        ).recon

    @staticmethod
    def _half_weight(reference: torch.Tensor, cond_width: int, weight: float) -> torch.Tensor:
        """Weight the conditioned half of a stage and leave the free half at one.

        An intervention replaces the class and keeps the identity, so the per-stage weight
        belongs on the conditioned half alone. A weight of one gives the identity, which
        is what reconstruction passes, so the training path is unaffected.

        Args:
            reference: the stage offset, used for width, dtype and device.
            cond_width: flat width of the conditioned half.
            weight: weight to apply there.
        """
        half = reference.new_ones((1, reference.size(1)))
        half[:, :cond_width] = weight
        return half

    def sampling(self, mu: torch.Tensor, log_var: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu) if self.training else eps.mul(std * temperature).add_(mu)
        return z


class BVAE(Abstract_VAE, abc.ABC):
    """Abstract base class for hierarchical VAE architectures handling network layers and dimension calculations."""

    h_dims: tuple[int, ...]
    k_dims: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]
    c_z_dims: list[int]
    depths: tuple[int, ...]
    l_dims_encoder: list[int]
    l_dims_decoder: list[int]
    resolutions: list[int]
    flat_dim: int
    act_cls: type[nn.Module] | None
    inference_block: InferenceBlock
    final_inference: ToLatentBlock

    def __init__(self) -> None:
        super().__init__()
        exp_cfg = Experiment.get_config()
        vae_arch = exp_cfg.autoencoder.model.architecture
        self.h_dims = vae_arch.h_dims
        self.k_dims = vae_arch.k_dims
        self.strides = vae_arch.strides
        self.paddings = vae_arch.paddings
        self.depths = vae_arch.depths
        self.act_cls = vae_arch.act_cls
        self.inference_block = vae_arch.inference_block
        self.l_dims_encoder, self.l_dims_decoder, self.flat_dim = self.size_calc()
        *self.c_z_dims, _ = self.z_dims
        *self.resolutions, _ = self.l_dims_encoder
        self._build_encoder_layers()
        self._build_decoder_layers()
        return

    @property
    @override
    def final_z_dim(self) -> int:
        return self.z_dims[-1]

    @property
    @override
    def final_res(self) -> int:
        return self.l_dims_encoder[-1]

    @property
    def flat_z_dim(self) -> int:
        """Flat width of the root latent."""
        return self.final_z_dim * ((self.final_res + 1) // 2) ** 2

    @property
    def flat_z_free_dim(self) -> int:
        """Flat width of the unconditioned root latent."""
        return self.flat_z_dim - self.flat_z_cond_dim

    @property
    def flat_z_cond_dim(self) -> int:
        """Flat width of the conditioned root latent."""
        return self.z_cond_dim * ((self.final_res + 1) // 2) ** 2

    def _build_encoder_layers(self) -> None:
        self.initial_convolution = nn.Conv2d(self.n_channels, self.h_dims[0], 1, bias=False)
        self.encoding = nn.ModuleList([])
        for (h_in, h_out), k_dim, stride, padding, depth in zip(
            itertools.pairwise(self.h_dims), self.k_dims, self.strides, self.paddings, self.depths, strict=False
        ):
            encode = ResBlock(
                ConvBlock,
                h_in,
                h_out,
                k_dim,
                stride,
                padding,
                deconv=False,
                depth=depth,
            )
            self.encoding.append(encode)

        if self.inference_block == InferenceBlock.Mobile:
            self.final_inference = ToLatentConvMobile(self.h_dims[-1], self.final_z_dim, self.final_res, 0)
        else:
            self.final_inference = ToLatentConvADAIN(self.h_dims[-1], self.final_z_dim, self.final_res, 0)

        return

    def _build_decoder_layers(self) -> None:
        self.root_prior_cond = LinearLayer(self.dim_target, 2 * self.flat_z_cond_dim, use_trunc_init=True)
        self.root_prior_free = nn.parameter.Parameter(torch.zeros(1, 2 * self.flat_z_free_dim))
        # Both the mean and the log-variance of the free half, so that class information
        # cannot sidestep the adversary by moving into the noise scale.
        self.information_eraser = InformationEraser(
            in_dim=2 * self.flat_z_free_dim,
            n_classes=self.dim_target,
            reversal_strength=self.reversal_strength
        )
        self.h = nn.parameter.Parameter(torch.ones(1, self.h_dims[-1], self.final_res, self.final_res))
        self.back_to_img = FromLatentConv(self.h_dims[-1], self.final_z_dim, self.final_res)
        self.decode_list = nn.ModuleList([])
        self.inference_list = nn.ModuleList([])
        self.generate_cond_list = nn.ModuleList([])
        self.generate_free_list = nn.ModuleList([])
        self.information_erasers = nn.ModuleList([])
        self.from_latent_list = nn.ModuleList([])
        self.merge = nn.ModuleList([])
        aux_dim = self.final_z_dim
        for (h_in, h_out), k_dim, stride, padding, z_dim, res, depth in zip(
            itertools.pairwise(reversed(self.h_dims)),
            reversed(self.k_dims),
            reversed(self.strides),
            reversed(self.paddings),
            reversed(self.c_z_dims),
            reversed(self.resolutions),
            reversed(self.depths),
            strict=False,
        ):
            decode = ResBlock(
                ConvBlock,
                h_in,
                h_out,
                k_dim,
                stride,
                padding,
                deconv=True,
                depth=depth,
            )
            self.decode_list.append(decode)
            if z_dim:
                from_latent = FromLatentConv(h_out, z_dim, res)
                inference: ToLatentBlock
                generate_cond: ToLatentBlock
                generate_free: ToLatentBlock
                eraser: nn.Module
                
                if self.inference_block == InferenceBlock.Mobile:
                    inference = ToLatentConvMobile(h_out, z_dim, res, cond_dim=self.dim_target)
                else:
                    inference = ToLatentConvADAIN(h_out, z_dim, res, cond_dim=self.dim_target)
                
                z_cond_dim_stage = max(0, min(self.z_cond_dim, z_dim - 1))
                z_free_dim_stage = z_dim - z_cond_dim_stage
                block = ToLatentConvMobile if self.inference_block == InferenceBlock.Mobile else ToLatentConvADAIN
                generate_cond = (
                    block(h_out, z_cond_dim_stage, res, aux_dim, cond_dim=self.dim_target)
                    if z_cond_dim_stage
                    else NullOp()
                )
                generate_free = block(h_out, z_free_dim_stage, res, aux_dim, cond_dim=0)
                in_dim = 2 * z_free_dim_stage * (((res + 1) // 2) ** 2)
                eraser = InformationEraser(
                    in_dim=in_dim,
                    n_classes=self.dim_target,
                    reversal_strength=self.reversal_strength,
                    intermediate_dim=min(512, max(16, 512000 // in_dim)),
                )

                merge = ConvBlock(2 * h_out, h_out)
                self.inference_list.append(inference)
                self.generate_cond_list.append(generate_cond)
                self.generate_free_list.append(generate_free)
                self.information_erasers.append(eraser)
                self.from_latent_list.append(from_latent)
                self.merge.append(merge)
            else:
                self.inference_list.append(NullOp())
                self.generate_cond_list.append(NullOp())
                self.generate_free_list.append(NullOp())
                self.information_erasers.append(NullOp())
                self.from_latent_list.append(NullOp())
                self.merge.append(NullOp())

            aux_dim = z_dim or 0

        self.final_convolution = ConvBlock(self.h_dims[0], self.n_channels, use_trunc_init=True)
        return

    def size_calc(self) -> tuple[list[int], list[int], int]:
        l_dims = self.x_dim
        l_dims_encoder = [l_dims]
        for k_dim, stride, padding in zip(self.k_dims, self.strides, self.paddings, strict=False):
            l_dims = self.conv_size_calc(L_in=l_dims, kernel_size=k_dim, stride=stride, padding=padding)
            l_dims_encoder.append(l_dims)

        flat_dim = self.h_dims[-1] * (l_dims_encoder[-1] ** 2)
        l_dims_decoder = [l_dims]
        for k_dim, stride, padding in reverse_zip(self.k_dims, self.strides, self.paddings):
            l_dims = self.convt_size_calc(L_in=l_dims, kernel_size=k_dim, stride=stride, padding=padding)
            l_dims_decoder.append(l_dims)

        return l_dims_encoder, l_dims_decoder, flat_dim

    @staticmethod
    def conv_size_calc(L_in: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: int = 0) -> int:
        return int((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    @staticmethod
    def convt_size_calc(
        L_in: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
    ) -> int:
        return (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    def root_prior(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_mu_cond, p_logvar_cond = self.root_prior_cond(self._process_condition(condition)).chunk(2, dim=1)
        p_mu_free, p_logvar_free = self.root_prior_free.chunk(2, dim=1)
        batch_size = condition.size(0)
        p_mu_free = p_mu_free.expand(batch_size, -1)
        p_logvar_free = p_logvar_free.expand(batch_size, -1)
        p_mu = torch.cat([p_mu_cond, p_mu_free], dim=1)
        p_logvar = torch.cat([p_logvar_cond, p_logvar_free], dim=1)
        return p_mu, p_logvar

    @override
    def encode(self, x: torch.Tensor) -> Outputs:
        out = Outputs()
        x = self.initial_convolution(x)
        for encode in self.encoding:
            out.hidden.append(x)
            x = encode(x)

        d_mu, d_logvar = self.final_inference(x)
        out.d_mu.append(d_mu)
        out.d_logvar.append(soft_clamp(d_logvar, -POSTERIOR_LOGVAR_BOUND, POSTERIOR_LOGVAR_BOUND))
        return out

    @override
    def decode(
        self,
        out: Outputs,
        condition: torch.Tensor,
        posterior_weights: tuple[float, ...],
    ) -> Outputs:
        out.prior_mu = []
        out.prior_log_var = []
        out.adv_logits = []
        stage_condition = self._process_condition(condition)
        p_mu, p_logvar = self.root_prior(condition)
        p_logvar = soft_clamp(p_logvar, -PRIOR_LOGVAR_BOUND, PRIOR_LOGVAR_BOUND)
        posterior = bool(out.d_mu)
        if posterior:
            root_weight = self._half_weight(out.d_mu[0], self.flat_z_cond_dim, posterior_weights[0])
            z = self.sampling(
                root_weight * out.d_mu[0] + p_mu,
                root_weight * out.d_logvar[0] + p_logvar,
                temperature=self.sampling_variance[0],
            )
            free = slice(self.flat_z_cond_dim, None)
            mu_free = p_mu[:, free] + out.d_mu[0][:, free]
            logvar_free = p_logvar[:, free] + out.d_logvar[0][:, free]
            out.adv_logits.append(self.information_eraser(torch.cat([mu_free, logvar_free], dim=1)))
        else:
            z = self.sampling(p_mu, p_logvar, temperature=self.sampling_variance[0])

        out.prior_mu.append(p_mu)
        out.prior_log_var.append(p_logvar)
        h = self.h.expand(z.size(0), -1, -1, -1)
        x = self.back_to_img(z, h)
        zipped = zip(
            self.decode_list, self.from_latent_list, self.generate_cond_list, self.generate_free_list, self.inference_list, self.merge, self.information_erasers, strict=False
        )
        for level, (decode, from_latent, generate_cond, generate_free, inference, merge, eraser) in enumerate(zipped, start=1):
            x = decode(x)
            h = out.hidden.pop() if posterior and out.hidden else None
            if h is not None:
                out.hidden.insert(0, h)

            if not isinstance(from_latent, NullOp):
                temperature = self.sampling_variance[level]
                cond_width = 0
                mu_parts, logvar_parts = [], []
                if not isinstance(generate_cond, NullOp):
                    p_mu_cond, p_logvar_cond = generate_cond(x, z, stage_condition)
                    cond_width = p_mu_cond.size(1)
                    mu_parts.append(p_mu_cond)
                    logvar_parts.append(p_logvar_cond)

                p_mu_free, p_logvar_free = generate_free(x, z, None)
                mu_parts.append(p_mu_free)
                logvar_parts.append(p_logvar_free)
                p_mu = torch.cat(mu_parts, dim=1)
                p_logvar = soft_clamp(torch.cat(logvar_parts, dim=1), -PRIOR_LOGVAR_BOUND, PRIOR_LOGVAR_BOUND)
                if h is not None:
                    h = merge(torch.cat([x, h], dim=1))
                    d_mu, d_logvar = inference(h, condition=stage_condition)
                    d_logvar = soft_clamp(d_logvar, -POSTERIOR_LOGVAR_BOUND, POSTERIOR_LOGVAR_BOUND)
                    stage_weight = self._half_weight(d_mu, cond_width, posterior_weights[level])
                    z = self.sampling(
                        stage_weight * d_mu + p_mu,
                        stage_weight * d_logvar + p_logvar,
                        temperature=temperature,
                    )
                    out.d_mu.append(d_mu)
                    out.d_logvar.append(d_logvar)
                    free_slice = slice(cond_width, None)
                    mu_free = p_mu[:, free_slice] + d_mu[:, free_slice]
                    logvar_free = p_logvar[:, free_slice] + d_logvar[:, free_slice]
                    out.adv_logits.append(eraser(torch.cat([mu_free, logvar_free], dim=1)))
                else:
                    z = self.sampling(p_mu, p_logvar, temperature=temperature)

                x = from_latent(z, x)
                out.prior_mu.append(p_mu)
                out.prior_log_var.append(p_logvar)

        out.recon = torch.sigmoid(self.final_convolution(x))
        return out

    def _process_condition(self, condition: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if condition.dim() == 1:
            if condition.is_floating_point():
                condition = condition.unsqueeze(1)
            else:
                condition = nn.functional.one_hot(condition.long(), num_classes=self.dim_target)

        if condition.size(1) == 1 and self.dim_target == 2:
            condition = torch.cat([1 - condition, condition], dim=1)

        if dtype is not None:
            condition = condition.to(dtype=dtype)
        else:
            condition = condition.float()

        return condition


class CVAE(BVAE):
    """Bottom-Up / Top-Down Hierarchical Conditional Variational Autoencoder."""

    @override
    def condition_from(self, inputs: Inputs) -> torch.Tensor:
        """Use the explicit label condition carried by the inputs."""
        condition = inputs.condition
        if condition.numel() == 0:
            raise ValueError('label MUST be in Inputs else fail')

        return self._process_condition(condition, dtype=inputs.image.dtype)

    @override
    def sample_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Sample uniform random class."""
        device = next(self.parameters()).device
        random_classes = torch.randint(0, self.dim_target, (batch_size,), device=device)
        return nn.functional.one_hot(random_classes, num_classes=self.dim_target).float()


class VAEX(BVAE):
    """Hierarchical VAE conditioned on the predictions of an external classifier."""

    temperature: float

    def __init__(self) -> None:
        super().__init__()
        exp_cfg = Experiment.get_config()
        self.temperature = exp_cfg.autoencoder.model.temperature
        return

    @override
    def condition_from(self, inputs: Inputs) -> torch.Tensor:
        """Temper the classifier's distribution into a conditioning vector."""
        if inputs.log_probs.numel() == 0:
            raise ValueError('log_probs MUST be in Inputs else fail')

        return torch.softmax(inputs.log_probs / self.temperature, dim=1)

    @override
    def sample_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Sample random probabilities from Dirichlet distribution."""
        device = next(self.parameters()).device
        alpha = torch.ones(self.dim_target, device=device)
        return torch.distributions.Dirichlet(concentration=alpha).sample((batch_size,))


def get_autoencoder() -> Abstract_VAE:
    """Instantiate CVAE / VAEX model architecture based on autoencoder config selection."""
    cfg = Experiment.get_config()
    model_name = cfg.autoencoder.model.name
    _REGISTRY: dict[Models, type[Abstract_VAE]] = {
        Models.CVAE: CVAE,
        Models.VAEX: VAEX,
    }
    cls = _REGISTRY.get(model_name, VAEX)
    return cls()
