"""Hierarchical VAE and VAEX autoencoder architectures."""

import abc
import itertools

import torch
import torch.nn as nn

from typing import override

from src.config.experiment import Experiment
from src.config.options import Models
from src.data.structures import Inputs, Outputs
from src.module.layers import (
    ConvBlock,
    FromLatentConv,
    LinearLayer,
    NullOp,
    ResBlock,
    ToLatentConvADAIN,
    View,
    reverse_zip,
    soft_clamp,
)

class Abstract_VAE(nn.Module, abc.ABC):
    """Abstract base class for hierarchical VAE architectures."""

    n_channels: int
    dim_target: int
    x_dim: int
    h_dims: list[int]
    k_dims: list[int]
    strides: list[int]
    paddings: list[int]
    z_dims: list[int]
    depths: list[int]
    l_dims_encoder: list[int]
    l_dims_decoder: list[int]
    flat_dim: int

    def __init__(self) -> None:
        super().__init__()
        exp_cfg = Experiment.get_config()
        self.n_channels = exp_cfg.data.n_channels
        self.dim_target = len(exp_cfg.data.dataset.classes)
        vae_arch = exp_cfg.autoencoder.model.architecture
        self.x_dim = vae_arch.x_dim
        self.h_dims = list(vae_arch.h_dims)
        self.k_dims = list(vae_arch.k_dims)
        self.strides = list(vae_arch.strides)
        self.paddings = list(vae_arch.paddings)
        self.z_dims = list(vae_arch.z_dims)
        self.depths = list(vae_arch.depths)
        self.l_dims_encoder, self.l_dims_decoder, self.flat_dim = self.size_calc()
        return

    @property
    @abc.abstractmethod
    def final_z_dim(self) -> int:
        """Final latent dimension size."""

    @property
    @abc.abstractmethod
    def final_res(self) -> int:
        """Final spatial resolution."""

    @abc.abstractmethod
    def generate_counterfactual(
        self,
        inputs: Inputs,
        target_dim: int = 0,
        target_value: float = 5.0,
        posterior_weight: float | None = None,
    ) -> Outputs:
        """Generate counterfactual sample for given inputs and target dimension."""

    @abc.abstractmethod
    def sample_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Sample a random condition tensor."""

    @abc.abstractmethod
    def encode(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> Outputs:
        """Encode input into latent space."""

    @abc.abstractmethod
    def decode(
        self,
        out: Outputs,
        condition: torch.Tensor | None = None,
        sample: torch.Tensor | None = None,
        posterior_weight: float = 1.0,
    ) -> Outputs:
        """Decode latent space into reconstruction."""

    @torch.inference_mode()
    def generate_sample(
        self,
        batch_size: int = 1,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate random sample for given batch size and condition."""
        if condition is None:
            condition = self.sample_condition(batch_size)

        out = Outputs()
        out.hidden = []
        out.z = []
        flat_z_dim = self.final_z_dim * ((self.final_res + 1) // 2) ** 2
        sample = torch.randn(batch_size, flat_z_dim, device=next(self.parameters()).device)
        return self.decode(out, condition=condition, sample=sample).recon

    def sampling(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        clamped_logvar = soft_clamp(log_var, min_val=-10.0, max_val=10.0)
        std = torch.exp(0.5 * clamped_logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu) if self.training else eps.mul(std / 3.0).add_(mu)
        return z

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


class CVAE(Abstract_VAE):
    """Bottom-Up / Top-Down Hierarchical Variational Autoencoder."""

    act_cls: type[nn.Module] | None
    c_z_dims: list[int]
    resolutions: list[int]
    aux_dim: int

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.act_cls = cfg.autoencoder.model.architecture.act_cls
        *self.c_z_dims, _ = self.z_dims
        *self.resolutions, _ = self.l_dims_encoder
        self.aux_dim = self.final_z_dim + (self.dim_target > 0)
        self._build_encoder_layers()
        self._build_decoder_layers()
        return

    @property
    def final_z_dim(self) -> int:
        return self.z_dims[-1]

    @property
    def final_res(self) -> int:
        return self.l_dims_encoder[-1]

    def _build_encoder_layers(self) -> None:
        self.map_condition_encoder = nn.Sequential(
            LinearLayer(self.dim_target, self.x_dim**2, act_cls=self.act_cls),
            View(-1, 1, self.x_dim, self.x_dim),
        )
        self.initial_convolution = nn.Conv2d(self.n_channels + 1, self.h_dims[0], 1, bias=False)
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

        self.final_inference = ToLatentConvADAIN(self.h_dims[-1], self.final_z_dim, self.final_res, 0)

    def _build_decoder_layers(self) -> None:
        self.map_condition_decoder = LinearLayer(
            self.dim_target, ((self.final_res + 1) // 2) ** 2, act_cls=self.act_cls
        )
        self.h = nn.parameter.Parameter(torch.ones(1, self.h_dims[-1], self.final_res, self.final_res))
        self.back_to_img = FromLatentConv(self.h_dims[-1], self.aux_dim, self.final_res)
        self.decode_list = nn.ModuleList([])
        self.inference_list = nn.ModuleList([])
        self.generate_list = nn.ModuleList([])
        self.from_latent_list = nn.ModuleList([])
        self.merge = nn.ModuleList([])
        aux_dim = self.final_z_dim + (self.dim_target > 0)
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
                generate = ToLatentConvADAIN(h_out, z_dim, res, aux_dim)
                merge = ConvBlock(2 * h_out, h_out)
                inference = ToLatentConvADAIN(h_out, z_dim, res)
                self.inference_list.append(inference)
                self.generate_list.append(generate)
                self.from_latent_list.append(from_latent)
                self.merge.append(merge)
            else:
                self.inference_list.append(NullOp())
                self.generate_list.append(NullOp())
                self.from_latent_list.append(NullOp())
                self.merge.append(NullOp())

            aux_dim = z_dim or 0

        self.final_convolution = ConvBlock(self.h_dims[0], self.n_channels, use_trunc_init=True)
        return

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

    def encode(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> Outputs:
        if condition is None:
            raise ValueError('condition must be provided for CVAE encoder')

        condition = self._process_condition(condition, dtype=x.dtype)
        cond_map = self.map_condition_encoder(condition)
        x_cond = torch.cat([x, cond_map], dim=1)
        out = Outputs()
        x = self.initial_convolution(x_cond)
        for encode in self.encoding:
            out.hidden.append(x)
            x = encode(x)

        mu, log_var = self.final_inference(x)
        z = self.sampling(mu, log_var)
        out.mu.append(mu)
        out.log_var.append(log_var)
        out.z.append(z)
        return out

    def decode(
        self,
        out: Outputs,
        condition: torch.Tensor | None = None,
        sample: torch.Tensor | None = None,
        posterior_weight: float = 1.0,
    ) -> Outputs:
        if condition is None:
            raise ValueError('condition must be provided for CVAE decoder')

        condition = self._process_condition(condition)
        cond_latent = self.map_condition_decoder(condition)
        out.prior_mu = []
        out.prior_log_var = []
        z = out.z.pop() if sample is None else sample
        z = torch.cat([z, cond_latent], dim=1)
        h = self.h.expand(z.size(0), -1, -1, -1)
        x = self.back_to_img(z, h)
        for decode, from_latent, generate, inference, merge in zip(
            self.decode_list, self.from_latent_list, self.generate_list, self.inference_list, self.merge, strict=False
        ):
            x = decode(x)
            h = out.hidden.pop() if sample is None and out.hidden else None
            if h is not None:
                out.hidden.insert(0, h)

            if not isinstance(from_latent, NullOp):
                p_mu, p_logvar = generate(x, z)
                if sample is None and h is not None:
                    h = merge(torch.cat([x, h], dim=1))
                    mu, log_var = inference(h)
                    z = self.sampling(posterior_weight * mu + p_mu, log_var + p_logvar)
                    out.mu.append(mu)
                    out.log_var.append(log_var)
                else:
                    z = self.sampling(p_mu, p_logvar)

                x = from_latent(z, x)
                out.prior_mu.append(p_mu)
                out.prior_log_var.append(p_logvar)
                posterior_weight *= posterior_weight

        out.recon = torch.sigmoid(self.final_convolution(x))
        return out

    def forward(self, inputs: Inputs, **kwargs) -> Outputs:
        if not isinstance(inputs, Inputs):
            raise TypeError('CVAE requires Inputs dataclass with label')

        x = inputs.image
        condition = inputs.condition
        if condition is None or condition.numel() == 0:
            raise ValueError('label MUST be in Inputs else fail')

        out = self.encode(x, condition=condition)
        out = self.decode(out, condition=condition)
        out.image = x
        out.condition = self._process_condition(condition, dtype=x.dtype)
        return out

    @override
    def sample_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Sample uniform random class."""
        device = next(self.parameters()).device
        random_classes = torch.randint(0, self.dim_target, (batch_size,), device=device)
        return nn.functional.one_hot(random_classes, num_classes=self.dim_target).float()

    def generate_counterfactual(
        self,
        inputs: Inputs,
        target_dim: int = 0,
        target_value: float = 1.0,
        posterior_weight: float | None = None,
    ) -> Outputs:
        if posterior_weight is None:
            posterior_weight = Experiment.get_config().autoencoder.objective.posterior_weight

        if inputs.condition is None or inputs.condition.numel() == 0:
            raise ValueError('label MUST be in Inputs else fail')

        x = inputs.image
        condition = inputs.condition

        condition = self._process_condition(condition, dtype=x.dtype)
        out = self.encode(x, condition=condition)

        # Interpolate the condition towards the target class (one-hot vector)
        target = torch.zeros_like(condition)
        target[:, target_dim] = 1.0
        cf_condition = (1.0 - target_value) * condition + target_value * target

        out = self.decode(out, condition=cf_condition, posterior_weight=posterior_weight)
        out.image = x
        out.condition = cf_condition
        return out


class VAEX(CVAE):
    """Conditional VAE with input attribute map (fc0) and top latent attribute map (fc1).
    Conditions on the classifier logits using temperature scaling.
    """

    temperature: float

    def __init__(self) -> None:
        super().__init__()
        exp_cfg = Experiment.get_config()
        self.temperature = exp_cfg.autoencoder.model.temperature
        return

    @override
    def forward(self, inputs: Inputs, **kwargs) -> Outputs:
        if not isinstance(inputs, Inputs):
            raise TypeError('VAEX requires Inputs dataclass with logits')

        if inputs.logits is None or inputs.logits.numel() == 0:
            raise ValueError('logits MUST be in Inputs else fail')

        # Scale logits by temperature and convert to probability distribution
        condition = torch.sigmoid(inputs.logits / self.temperature)
        inputs = inputs._replace(condition=condition)
        return super().forward(inputs, **kwargs)

    @override
    def generate_counterfactual(
        self,
        inputs: Inputs,
        target_dim: int = 0,
        target_value: float = 1.0,
        posterior_weight: float | None = None,
    ) -> Outputs:
        if inputs.logits is None or inputs.logits.numel() == 0:
            raise ValueError('logits MUST be in Inputs else fail')

        condition = torch.sigmoid(inputs.logits / self.temperature)
        inputs = inputs._replace(condition=condition)

        return super().generate_counterfactual(
            inputs, target_dim=target_dim, target_value=target_value, posterior_weight=posterior_weight
        )

    @override
    def sample_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Sample random probabilities from Dirichlet distribution."""
        device = next(self.parameters()).device
        alpha = torch.ones(self.dim_target, device=device)
        return torch.distributions.Dirichlet(concentration=alpha).sample((batch_size,))


def get_autoencoder() -> Abstract_VAE:
    """Instantiate CVAE / VAEX model architecture based on autoencoder config selection."""
    cfg = Experiment.get_config()
    cls_name = cfg.autoencoder.model.class_name
    _REGISTRY: dict[Models, type[Abstract_VAE]] = {
        Models.CVAE: CVAE,
        Models.VAEX: VAEX,
    }
    cls = _REGISTRY.get(cls_name, VAEX)
    return cls()
