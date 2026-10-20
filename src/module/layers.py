"""Blocks of the hierarchical VAE."""

import functools
from collections.abc import Callable
from typing import Final, Protocol, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.torch import ActClass, NormClass

__all__ = [
    'ConvBlock',
    'FromLatentConv',
    'MobileBlock',
    'NormFactory',
    'ResBlock',
    'ResLayer',
    'ResampleSkip',
    'SqueezeExcite',
    'ToLatentBlock',
    'ToLatentConvADAIN',
    'ToLatentConvMobile',
    'get_batch_norm',
    'soft_clamp',
]

type NormFactory = Callable[[int], nn.Module]


class ToLatentBlock(Protocol):
    """Block giving the mean and the log-variance of a latent from features."""

    def __call__(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and the log-variance of the latent, flattened.

        Args:
            x: features at the resolution of the stage.
            aux: latent of the stage above, flattened, for the prior.
        """
        ...


def get_batch_norm(momentum: float, eps: float, norm_cls: NormClass = nn.BatchNorm2d) -> NormFactory:
    """Get batch normalization layers with the given momentum and epsilon.

    Args:
        momentum: momentum of the running statistics.
        eps: value added to the variance.
        norm_cls: class of the normalization layer.

    Returns:
        A function from the number of channels to the layer.
    """
    return functools.partial(norm_cls, eps=eps, momentum=momentum)


def soft_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Smoothly bound ``x`` to ``(min_val, max_val)``; unlike ``torch.clamp``, keep a gradient near the bounds.

    Args:
        x: the values to bound.
        min_val: the lower bound.
        max_val: the upper bound.

    Returns:
        The bounded values.
    """
    center = (min_val + max_val) / 2
    half_range = (max_val - min_val) / 2
    return center + half_range * torch.tanh((x - center) / half_range)


class SqueezeExcite(nn.Module):
    """Rescale each channel by a gate computed from all channels.

    Attributes:
        gate: global average pooling, then a bottleneck ending in a sigmoid.
    """

    gate: Final[nn.Sequential]

    def __init__(self, h_dim: int, reduction_ratio: float = 0.1) -> None:
        """Initialize.

        Args:
            h_dim: channels of the features.
            reduction_ratio: width of the bottleneck relative to the channels.
        """
        super().__init__()
        h_reduced = max(1, int(h_dim * reduction_ratio))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(h_dim, h_reduced),
            nn.ReLU(),
            nn.Linear(h_reduced, h_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (h_dim, 1, 1)),
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply each channel by its gate."""
        return x * self.gate(x)


class ConvBlock(nn.Sequential):
    """Pre-activation convolution: normalization, activation, convolution without bias."""

    def __init__(
        self,
        h_in: int,
        h_out: int,
        *,
        act_cls: ActClass,
        norm: NormFactory,
        deconv: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        weight_norm: bool = False,
    ) -> None:
        """Initialize.

        Args:
            h_in: channels of the input.
            h_out: channels of the output.
            act_cls: class of the activation.
            norm: normalization of the input channels.
            deconv: whether the convolution is transposed.
            kernel_size: kernel size of the convolution.
            stride: stride of the convolution.
            padding: padding of the convolution.
            weight_norm: whether the convolution has weight normalization.
        """
        conv_cls = nn.ConvTranspose2d if deconv else nn.Conv2d
        conv = conv_cls(h_in, h_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if weight_norm:
            conv = nn.utils.parametrizations.weight_norm(conv)

        super().__init__(norm(h_in), act_cls(), conv)
        return


class MobileBlock(nn.Sequential):
    """Mobile inverted bottleneck: expansion, depthwise convolution, projection."""

    def __init__(
        self,
        h_in: int,
        h_out: int,
        *,
        act_cls: ActClass,
        norm: NormFactory,
        deconv: bool = False,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        expansion_factor: int = 4,
    ) -> None:
        """Initialize.

        Args:
            h_in: channels of the input.
            h_out: channels of the output.
            act_cls: class of the activation.
            norm: normalization layers.
            deconv: whether the depthwise convolution is transposed.
            kernel_size: kernel size of the depthwise convolution.
            stride: stride of the depthwise convolution.
            padding: padding of the depthwise convolution.
            expansion_factor: width of the expansion relative to the input channels.
        """
        h_exp = h_in * expansion_factor
        conv_cls = nn.ConvTranspose2d if deconv else nn.Conv2d
        super().__init__(
            norm(h_in),
            act_cls(),
            nn.Conv2d(h_in, h_exp, 1, 1, bias=False),
            norm(h_exp),
            act_cls(),
            conv_cls(h_exp, h_exp, kernel_size, stride=stride, padding=padding, groups=h_exp, bias=False),
            norm(h_exp),
            act_cls(),
            nn.Conv2d(h_exp, h_out, 1, 1, bias=False),
            norm(h_out),
        )
        return


class ResLayer(nn.Module):
    """Convolution at constant width, added to its input after Squeeze-and-Excitation.

    Attributes:
        block: the convolution.
        se: the Squeeze-and-Excitation.
    """

    block: Final[ConvBlock]
    se: Final[SqueezeExcite]

    def __init__(self, h_dim: int, *, act_cls: ActClass, norm: NormFactory) -> None:
        """Initialize.

        Args:
            h_dim: channels of the features.
            act_cls: class of the activation.
            norm: normalization layers.
        """
        super().__init__()
        self.block = ConvBlock(h_dim, h_dim, act_cls=act_cls, norm=norm)
        self.se = SqueezeExcite(h_dim)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the gated convolution to the input."""
        return x + self.se(self.block(x))


class ResampleSkip(nn.Module):
    """Bring the input of a residual block to the shape of its output.

    The input is resized when the block changes the resolution, and its channels are averaged
    in pairs or repeated when the block halves or doubles them.

    Attributes:
        h_in: channels of the input.
        h_out: channels of the output.
        stride: stride of the block.
    """

    h_in: Final[int]
    h_out: Final[int]
    stride: Final[int]

    def __init__(self, h_in: int, h_out: int, stride: int) -> None:
        """Initialize.

        Args:
            h_in: channels of the input.
            h_out: channels of the output.
            stride: stride of the block.
        """
        super().__init__()
        self.h_in = h_in
        self.h_out = h_out
        self.stride = stride
        return

    def forward(self, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Reshape the input ``x`` as the output ``like`` of the block."""
        output_size = like.size()
        if self.stride != 1:
            x = F.interpolate(x, size=output_size[2:], mode='bilinear', align_corners=False)

        if self.h_out < self.h_in:
            x = x.view(output_size[0], output_size[1], 2, output_size[2], output_size[3]).mean(2)
        elif self.h_out > self.h_in:
            x = x.repeat(1, 2, 1, 1)

        return x

    @override
    def extra_repr(self) -> str:
        return f'h_in={self.h_in}, h_out={self.h_out}, stride={self.stride}'


class ResBlock(nn.Module):
    """Residual layers and a change of resolution, added to the resampled input.

    Attributes:
        layers: the residual layers, then the convolution changing the resolution.
        se: the Squeeze-and-Excitation of the output.
        skip: the connection carrying the input to the shape of the output.
    """

    layers: Final[nn.Sequential]
    se: Final[SqueezeExcite]
    skip: Final[ResampleSkip]

    def __init__(
        self,
        h_in: int,
        h_out: int,
        *,
        act_cls: ActClass,
        norm: NormFactory,
        kernel_size: int,
        stride: int,
        padding: int,
        deconv: bool = False,
        depth: int = 0,
    ) -> None:
        """Initialize.

        Args:
            h_in: channels of the input.
            h_out: channels of the output.
            act_cls: class of the activation.
            norm: normalization layers.
            kernel_size: kernel size of the change of resolution.
            stride: stride of the change of resolution.
            padding: padding of the change of resolution.
            deconv: whether the change of resolution is transposed.
            depth: number of residual layers before the change of resolution.
        """
        super().__init__()
        self.layers = nn.Sequential(*(ResLayer(h_in, act_cls=act_cls, norm=norm) for _ in range(depth)))
        self.layers.append(
            ConvBlock(
                h_in,
                h_out,
                act_cls=act_cls,
                norm=norm,
                deconv=deconv,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        self.se = SqueezeExcite(h_out)
        self.skip = ResampleSkip(h_in, h_out, stride)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the resampled input to the gated output of the layers."""
        y = self.layers(x)
        return self.skip(x, y) + self.se(y)


class FromLatentConv(nn.Module):
    """Upsample a latent to the features of its stage and add it to them.

    Attributes:
        to_features: unflattening, transposed convolution and Squeeze-and-Excitation.
    """

    to_features: Final[nn.Sequential]

    def __init__(self, h_dim: int, z_dim: int, l_dim: int, *, act_cls: ActClass, norm: NormFactory) -> None:
        """Initialize.

        Args:
            h_dim: channels of the features.
            z_dim: channels of the latent.
            l_dim: resolution of the features.
            act_cls: class of the activation.
            norm: normalization layers.
        """
        super().__init__()
        side = (l_dim + 1) // 2
        self.to_features = nn.Sequential(
            nn.Unflatten(1, (z_dim, side, side)),
            ConvBlock(z_dim, h_dim, act_cls=act_cls, norm=norm, stride=2, deconv=True),
            SqueezeExcite(h_dim),
        )
        return

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add the upsampled latent ``z`` to the features ``x``."""
        return x + self.to_features(z)


class ToLatentConvADAIN(nn.Module):
    """Latent from features, restyled by the latent above with adaptive instance normalization.

    Attributes:
        style: linear layer giving the shift and the scale of each channel from the latent above,
            when there is one.
        instance_norm: normalization of each channel of the features.
        to_latent: strided convolution giving mean and log-variance, flattened.
    """

    style: Final[nn.Sequential | None]
    instance_norm: Final[nn.InstanceNorm2d]
    to_latent: Final[nn.Sequential]

    def __init__(
        self, h_dim: int, z_dim: int, l_dim: int, aux_dim: int = 0, *, act_cls: ActClass, norm: NormFactory
    ) -> None:
        """Initialize.

        Args:
            h_dim: channels of the features.
            z_dim: channels of the latent.
            l_dim: resolution of the features.
            aux_dim: channels of the latent above; zero when there is none.
            act_cls: class of the activation.
            norm: normalization layers.
        """
        super().__init__()
        self.style = None
        if aux_dim:
            self.style = nn.Sequential(
                nn.Linear(aux_dim * ((l_dim + 3) // 4) ** 2, 2 * h_dim),
                nn.Unflatten(1, (2 * h_dim, 1, 1)),
            )

        self.instance_norm = nn.InstanceNorm2d(h_dim)
        self.to_latent = nn.Sequential(
            ConvBlock(h_dim, 2 * z_dim, act_cls=act_cls, norm=norm, stride=2, weight_norm=True),
            nn.Flatten(),
        )
        return

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and the log-variance of the latent, flattened."""
        if aux is not None and self.style is not None:
            shift, scale = self.style(aux).chunk(2, 1)
            x = self.instance_norm(x) * torch.sigmoid(scale) + shift

        mu, log_var = self.to_latent(x).chunk(2, 1)
        return mu, log_var


class ToLatentConvMobile(nn.Module):
    """Latent from features concatenated with the upsampled latent above (VAEXcat).

    Attributes:
        aux_conv: upsampling of the latent above with an inverted bottleneck, when there is one.
        to_latent: strided convolution giving mean and log-variance, flattened.
    """

    aux_conv: Final[nn.Sequential | None]
    to_latent: Final[nn.Sequential]

    def __init__(
        self, h_dim: int, z_dim: int, l_dim: int, aux_dim: int = 0, *, act_cls: ActClass, norm: NormFactory
    ) -> None:
        """Initialize.

        Args:
            h_dim: channels of the features.
            z_dim: channels of the latent.
            l_dim: resolution of the features.
            aux_dim: channels of the latent above; zero when there is none.
            act_cls: class of the activation.
            norm: normalization layers.
        """
        super().__init__()
        self.aux_conv = None
        if aux_dim:
            side = (l_dim + 3) // 4
            self.aux_conv = nn.Sequential(
                nn.Unflatten(1, (aux_dim, side, side)),
                MobileBlock(aux_dim, aux_dim, act_cls=act_cls, norm=norm, deconv=True, stride=4),
                SqueezeExcite(aux_dim),
            )

        self.to_latent = nn.Sequential(
            ConvBlock(h_dim + aux_dim, 2 * z_dim, act_cls=act_cls, norm=norm, stride=2, weight_norm=True),
            nn.Flatten(),
        )
        return

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and the log-variance of the latent, flattened."""
        if aux is not None and self.aux_conv is not None:
            x = torch.cat([x, self.aux_conv(aux)], dim=1)

        mu, log_var = self.to_latent(x).chunk(2, 1)
        return mu, log_var
