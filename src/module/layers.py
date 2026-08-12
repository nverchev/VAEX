"""Neural network layer building blocks."""

import abc
import functools
import itertools
import math
import sys
import warnings

from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar, cast, override, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config.torch import DEFAULT_ACT, DEFAULT_NORM
from src.config.torch import ActClass, NormClass

DEBUG_MODE = any(module in sys.modules for module in ('pydevd', 'debugpy', 'pdb')) or sys.gettrace() is not None

type _grad_t = tuple[torch.Tensor, ...] | torch.Tensor


@runtime_checkable
class HasInplace(Protocol):
    """Protocol for classes that have an inplace option."""

    inplace: bool


@runtime_checkable
class HasResetParamPublic(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""


@runtime_checkable
class HasResetParamPrivate(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def _reset_parameters(self) -> None:
        """Reset the parameters of the module."""


@runtime_checkable
class Weighted(Protocol):
    """Protocol for classes that have a weight attribute."""

    weight: torch.Tensor
    bias: torch.Tensor | None


@runtime_checkable
class ToLatentBlock(Protocol):
    """Protocol for inference/generate modules projecting feature maps to latent parameters (mu, log_var)."""

    def __call__(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project input x, an optional aux latent and an optional condition to mu and log_var."""
        ...



class View(nn.Module):
    """A simple module that reshapes the input tensor according to the specified shape."""

    shape: tuple[int, ...]

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.shape)


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class NullOp(nn.Module):
    """No-operation placeholder module."""

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(args) == 1:
            return args[0]

        return args


BatchAct = DEFAULT_ACT
BatchNorm = DEFAULT_NORM


def reverse_zip(*args: Sequence) -> zip:
    """Zip sequences in reverse order."""
    rev_args = [arg[::-1] for arg in args]
    return zip(*rev_args, strict=False)


def soft_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Smoothly bound `x` to (min_val, max_val); unlike torch.clamp, gradient never vanishes."""
    center = (min_val + max_val) / 2
    half_range = (max_val - min_val) / 2
    return center + half_range * torch.tanh((x - center) / half_range)


class BaseLayer(nn.Module, metaclass=abc.ABCMeta):
    """A class that includes a layer followed by an optional activation and grouped normalization."""

    in_dim: int
    out_dim: int
    n_groups_layer: int
    use_bias: bool
    use_trunc_init: bool
    init_weight_fn: Callable[[torch.Tensor], torch.Tensor]
    init_bias_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        n_groups_layer: int = 1,
        use_trunc_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups_layer = n_groups_layer
        self.use_bias = True if norm_cls is None else False
        self.module: nn.Module = self.get_module()
        self.norm: nn.Module | None = self.get_norm_layer(norm_cls, out_dim)
        self.act: nn.Module | None = self.get_activation(act_cls)
        self.use_trunc_init = use_trunc_init
        self.init_weight_fn = self.get_init_weight(self.act, use_trunc_init)
        self.init_bias_fn = self.get_init_bias(use_trunc_init)
        self.reset_parameters()
        if DEBUG_MODE:
            warnings.filterwarnings('ignore', message='.*Full backward hook is firing.*')
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    def reset_parameters(self) -> None:
        self.init_weight_fn(cast(torch.Tensor, self.module.weight))
        if self.norm is None:
            self.init_bias_fn(cast(torch.Tensor, self.module.bias))
            return

        _reset_parameters_or_warn(self.norm)
        self.init_bias_fn(cast(torch.Tensor, self.norm.bias))
        return

    @abc.abstractmethod
    def get_module(self) -> nn.Module:
        """Get wrapped module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.module(x)
        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

        return y

    @staticmethod
    def get_activation(act_cls: ActClass | None) -> nn.Module | None:
        if act_cls is None:
            return None

        act = act_cls()
        if isinstance(act, HasInplace):
            act.inplace = True

        return act

    @staticmethod
    def get_init_weight(act: nn.Module | None, use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        if use_trunc_init:
            return functools.partial(nn.init.trunc_normal_, mean=0.0, std=0.02, a=-0.04, b=0.04)

        if act is None:
            return functools.partial(nn.init.xavier_normal_, gain=1)

        if isinstance(act, nn.Hardtanh):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('tanh'))

        if isinstance(act, nn.Sigmoid):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('sigmoid'))

        if isinstance(act, nn.ReLU):
            return functools.partial(nn.init.kaiming_normal_, nonlinearity='relu')

        if isinstance(act, nn.LeakyReLU):
            return functools.partial(nn.init.kaiming_normal_, a=act.negative_slope, nonlinearity='leaky_relu')

        if isinstance(act, nn.GELU):
            return functools.partial(nn.init.xavier_normal_, gain=1)

        return functools.partial(nn.init.xavier_normal_, gain=1)

    @staticmethod
    def get_init_bias(use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.nn.init.zeros_ if use_trunc_init else lambda x: x

    @staticmethod
    def get_norm_layer(norm_cls: NormClass | None, out_dim: int) -> nn.Module | None:
        if norm_cls is None:
            return None

        return norm_cls(out_dim)


class LinearLayer(BaseLayer):
    """Linear layer block."""

    @override
    def get_module(self) -> nn.Module:
        if self.n_groups_layer > 1:
            raise NotImplementedError('nn.Linear has no option for groups')

        return nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)


class Conv2dLayer(BaseLayer):
    """2D convolutional layer built on BaseLayer for proper weight initialisation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        use_trunc_init: bool = False,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        super().__init__(in_dim, out_dim, act_cls=act_cls, norm_cls=norm_cls, use_trunc_init=use_trunc_init)
        return

    @override
    def get_module(self) -> nn.Module:
        return nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.use_bias,
            groups=self.groups,
        )


class ConvTranspose2dLayer(BaseLayer):
    """2D transposed convolutional layer built on BaseLayer for proper weight initialisation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        use_trunc_init: bool = False,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        super().__init__(in_dim, out_dim, act_cls=act_cls, norm_cls=norm_cls, use_trunc_init=use_trunc_init)
        return

    @override
    def get_module(self) -> nn.Module:
        return nn.ConvTranspose2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.use_bias,
            groups=self.groups,
        )


Layer = TypeVar('Layer', bound=BaseLayer)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block for 4D feature maps."""

    def __init__(self, h_dim: int, reduction_ratio: float = 0.1) -> None:
        super().__init__()
        h_reduced = max(1, int(h_dim * reduction_ratio))
        self.mlp = nn.Sequential(
            LinearLayer(h_dim, h_reduced, act_cls=nn.LeakyReLU),
            LinearLayer(h_reduced, h_dim, act_cls=nn.Sigmoid),
        )
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.mean((2, 3))
        x = self.mlp(x).unsqueeze(-1).unsqueeze(-1)
        return inputs.mul(x)


class ConvBlock(nn.Module):
    """Post-activation convolutional block: Conv2dLayer → norm → act.

    Args:
        use_trunc_init: if True, skips norm/act to keep a near-zero output.
    """

    def __init__(
        self,
        h_in: int,
        h_out: int,
        deconv: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_trunc_init: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        LayerCls = ConvTranspose2dLayer if deconv else Conv2dLayer
        self.block = LayerCls(
            h_in,
            h_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_cls=None if use_trunc_init else BatchAct,
            norm_cls=None if use_trunc_init else BatchNorm,
            use_trunc_init=use_trunc_init,
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileBlock(nn.Module):
    """Mobile inverted bottleneck block."""

    def __init__(
        self,
        h_in: int,
        h_out: int,
        deconv: bool = False,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        h_exp = h_in * expansion_factor
        self.stride = stride
        LayerCls = ConvTranspose2dLayer if deconv else Conv2dLayer
        self.block = nn.Sequential(
            Conv2dLayer(
                h_in,
                h_exp,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                act_cls=BatchAct,
                norm_cls=BatchNorm,
                ),
            LayerCls(
                h_exp,
                h_exp,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=h_exp,
                act_cls=BatchAct,
                norm_cls=BatchNorm,
                ),
            Conv2dLayer(h_exp, h_out, kernel_size=1, stride=1, padding=0, groups=1, act_cls=None, norm_cls=BatchNorm),
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResLayer(nn.Module):
    """Residual layer wrapping a block with Squeeze-and-Excitation."""

    def __init__(self, Block: type[nn.Module], h_in: int) -> None:
        super().__init__()
        self.block = Block(h_in, h_in, use_trunc_init=True)
        self.se = SqueezeExcite(h_in)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se(self.block(x))


class ResBlock(nn.Module):
    """Residual block supporting multiple depths, always adding a resampled skip connection."""

    def __init__(
        self,
        Block: type[nn.Module],
        h_in: int,
        h_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        deconv: bool = False,
        depth: int = 0,
    ) -> None:
        super().__init__()
        modules = []
        self.stride = stride
        self.deconv = deconv
        self.h_in = h_in
        self.h_out = h_out
        self.se = SqueezeExcite(h_out)

        for _ in range(depth):
            modules.append(ResLayer(Block, h_in))

        modules.append(Block(h_in, h_out, deconv=deconv, kernel_size=kernel_size, stride=stride, padding=padding))
        self.outer_block = nn.Sequential(*modules)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.outer_block(x)
        output_size = y.size()
        if self.stride != 1:
            x = F.interpolate(x, size=output_size[2:], mode='bilinear', align_corners=False)

        if self.h_out < self.h_in:
            x = x.view(output_size[0], output_size[1], 2, output_size[2], output_size[3]).mean(2)
        elif self.h_out > self.h_in:
            x = x.repeat(1, 2, 1, 1)

        return x + self.se(y)


class FromLatentConv(nn.Module):
    """Module converting top latent variables back into spatial feature channels.

    Uses truncated-normal init on the deconv so the initial reconstruction
    contribution is near zero, letting the decoder warm up gradually.
    """

    def __init__(self, h_dim: int, z_dim: int, l_dim: int) -> None:
        super().__init__()
        self.toimage = nn.Sequential(
            View(-1, z_dim, (l_dim + 1) // 2, (l_dim + 1) // 2),
            ConvTranspose2dLayer(z_dim, h_dim, kernel_size=3, stride=2, padding=1, use_trunc_init=True),
            SqueezeExcite(h_dim),
        )
        return

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = self.toimage(z)
        return x + z


class ToLatentConvADAIN(nn.Module):
    """Inference block using adaptive instance normalisation.

    The latent projection (tolatent) uses truncated-normal init so mu/logvar
    start near zero for training stability.
    """

    def __init__(self, h_dim: int, z_dim: int, l_dim: int, aux_dim: int = 0, cond_dim: int = 0) -> None:
        super().__init__()
        if aux_dim:
            self.aux_conv = nn.Sequential(
                nn.Linear(aux_dim * ((l_dim + 3) // 4) ** 2, 2 * h_dim),
                View(-1, 2 * h_dim, 1, 1),
            )

        if cond_dim:
            self.cond_conv = nn.Sequential(
                LinearLayer(cond_dim, 2 * h_dim, use_trunc_init=True),
                View(-1, 2 * h_dim, 1, 1),
            )

        self.tolatent = Conv2dLayer(h_dim, 2 * z_dim, kernel_size=3, stride=2, padding=1, use_trunc_init=True)
        return

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if aux is not None:
            mu, std = self.aux_conv(aux).chunk(2, 1)
            x = F.instance_norm(x) * torch.sigmoid(std) + mu

        if condition is not None:
            scale, shift = self.cond_conv(condition).chunk(2, 1)
            x = x * (1 + scale) + shift

        x = self.tolatent(x)
        return x.flatten(1).chunk(2, 1)


class ToLatentConvMobile(nn.Module):
    """Inference block using MobileBlock and concatenation."""

    def __init__(self, h_dim: int, z_dim: int, l_dim: int, aux_dim: int = 0, cond_dim: int = 0) -> None:
        super().__init__()
        if aux_dim:
            self.aux_conv = nn.Sequential(
                View(-1, aux_dim, (l_dim + 3) // 4, (l_dim + 3) // 4),
                MobileBlock(aux_dim, aux_dim, deconv=True, kernel_size=5, stride=4, padding=2),
                SqueezeExcite(aux_dim),
            )
        else:
            self.aux_conv = None

        if cond_dim:
            self.cond_conv = nn.Sequential(
                LinearLayer(cond_dim, 2 * (h_dim + aux_dim), use_trunc_init=True),
                View(-1, 2 * (h_dim + aux_dim), 1, 1),
            )

        self.tolatent = Conv2dLayer(h_dim + aux_dim, 2 * z_dim, kernel_size=3, stride=2, padding=1, use_trunc_init=True)
        return

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if aux is not None and self.aux_conv is not None:
            aux = self.aux_conv(aux)
            x = torch.cat([x, aux], dim=1)

        if condition is not None:
            scale, shift = self.cond_conv(condition).chunk(2, 1)
            x = x * (1 + scale) + shift

        x = self.tolatent(x)
        return x.flatten(1).chunk(2, 1)


class ProjectionLayer(nn.Module):
    """A layer that projects the input to a different dimension."""

    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        return

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.in_dim:
            raise ValueError(f'Input dimension {x.shape[1]} does not match expected dimension {self.in_dim}')

        if self.in_dim == self.out_dim:
            return x

        if self.in_dim > self.out_dim:
            out = torch.cat([t.mean(1, keepdim=True) for t in torch.chunk(x, self.out_dim, dim=1)], dim=1)
        else:
            out = x.repeat_interleave(self.out_dim // self.in_dim + 1, 1)[:, : self.out_dim, ...]

        var_adjust = math.sqrt(self.in_dim / self.out_dim)
        return out * var_adjust


class BaseResBlock(nn.Module, Generic[Layer], abc.ABC):
    """A block consisting of a sequence of layers."""

    dims: Sequence[int]
    act_cls: ActClass | None
    norm_cls: NormClass | None
    n_groups_layer: int

    def __init__(
        self,
        dims: Sequence[int],
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        n_groups_layer: int = 1,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.act_cls = act_cls
        self.norm_cls = norm_cls
        self.n_groups_layer = n_groups_layer
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        for in_dim, out_dim in itertools.pairwise(dims):
            self.layers.append(self.get_layer(in_dim, out_dim, act_cls, norm_cls, n_groups_layer))
            if in_dim == out_dim:
                self.projections.append(nn.Identity())
            else:
                self.projections.append(self.get_projection(in_dim, out_dim))

        self._adjust_variance_init()
        return

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = self.projections[i](x) + layer(x)

        return x

    def reset_parameters(self) -> None:
        for layer in self.layers:
            _reset_parameters_or_warn(layer)

        self._adjust_variance_init()
        return

    @classmethod
    @abc.abstractmethod
    def get_layer(
        cls, in_dim: int, out_dim: int, act_cls: ActClass | None, norm_cls: NormClass | None, n_groups_layer: int
    ) -> Layer:
        """Get layer instance."""

    @classmethod
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        return ProjectionLayer(in_dim, out_dim)

    def _adjust_variance_init(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer.module, Weighted):
                    layer.module.weight.mul_(1 / math.sqrt(len(self.dims) - 1))
                    if layer.module.bias is not None:
                        layer.module.bias.mul_(0)

        return


class LinearResBlock(BaseResBlock[LinearLayer]):
    """Residual linear block."""

    @classmethod
    @override
    def get_layer(
        cls, in_dim: int, out_dim: int, act_cls: ActClass | None, norm_cls: NormClass | None, n_groups_layer: int
    ) -> LinearLayer:
        return LinearLayer(in_dim, out_dim, act_cls, norm_cls, n_groups_layer)


def debug_check(_not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t) -> None:
    """Debug hook to check for NaN or Inf values in the output tensor."""

    import pdb

    if isinstance(tensor_out, tuple):
        tensor = tensor_out[0]
    else:
        tensor = tensor_out

    if torch.any(torch.isnan(tensor)):
        breakpoint()
        pdb.set_trace()
    elif torch.any(torch.isinf(tensor)):
        breakpoint()
        pdb.set_trace()

    return None


def _reset_parameters_or_warn(module: nn.Module) -> None:
    if isinstance(module, HasResetParamPublic):
        module.reset_parameters()
        return

    if isinstance(module, HasResetParamPrivate):
        module._reset_parameters()
        return

    if isinstance(module, nn.Module):
        warnings.warn(f'Module {module.__class__.__name__} does not implement reset_parameters().', stacklevel=3)

    return


def reset_parameters_recursive(module: nn.Module, warn: bool = False) -> None:
    if isinstance(module, HasResetParamPublic):
        module.reset_parameters()
        return

    if isinstance(module, HasResetParamPrivate):
        module._reset_parameters()
        return

    has_params = any(True for _ in module.parameters(recurse=False))
    has_children = any(True for _ in module.children())
    if has_params and not has_children and warn:
        warnings.warn(
            f'Leaf module {module.__class__.__name__} has parameters but does not implement reset_parameters().',
            stacklevel=2,
        )
    for child in module.children():
        reset_parameters_recursive(child, warn=warn)

    return


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple:
        grad_output = grad_outputs[0]
        return -ctx.lambda_ * grad_output, None


class InformationEraser(nn.Module):
    """Predict logits but reverse the gradient to erase information with the target."""

    def __init__(self, in_dim: int, n_classes: int, reversal_strength: float = 0.5, intermediate_dim: int = 512) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.intermediate_dim: int = intermediate_dim
        self.n_classes: int = n_classes
        self.reversal_strength = reversal_strength
        self.warmup_scale = 1.0
        self.mlp = nn.Sequential(
            LinearLayer(self.in_dim, self.intermediate_dim, act_cls=nn.ReLU),
            LinearLayer(self.intermediate_dim, self.n_classes),
        )
        return

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        reversed_features = GradReverse.apply(features, self.reversal_strength * self.warmup_scale)
        return self.mlp(reversed_features)
