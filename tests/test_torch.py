"""Tests for the torch config utilities."""

import pytest
import torch

import torch.nn as nn

from src.config.torch import get_device, get_mixed_precision, get_norm_cls


@pytest.mark.parametrize('device', ['cpu', torch.device('cpu')])
def test_get_device(device: str | torch.device) -> None:
    """A device is resolved by name or as a torch.device."""
    resolved = get_device(device)

    assert resolved == torch.device('cpu')


def test_get_device_auto() -> None:
    """'auto' is kept as is."""
    assert get_device('auto') == 'auto'


def test_get_device_raises() -> None:
    """An invalid device name raises ValueError."""
    with pytest.raises(ValueError, match=r'not auto or the name of a device'):
        get_device('invalid_device_name_123')


@pytest.mark.parametrize('precision', ['bfloat16', torch.bfloat16])
def test_get_mixed_precision(precision: str | torch.dtype) -> None:
    """The autocast precision is resolved by name or as a dtype."""
    dtype = get_mixed_precision(precision)

    assert dtype is torch.bfloat16


def test_get_mixed_precision_none() -> None:
    """None, which trains in full precision, is kept."""
    dtype = get_mixed_precision(None)

    assert dtype is None


@pytest.mark.parametrize('precision', ['float32', torch.float32, 'half'])
def test_get_mixed_precision_raises(precision: str | torch.dtype) -> None:
    """A precision other than float16 and bfloat16 raises ValueError."""
    with pytest.raises(ValueError, match='not float16 or bfloat16'):
        get_mixed_precision(precision)


@pytest.mark.parametrize('norm', ['BatchNorm2d', nn.BatchNorm2d])
def test_get_norm_cls(norm: str | type[nn.Module]) -> None:
    """The normalization class is resolved by name or class."""
    norm_cls = get_norm_cls(norm)

    assert norm_cls is nn.BatchNorm2d


def test_get_norm_cls_raises() -> None:
    """An unknown normalization class raises ValueError."""
    with pytest.raises(ValueError, match=r'not the name of a pytorch nn\.Module'):
        get_norm_cls('UnknownNorm')
