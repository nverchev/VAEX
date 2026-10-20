"""Tests for the layer building blocks."""

import torch

from src.module.layers import ResampleSkip, soft_clamp


class TestResampleSkip:
    """Test the skip connection of the residual blocks."""

    def test_halving_the_channels_averages_them_in_pairs(self) -> None:
        """Each output channel is the mean of two consecutive input channels."""
        x = torch.randn(2, 4, 3, 3)
        like = torch.empty(2, 2, 3, 3)

        out = ResampleSkip(4, 2, stride=1)(x, like)

        torch.testing.assert_close(out, torch.stack([x[:, 0:2].mean(1), x[:, 2:4].mean(1)], dim=1))

    def test_doubling_the_channels_repeats_them(self) -> None:
        """The channels are repeated as a block."""
        x = torch.randn(2, 2, 3, 3)
        like = torch.empty(2, 4, 3, 3)

        out = ResampleSkip(2, 4, stride=1)(x, like)

        torch.testing.assert_close(out, x.repeat(1, 2, 1, 1))

    def test_a_stride_resizes_the_input_to_the_output(self) -> None:
        """With a stride, the input takes the resolution of the output."""
        x = torch.randn(2, 3, 8, 8)
        like = torch.empty(2, 3, 4, 4)

        out = ResampleSkip(3, 3, stride=2)(x, like)

        assert out.shape == like.shape


def test_the_representation_shows_the_settings() -> None:
    """The skip connection shows its settings in its representation."""
    text = repr(ResampleSkip(4, 2, stride=2))

    assert 'h_in=4, h_out=2, stride=2' in text


def test_soft_clamp() -> None:
    """Values stay within the bounds, and the gradient is positive between them."""
    x = torch.tensor([-100.0, -1.0, 0.0, 50.0], requires_grad=True)

    bounded = soft_clamp(x, -4.0, 2.0)
    bounded.sum().backward()

    assert bool(((bounded >= -4.0) & (bounded <= 2.0)).all())
    torch.testing.assert_close(soft_clamp(torch.tensor(-1.0), -4.0, 2.0), torch.tensor(-1.0))
    assert x.grad is not None
    assert bool((x.grad[1:3] > 0).all())
