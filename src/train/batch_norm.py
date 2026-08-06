"""Re-estimation of the batch normalization statistics of a model."""

import torch

from drytorch.lib import runners
from typing import override

from src.data.structures import Inputs, Outputs, Targets

AbstractBatchNorm = torch.nn.modules.batchnorm._BatchNorm


class BatchNormUpdater(runners.ModelRunner[Inputs, Targets, Outputs]):
    """Update the running statistics in the batch normalization layers.

    Training accumulates those statistics on the path the training loop takes, so a model
    whose call runs a different path normalises its activations with the wrong ones. A
    pass over the loader re-estimates them on whatever path the model itself defines, as
    a cumulative average over the whole loader.
    """

    @override
    @torch.no_grad()
    def __call__(self, store_outputs: bool = False) -> None:
        """Single pass on the dataset."""
        bn_modules = [
            module
            for module in self.model.module.modules()
            if isinstance(module, AbstractBatchNorm)
        ]
        if not bn_modules:
            return

        previous = [(module, module.momentum, module.training) for module in bn_modules]
        for bn_module in bn_modules:
            bn_module.reset_running_stats()
            bn_module.momentum = None  # cumulative average over the pass
            # Only the normalization layers switch mode, so that everything else keeps
            # behaving the way it does at inference while the statistics are measured.
            bn_module.train()

        try:
            super().__call__(store_outputs)
        finally:
            for bn_module, momentum, was_training in previous:
                bn_module.momentum = momentum
                bn_module.train(was_training)

        return
