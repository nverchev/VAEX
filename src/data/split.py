"""Dataset split enumeration and dataset wrappers."""

from __future__ import annotations

import enum

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.structures import Inputs, Targets


class Partitions(enum.Enum):
    """Named dataset splits."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class CelebaDataset(Dataset[tuple[Inputs, Targets]]):
    """CelebA dataset split container."""

    def __init__(
        self,
        images: torch.Tensor,
        metadata: pd.DataFrame,
        target_attr: str,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.target_attr = target_attr
        self._images = images
        if self.target_attr in self.metadata:
            self.attr_is_true = torch.tensor(
                self.metadata[self.target_attr].values == 1, dtype=torch.long
            )
        else:
            raise ValueError(f'target_attr {self.target_attr!r} not in dataset metadata')

        return

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        target = self.attr_is_true[index]
        img_tensor = self._images[index].to(torch.float32).mul_(1.0 / 255.0)
        return Inputs(image=img_tensor, condition=target), Targets(label=target, ref=img_tensor)


