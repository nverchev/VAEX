"""Dataset classes and factory for the training/eval split."""

import torch.distributed as dist

from src.config.experiment import Experiment
from src.data.celeba import CelebaDataBuilder
from src.data.databuilder import DataBuilder
from src.data.split import Partitions, DataSplit


def get_databuilder() -> DataBuilder:
    """Return the dataset builder instance."""
    return CelebaDataBuilder()


def get_split(partition: Partitions) -> DataSplit:
    """Return dataset instance for a given partition."""
    databuilder = get_databuilder()
    return databuilder.get_split(partition)


def _get_splits() -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    return get_split(train_partition), get_split(eval_partition)


def get_splits() -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets in a multiprocess-safe way."""
    cfg = Experiment.get_config()
    if not cfg.user.n_subprocesses:
        return _get_splits()

    rank = dist.get_rank()
    splits: tuple[DataSplit, DataSplit]
    for i in range(cfg.user.n_subprocesses):
        if rank == i:
            splits = _get_splits()

        dist.barrier()

    return splits
