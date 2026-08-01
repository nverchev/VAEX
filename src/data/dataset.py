"""Dataset classes and factory for the training/eval split."""

from torch.utils.data import Dataset

from src.config.experiment import Experiment
from src.data.celeba import CelebaDataBuilder
from src.data.databuilder import DataBuilder
from src.data.split import Partitions
from src.data.structures import Inputs, Targets


def get_databuilder() -> DataBuilder:
    """Return the dataset builder instance."""
    return CelebaDataBuilder()


def get_dataset(partition: Partitions) -> Dataset[tuple[Inputs, Targets]]:
    """Return dataset instance for a given partition."""
    databuilder = get_databuilder()
    return databuilder.get_split(partition)


def _get_datasets() -> tuple:
    """Get training and evaluation datasets."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    return get_dataset(train_partition), get_dataset(eval_partition)


def get_datasets() -> tuple:
    """Get training and evaluation datasets in a multiprocess-safe way."""
    cfg = Experiment.get_config()
    splits: tuple | None = None
    if cfg.user.n_subprocesses:
        import torch.distributed as dist

        rank = dist.get_rank()
        for i in range(cfg.user.n_subprocesses):
            if rank == i:
                splits = _get_datasets()

            dist.barrier() if cfg.user.cpu else dist.barrier(device_ids=[rank])
    else:
        splits = _get_datasets()

    if splits is None:
        raise RuntimeError('Splits could not be created.')

    return splits

