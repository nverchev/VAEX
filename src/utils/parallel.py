"""Defines the settings for data-distributed parallelism."""

import os
import socket

from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

P = ParamSpec('P')
T = TypeVar('T')


class DistributedWorker[**P, T]:
    """Callable wrapper to distribute a worker across multiple processes."""

    def __init__(self, worker: Callable[P, T], world_size: int) -> None:
        """Initialize."""
        self.worker = worker
        self.world_size = world_size
        self.port = self._get_free_port()
        return

    def __call__(self, rank: int, *args: P.args, **kwargs: P.kwargs) -> None:
        """Run the worker in a distributed environment."""
        self._setup_distributed(rank)
        try:
            self.worker(*args, **kwargs)
        finally:
            self._cleanup_distributed()

        return

    def spawn(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Spawn the worker in multiple processes."""
        spawn(self, args=args, nprocs=self.world_size)
        return

    def _setup_distributed(self, rank: int) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        acc_or_none = torch.accelerator.current_accelerator()
        acc = torch.device('cpu') if acc_or_none is None else acc_or_none
        backend = torch.distributed.get_default_backend_for_device(acc)
        dist.init_process_group(
            backend, rank=rank, init_method=f'tcp://127.0.0.1:{self.port}', world_size=self.world_size
        )

        return

    @staticmethod
    def _get_free_port() -> str:
        """Find an available port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return str(s.getsockname()[1])

    @staticmethod
    def _cleanup_distributed() -> None:
        """Clean up the distributed environment."""
        dist.destroy_process_group()
        return
