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
        if 'OMP_NUM_THREADS' not in os.environ:
            torch.set_num_threads(max(1, (os.cpu_count() or 0) // self.world_size))

        accelerator = torch.accelerator.current_accelerator(check_available=True)
        device_id: torch.device | None = None
        backend = dist.get_default_backend_for_device(torch.device('cpu'))
        if accelerator is not None:
            device_id = torch.device(accelerator.type, rank)
            torch.accelerator.set_device_index(rank)
            backend = dist.get_default_backend_for_device(device_id)

        dist.init_process_group(
            backend,
            device_id=device_id,
            init_method=f'tcp://127.0.0.1:{self.port}',
            rank=rank,
            world_size=self.world_size,
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
