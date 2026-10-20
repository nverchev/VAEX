"""Profiling utilities for PyTorch models."""

import csv
import logging
import pathlib
from typing import Any

import torch
import torch.profiler

from torch.autograd import profiler_util

logger = logging.getLogger(__name__)


def log_parameter_counts(model: torch.nn.Module, profile_dir: pathlib.Path) -> None:
    """Log the total and trainable parameter counts of each module to ``parameters.csv``."""
    param_decomp = _get_parameter_decomposition(model)
    with open(profile_dir / 'parameters.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['module', 'total_parameters', 'trainable_parameters'])
        for name, total_p, trainable_p in param_decomp:
            writer.writerow([name, total_p, trainable_p])

    return


def profile_forward(
    model: torch.nn.Module,
    inputs: Any,
    device: torch.device,
    profile_dir: pathlib.Path,
) -> Any:
    """Profile the forward pass of the model and return the outputs."""
    model = model.to(device)
    model.train()
    for _ in range(3):
        _ = model(inputs)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        acc_events=True,
    ) as prof_forward:
        with torch.profiler.record_function('forward'):
            out = model(inputs)

    _log_profile(prof_forward.key_averages(), profile_dir / 'fwd_profile.csv', device)
    return out


def profile_backward(
    loss: torch.Tensor,
    device: torch.device,
    profile_dir: pathlib.Path,
) -> None:
    """Profile the backward pass from the loss tensor."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        acc_events=True,
    ) as prof_backward:
        with torch.profiler.record_function('backward'):
            loss.backward()

    _log_profile(prof_backward.key_averages(), profile_dir / 'bwd_profile.csv', device)
    return


def _log_profile(key_averages: profiler_util.EventList, path: pathlib.Path, device: torch.device) -> None:
    headers = [
        'name',
        'count',
        'cpu_time_total_us',
        'cpu_time_avg_us',
        'cpu_memory_usage',
        'self_cpu_memory_usage',
    ]
    if device.type == 'cuda':
        headers.extend(
            [
                'cuda_time_total_us',
                'cuda_time_avg_us',
                'cuda_memory_usage',
                'self_cuda_memory_usage',
            ]
        )

    rows = []
    for event in key_averages:
        row = [
            event.key,
            event.count,
            f'{event.cpu_time_total:.2e}',
            f'{event.cpu_time:.2e}',
            f'{event.cpu_memory_usage:.2e}',
            f'{event.self_cpu_memory_usage:.2e}',
        ]
        if device.type == 'cuda':
            row.extend(
                [
                    f'{event.device_time_total:.2e}',
                    f'{event.device_time:.2e}',
                    f'{event.device_memory_usage:.2e}',
                    f'{event.self_device_memory_usage:.2e}',
                ]
            )

        rows.append(row)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return


def _get_parameter_decomposition(module: torch.nn.Module, max_depth: int = 3) -> list[tuple[str, int, int]]:
    decomp = []

    def _recurse(m: torch.nn.Module, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return

        total_p = sum(p.numel() for p in m.parameters(recurse=True))
        trainable_p = sum(p.numel() for p in m.parameters(recurse=True) if p.requires_grad)
        if total_p == 0:
            return

        name = prefix if prefix else 'root'
        decomp.append((name, total_p, trainable_p))
        for child_name, child in m.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            _recurse(child, depth + 1, child_prefix)

        return

    _recurse(module, 0, '')
    return decomp
