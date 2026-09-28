"""Device selection helpers."""

from __future__ import annotations

import re


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        # A broken torch/CUDA install can raise OSError/RuntimeError (e.g. a
        # missing DLL on Windows) rather than ImportError; degrade to CPU.
        return False


def resolve_device(requested: str | None = "auto") -> str:
    """Resolve auto/cuda/cpu requests to a concrete inference device.

    Only ``auto`` may fall back to CPU. Explicit CUDA requests fail early if
    unavailable, and indexed requests must refer to a visible CUDA device.
    This function does not change PyTorch's process-wide current device.
    """
    if requested is None:
        requested = "auto"
    if not isinstance(requested, str):
        raise ValueError("Device must be auto, cpu, cuda, or cuda:N (a non-negative integer).")
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda" if cuda_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    match = re.fullmatch(r"cuda(?::([0-9]+))?", normalized)
    if match is None:
        raise ValueError(
            f"Invalid device {requested!r}; use auto, cpu, cuda, or cuda:N "
            "with a non-negative integer index."
        )
    if not cuda_available():
        raise RuntimeError(
            f"CUDA was explicitly requested ({normalized}), but it is unavailable or failed "
            "to initialize. Use device='auto' or 'cpu', or repair the CUDA/PyTorch installation."
        )
    if match[1] is None:
        return "cuda"

    index = int(match[1])
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception as error:
        raise RuntimeError(
            f"Unable to enumerate CUDA devices for explicit request {normalized!r}. "
            "Check the CUDA/PyTorch installation or select device='auto' or 'cpu'."
        ) from error
    if index >= count:
        raise ValueError(
            f"CUDA device index {index} is out of range: {count} visible device(s). "
            "Choose an index smaller than the visible device count."
        )
    return f"cuda:{index}"
