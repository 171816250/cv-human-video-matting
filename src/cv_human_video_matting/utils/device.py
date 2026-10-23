"""Device selection helpers."""

from __future__ import annotations

import re


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        # DLL 缺失等异常也按 CUDA 不可用处理；是否回退 CPU 由调用方决定。
        return False


def resolve_device(requested: str | None = "auto") -> str:
    """解析推理设备；只有 auto 允许回退到 CPU。

    显式 CUDA 不可用或设备编号越界时直接报错，不改变进程的当前设备。
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
