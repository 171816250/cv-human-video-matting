"""Device selection helpers."""

from __future__ import annotations


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

    Any CUDA-family request (``cuda``, ``cuda:0``, ...) falls back to CPU when
    CUDA is unavailable, so a device string never crashes torch downstream.
    """
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if cuda_available() else "cpu"
    if normalized.startswith("cuda") and not cuda_available():
        return "cpu"
    return normalized
