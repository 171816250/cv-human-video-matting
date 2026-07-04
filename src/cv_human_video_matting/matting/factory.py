"""Matting runner factory."""

from __future__ import annotations

from pathlib import Path

from cv_human_video_matting.matting.base import MattingRunner
from cv_human_video_matting.matting.birefnet import BiRefNetRunner
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.matting.rvm import RVMRunner


def create_matting_runner(
    model: str,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> MattingRunner:
    normalized = model.lower()
    if normalized == "dummy":
        return DummyMattingRunner()
    if normalized in {"rvm", "rvm_mobilenetv3", "mobilenetv3"}:
        return RVMRunner(model_name="mobilenetv3", checkpoint=checkpoint, device=device)
    if normalized in {"rvm_resnet50", "resnet50"}:
        return RVMRunner(model_name="resnet50", checkpoint=checkpoint, device=device)
    if normalized in {"birefnet", "ultra"}:
        return BiRefNetRunner(device=device)
    raise ValueError(f"Unsupported matting model: {model}")
