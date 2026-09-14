"""Reproducible run metadata for performance comparisons and acceptance reports."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import torch

from cv_human_video_matting.io.video import probe_video
from cv_human_video_matting.pipeline import PipelineStats


def write_run_report(path, input_path, stats: PipelineStats, options: dict, total_seconds: float):
    """Write measured throughput, conditions and versions; never imply quality scores."""
    target = Path(path)
    if target.resolve() in {Path(input_path).resolve(), stats.output_path.resolve()}:
        raise ValueError("Report path must differ from input and output paths.")
    dependencies = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "gradio"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    info = probe_video(input_path)
    device = str(options.get("device", "cpu"))
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(info.path), "width": info.width, "height": info.height,
                  "fps": info.fps, "frame_count": info.frame_count},
        "options": options,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": dependencies,
                        "gpu": torch.cuda.get_device_name(torch.device(device))
                        if device.startswith("cuda") and torch.cuda.is_available() else None},
        "result": {**asdict(stats), "output_path": str(stats.output_path),
                   "total_seconds": total_seconds,
                   "total_fps": stats.frames / total_seconds if total_seconds > 0 else 0.0},
        "timing_scope": {
            "elapsed_seconds": "Video pipeline including first frame, I/O, depth initialization and writer close; excludes matting model loading.",
            "total_seconds": "Matting model loading and video pipeline; excludes JSON report generation.",
        },
        "quality_metrics": None,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
