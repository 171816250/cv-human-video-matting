"""Measured runtime and artifact provenance for reproducible acceptance reports."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from time import perf_counter
from typing import TYPE_CHECKING

from cv_human_video_matting import __version__

if TYPE_CHECKING:
    from cv_human_video_matting.pipeline import PipelineStats


def _file_record(path: Path) -> dict:
    """Hash large videos in bounded memory and refuse files changed while reading."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while computing SHA256: {path}")
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_text(target: Path, text: str) -> Path:
    """Replace an existing artifact only after the new UTF-8 text is fully written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}-",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _protect_paths(target: Path, sources) -> None:
    resolved_target = target.absolute().resolve()
    for source in sources:
        source = Path(source).expanduser().absolute().resolve()
        if resolved_target == source:
            raise ValueError(f"Report path must differ from input and output paths: {source}")
        if source.is_dir() and resolved_target.is_relative_to(source):
            raise ValueError(f"Report must be outside the output sequence directory: {source}")
        if resolved_target.exists() and source.exists() and resolved_target.samefile(source):
            raise ValueError(f"Report path aliases an existing input or output: {source}")


def validate_artifact_paths(report_path, input_path, output_path, asset_paths=()) -> None:
    """Reject publication paths that could overwrite inputs or model resources."""
    output = Path(output_path)
    sources = [Path(input_path), *(Path(item) for item in asset_paths if item)]
    # Directories such as verified model bundles protect their whole subtree.
    _protect_paths(output, sources)
    if report_path is not None:
        report = Path(report_path)
        if report.suffix.lower() != ".json":
            raise ValueError("Report path must use the .json extension.")
        if report.is_dir():
            raise ValueError(f"Report path is an existing directory: {report}")
        _protect_paths(report, (*sources, output))


def _code_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    code = {"version": __version__, "git_commit": None, "git_dirty": None,
            "source_sha256": None}
    # A source digest distinguishes uncommitted changes and also works in source ZIPs.
    sources = sorted((root / "src" / "cv_human_video_matting").rglob("*.py"))
    if (root / "app.py").is_file():
        sources.append(root / "app.py")
    if sources:
        code["source_sha256"] = _canonical_hash([
            {"path": source.relative_to(root).as_posix(), **_file_record(source)}
            for source in sorted(sources)
        ])
    if not (root / ".git").exists():
        return code

    def git(*args):
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        code["git_commit"] = commit if re.fullmatch(r"[0-9a-fA-F]{40,64}", commit) else None
        code["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        # Git is optional; artifact and source hashes remain usable in an offline ZIP.
        pass
    return code


def _environment(device: str) -> dict:
    import torch

    dependencies = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless",
                 "gradio", "imageio-ffmpeg",
                 "transformers", "timm", "safetensors"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    actual_device = None
    gpu = None
    device_error = None
    if device == "cpu":
        actual_device = "cpu"
    elif re.fullmatch(r"cuda(?::[0-9]+)?", device):
        try:
            if not torch.cuda.is_available():
                device_error = "CUDA requested but unavailable; actual inference device is unverified."
            else:
                index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
                gpu = torch.cuda.get_device_name(index)
                actual_device = f"cuda:{index}"
        except Exception as exc:
            device_error = str(exc)
    else:
        device_error = "Caller did not provide the resolved inference device."
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "packages": dependencies, "device": actual_device, "gpu": gpu,
        "device_error": device_error, "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
    }


def _video_metadata(info, allow_unknown_frames=False) -> dict:
    if (
        info.width <= 0 or info.height <= 0 or (info.frame_count <= 0 and not allow_unknown_frames)
        or not math.isfinite(info.fps) or info.fps <= 0
    ):
        raise ValueError(f"Video has invalid dimensions, frame count, or FPS: {info.path}")
    metadata = {"width": info.width, "height": info.height, "fps": info.fps,
                "frame_count": info.frame_count, "duration_seconds": info.duration_seconds}
    if info.frame_count <= 0:
        metadata.update(frame_count=None, duration_seconds=None,
            completeness="Source frame count unavailable; full decode completeness is unverified.")
    return metadata


def _output_metadata(path: Path, source_fps: float) -> dict:
    if not path.is_dir():
        from cv_human_video_matting.io.video import probe_video

        return {"type": "video", "path": str(path), **_file_record(path),
                **_video_metadata(probe_video(path))}

    import cv2

    files = sorted(path.glob("frame_*.png"))
    if not files:
        raise ValueError(f"PNG sequence contains no frame_*.png files: {path}")
    manifest = []
    shape = None
    for frame in files:
        image = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Invalid transparent PNG frame: {frame}")
        if shape is not None and image.shape != shape:
            raise ValueError(f"PNG sequence has inconsistent frame sizes: {frame}")
        shape = image.shape
        manifest.append({"name": frame.name, **_file_record(frame)})
    return {
        "type": "png_sequence", "path": str(path), "files": manifest,
        "manifest_sha256": _canonical_hash(manifest),
        "size_bytes": sum(item["size_bytes"] for item in manifest),
        "width": shape[1], "height": shape[0], "channels": shape[2],
        "frame_count": len(files), "fps": source_fps,
        "duration_seconds": len(files) / source_fps,
        "timing_source": "Source video FPS; PNG files do not contain a playback timebase.",
    }


def write_run_report(path, input_path, stats: PipelineStats, options: dict, total_seconds: float,
                     export_seconds: float = 0.0, published_output_path=None, model_bundle_info=None):
    """Write measured throughput and hashes without implying matting quality scores.

    ``total_seconds`` remains the caller's measurement ending after delivery export.
    Subsequent hashing/probing/provenance work is measured as ``metadata_seconds``.
    ``published_output_path`` optionally records the eventual public path while
    probing and hashing the actual staged output before transactional publication.
    """
    started = perf_counter()
    target, source, output = Path(path), Path(input_path), Path(stats.output_path)
    published_output = Path(published_output_path) if published_output_path is not None else output
    if target.suffix.lower() != ".json":
        raise ValueError("Report path must use the .json extension.")
    assets_paths = [options.get(name) for name in (
        "checkpoint", "background", "model_bundle", "depth_checkpoint")]
    validate_artifact_paths(target, source, output, assets_paths)
    validate_artifact_paths(target, source, published_output, assets_paths)
    _protect_paths(target, (source, output, published_output))
    if published_output_path is not None:
        _protect_paths(published_output, (source, target))
    for name, value in (("total_seconds", total_seconds), ("export_seconds", export_seconds),
                        ("elapsed_seconds", stats.elapsed_seconds), ("fps", stats.fps)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number.")
    if isinstance(stats.frames, bool) or not isinstance(stats.frames, int) or stats.frames <= 0:
        raise ValueError("frames must be a positive integer.")

    from cv_human_video_matting.io.video import probe_video

    info = probe_video(source)
    assets = {}
    for name in ("checkpoint", "background"):
        if options.get(name):
            asset = Path(options[name])
            assets[name] = {"path": str(asset), **_file_record(asset)}
    if options.get("model_bundle"):
        from cv_human_video_matting.utils.model_bundle import verify_model_bundle

        bundle = verify_model_bundle(options["model_bundle"])
        if model_bundle_info is not None:
            if (model_bundle_info.get("content_sha256") != bundle["content_sha256"]
                    or model_bundle_info.get("model_name") != bundle["model_name"]):
                raise ValueError("Model bundle changed after model initialization; report was not published.")
        weights = next(item for item in bundle["files"] if item["path"] == bundle["weights_path"])
        assets["model_bundle"] = {
            "model_name": bundle["model_name"], "content_sha256": bundle["content_sha256"],
            "weights": weights, "load_identity_confirmed": model_bundle_info is not None,
            "integrity_scope": "Local byte identity; not an upstream signature or quality guarantee.",
        }
    if (options.get("enable_depth") and options.get("depth_model") == "DepthAnythingV2_small"
            and options.get("mode") not in {"alpha", "transparent"}):
        from cv_human_video_matting.depth import validate_depth_checkpoint

        depth_path = options.get("depth_checkpoint")
        validate_depth_checkpoint("DepthAnythingV2_small", depth_path)
        directory = Path(depth_path).expanduser()
        files = {name: _file_record(directory / name) for name in (
            "config.json", "preprocessor_config.json", "model.safetensors")}
        assets["depth_checkpoint"] = {
            "path": str(directory), "files": files, "content_sha256": _canonical_hash(files),
            "integrity_scope": "Local files at report time; loaded weight identity is not confirmed.",
        }
    report = {
        "schema_version": 3, "kind": "run_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": _code_provenance(),
        "input": {"path": str(source), **_file_record(source), **_video_metadata(info, allow_unknown_frames=True)},
        "output": _output_metadata(output, info.fps),
        "assets": assets, "options": options,
        "environment": _environment(str(options.get("device", "unknown"))),
        "result": {**asdict(stats), "output_path": str(output), "export_seconds": export_seconds,
                   "total_seconds": total_seconds,
                   "total_fps": stats.frames / total_seconds if total_seconds > 0 else 0.0},
        "timing_scope": {
            "elapsed_seconds": "Video pipeline including first frame, I/O, depth initialization and writer close; excludes matting model loading.",
            "total_seconds": "Matting model loading, video pipeline and delivery export; excludes metadata collection, JSON report generation and final publication/copy/rollback preparation.",
            "export_seconds": "Delivery video encoding/audio merge or transparent archive packaging.",
            "metadata_seconds": "Additional code/version collection, artifact hashing and media probing; excludes JSON serialization and disk write. Not included in total_seconds.",
        },
        "quality_metrics": None,
    }
    report["result"]["metadata_seconds"] = perf_counter() - started
    report["output"]["path"] = str(published_output)
    report["result"]["output_path"] = str(published_output)
    return _atomic_write_text(target, json.dumps(report, ensure_ascii=False, indent=2,
                                               allow_nan=False) + "\n")
