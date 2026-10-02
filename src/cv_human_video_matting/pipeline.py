"""Video-level matting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from cv_human_video_matting.depth import DepthEstimator, draw_distance_label, validate_depth_checkpoint
from cv_human_video_matting.io.video import VideoInfo, probe_video
from cv_human_video_matting.matting.base import MattingRunner
from cv_human_video_matting.postprocess import TemporalAlphaSmoother, refine_alpha
from cv_human_video_matting.render.compositor import (
    alpha_to_bgr,
    compose_on_blur,
    compose_on_color,
    compose_on_image,
    compose_transparent,
    side_by_side,
)
from cv_human_video_matting.utils.paths import ensure_parent_dir
from cv_human_video_matting.utils.config import validate_project_config
from cv_human_video_matting.utils.progress import (
    CancelCallback, ProgressCallback, check_cancelled,
)


@dataclass(frozen=True)
class PipelineStats:
    output_path: Path
    frames: int
    elapsed_seconds: float
    fps: float


@dataclass(frozen=True)
class PreparedVideoRequest:
    """Read-only preflight results; no model or output resources are acquired."""

    info: VideoInfo
    output_path: Path
    background_bgr: np.ndarray | None
    total_frames: int | None


def _load_background(path: str | Path | None):
    if path is None:
        return None
    background = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"Failed to read background image: {path}")
    return background


def _transparent_output_dir(output_path: Path) -> Path:
    """Resolve the PNG sequence directory for transparent renders.

    Always appends a ``_frames`` suffix so a suffix-less ``--output`` (e.g. a
    directory path) does not turn an arbitrary existing directory into the
    frame sink — protecting unrelated ``frame_*.png`` files from deletion.
    """
    if output_path.suffix:
        return output_path.parent / f"{output_path.stem}_frames"
    return output_path.parent / f"{output_path.name}_frames"


def _prepare_png_sequence_dir(output_dir: Path) -> Path:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"PNG output path is not a directory: {output_dir}")
    if any(output_dir.glob("frame_*.png")):
        raise ValueError(f"PNG sequence already exists; choose a new output path: {output_dir}")
    return output_dir


def _commit_png_sequence(
    staging: Path, output: Path, should_cancel: CancelCallback | None,
) -> None:
    """Publish a complete sequence, preserving unrelated pre-existing files."""
    check_cancelled(should_cancel)
    _prepare_png_sequence_dir(output)
    if not output.exists():
        staging.rename(output)
        return

    # Existing directories may contain notes or other user files. Create each
    # destination exclusively, and roll back only the exact files we created if
    # cancellation, an I/O failure or a competing writer prevents completion.
    published = []
    try:
        for source in sorted(staging.glob("frame_*.png")):
            check_cancelled(should_cancel)
            target = output / source.name
            with target.open("xb") as destination:
                identity = os.fstat(destination.fileno())
                published.append((target, identity.st_dev, identity.st_ino))
                with source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, destination)
        check_cancelled(should_cancel)
    except BaseException:
        for target, device, inode in reversed(published):
            try:
                identity = target.stat()
                if (identity.st_dev, identity.st_ino) == (device, inode):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise


def _write_transparent_frame(output_dir: Path, index: int, frame_bgra: np.ndarray) -> None:
    if frame_bgra.ndim != 3 or frame_bgra.shape[2] != 4:
        raise ValueError(
            "Transparent render mode must produce BGRA frames for PNG sequence output."
        )
    frame_path = output_dir / f"frame_{index:06d}.png"
    if not cv2.imwrite(str(frame_path), frame_bgra):
        raise ValueError(f"Failed to write transparent PNG frame: {frame_path}")


def load_background_image(path: str | Path | None) -> np.ndarray | None:
    """Load an optional BGR background image for video or live-frame rendering."""
    return _load_background(path)


def _render_frame(
    frame_bgr, alpha, mode: str, background_bgr=None, foreground_bgr=None,
    background_color_bgr: tuple[int, int, int] = (0, 255, 0),
):
    foreground = frame_bgr if foreground_bgr is None else foreground_bgr
    if mode == "alpha":
        return alpha_to_bgr(alpha)
    if mode == "green":
        return compose_on_color(foreground, alpha, color_bgr=background_color_bgr)
    if mode == "foreground":
        return compose_on_color(foreground, alpha, color_bgr=(0, 0, 0))
    if mode == "blur":
        return compose_on_blur(frame_bgr, alpha, blur_ksize=45, foreground_bgr=foreground)
    if mode == "transparent":
        return compose_transparent(foreground, alpha)
    if mode == "image":
        if background_bgr is None:
            raise ValueError("Render mode 'image' requires a background image.")
        return compose_on_image(foreground, alpha, background_bgr)
    if mode == "side_by_side":
        composed = compose_on_color(foreground, alpha, color_bgr=background_color_bgr)
        return side_by_side(frame_bgr, composed)
    raise ValueError(f"Unsupported render mode: {mode}")


def render_matting_frame(
    frame_bgr: np.ndarray,
    runner: MattingRunner,
    mode: str,
    background_bgr: np.ndarray | None = None,
    edge_feather: int = 0,
    mask_open: int = 0,
    mask_close: int = 0,
    temporal_smoother: TemporalAlphaSmoother | None = None,
    depth_estimator: DepthEstimator | None = None,
    depth_map: np.ndarray | None = None,
    background_color_bgr: tuple[int, int, int] = (0, 255, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Run matting on one BGR frame and return rendered frame with alpha.

    Args:
        frame_bgr: Input frame in BGR format.
        runner: Matting model runner.
        mode: Render mode.
        background_bgr: Optional background image.
        edge_feather: Edge feathering radius.
        mask_open: Morphological opening radius.
        mask_close: Morphological closing radius.
        temporal_smoother: Optional temporal smoother.
        depth_estimator: Optional depth estimator for distance labeling.
        depth_map: Optional pre-computed depth map (to avoid recomputation).
        background_color_bgr: Solid background color for green and side_by_side modes.

    Returns:
        Tuple of (rendered_frame, alpha_matte).
    """
    _validate_background_color(background_color_bgr)
    result = runner.predict(frame_bgr)
    alpha = _postprocess_alpha(
        result.alpha,
        edge_feather=edge_feather,
        mask_open=mask_open,
        mask_close=mask_close,
        temporal_smoother=temporal_smoother,
    )
    rendered = _render_frame(
        frame_bgr, alpha, mode, background_bgr, result.foreground, background_color_bgr,
    )

    # Add distance labeling if depth estimator is provided
    if depth_estimator is not None and mode not in ("transparent", "alpha"):
        if depth_map is None:
            depth_map = depth_estimator.estimate(frame_bgr)
        distance, label = depth_estimator.person_distance(depth_map, alpha)
        rendered = draw_distance_label(rendered, alpha, distance, label)

    return rendered, alpha


def _postprocess_alpha(
    alpha,
    edge_feather: int,
    mask_open: int,
    mask_close: int,
    temporal_smoother: TemporalAlphaSmoother | None,
):
    refined = refine_alpha(
        alpha,
        feather_radius=edge_feather,
        open_radius=mask_open,
        close_radius=mask_close,
    )
    if temporal_smoother is not None:
        return temporal_smoother.apply(refined)
    return refined


def _validate_background_color(color: tuple[int, int, int]) -> None:
    if (
        not isinstance(color, (tuple, list)) or len(color) != 3
        or any(isinstance(channel, bool) or not isinstance(channel, (int, np.integer))
               or not 0 <= channel <= 255 for channel in color)
    ):
        raise ValueError("background_color_bgr must contain three integer channels in [0, 255].")


def _video_frame_size(frame: np.ndarray, resize_short: int | None) -> tuple[int, int]:
    """Compute the requested size; do not resample frames already below the limit."""
    height, width = frame.shape[:2]
    if resize_short is None:
        return width, height
    if min(width, height) < 2:
        raise ValueError("Resizing video requires both input dimensions to be at least 2.")
    scale = min(1.0, resize_short / min(width, height))
    if scale == 1.0:
        return width, height
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def _verify_written_frame_count(path: Path, expected: int) -> None:
    """Verify the closed output container instead of trusting VideoWriter.write()."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Written video could not be reopened for verification: {path}")
        actual = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(actual) or actual != expected:
            raise RuntimeError(
                f"Written video frame count mismatch: expected {expected}, found {actual}; "
                f"check available disk space and the video encoder: {path}"
            )
    finally:
        capture.release()


def validate_video_request(
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    max_frames: int | None = None,
    background_path: str | Path | None = None,
    edge_feather: int = 0,
    mask_open: int = 0,
    mask_close: int = 0,
    temporal_smooth: bool = False,
    temporal_blend: float = 0.75,
    enable_depth: bool = False,
    depth_interval: int = 5,
    resize_short: int | None = None,
    depth_model: str = "MiDaS_small",
    depth_checkpoint: str | Path | None = None,
    depth_near: float = 0.66,
    depth_mid: float = 0.33,
    background_color_bgr: tuple[int, int, int] = (0, 255, 0),
    temporal_mode: str = "ema",
    temporal_motion_threshold: float = 0.15,
) -> PreparedVideoRequest:
    """Validate shared CLI/UI processing options before loading any model.

    Probe the input and read the optional background, but never create output
    directories or initialize inference models. ``output_path`` in the return
    value is the resolved PNG sequence destination for transparent mode.
    The video metadata proves the container can be opened; completeness is
    checked during decoding by ``process_video``.
    """
    if not isinstance(mode, str) or mode not in {
        "alpha", "green", "foreground", "blur", "transparent", "image", "side_by_side",
    }:
        raise ValueError(f"Unsupported render mode: {mode}")
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer or None.")
    validate_project_config({
        "input": {"resize_short": resize_short},
        "postprocess": {
            "enable_temporal_smooth": temporal_smooth, "temporal_alpha": temporal_blend,
            "edge_feather": edge_feather, "mask_open": mask_open, "mask_close": mask_close,
            "temporal_mode": temporal_mode, "temporal_motion_threshold": temporal_motion_threshold,
        },
        "depth": {
            "enabled": enable_depth, "model": depth_model,
            "checkpoint": str(depth_checkpoint) if depth_checkpoint is not None else None,
            "interval": depth_interval,
            "thresholds": {"near": depth_near, "mid": depth_mid},
        },
    })
    if enable_depth and mode not in {"alpha", "transparent"}:
        validate_depth_checkpoint(depth_model, depth_checkpoint)
    _validate_background_color(background_color_bgr)
    requested_output = Path(output_path)
    source = Path(input_path)
    if (source.resolve() == requested_output.resolve()
            or (source.exists() and requested_output.exists()
                and source.samefile(requested_output))):
        raise ValueError("Input and output paths must be different.")
    output = (
        _transparent_output_dir(requested_output) if mode == "transparent" else requested_output
    )
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different.")
    if mode == "transparent":
        _prepare_png_sequence_dir(output)
    elif output.exists() and not output.is_file():
        raise ValueError(f"Video output path is not a file: {output}")
    for parent in output.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Output parent is not a directory: {parent}")
    background_bgr = _load_background(background_path)
    if mode == "image" and background_bgr is None:
        raise ValueError("Render mode 'image' requires a background image.")
    info = probe_video(input_path)
    if (
        not math.isfinite(info.fps) or info.fps <= 0
        or not math.isfinite(info.width) or info.width <= 0
        or not math.isfinite(info.height) or info.height <= 0
    ):
        raise ValueError(
            f"Input video has invalid dimensions or frame rate; verify/re-encode source: {info.path}"
        )
    total_frames = info.frame_count if info.frame_count > 0 else None
    if total_frames is not None and max_frames is not None:
        total_frames = min(total_frames, max_frames)
    return PreparedVideoRequest(info, output, background_bgr, total_frames)


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    runner: MattingRunner,
    mode: str,
    max_frames: int | None = None,
    background_path: str | Path | None = None,
    edge_feather: int = 0,
    mask_open: int = 0,
    mask_close: int = 0,
    temporal_smooth: bool = False,
    temporal_blend: float = 0.75,
    enable_depth: bool = False,
    depth_interval: int = 5,
    device: str = "cpu",
    resize_short: int | None = None,
    depth_model: str = "MiDaS_small",
    depth_checkpoint: str | Path | None = None,
    depth_near: float = 0.66,
    depth_mid: float = 0.33,
    background_color_bgr: tuple[int, int, int] = (0, 255, 0),
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    temporal_mode: str = "ema",
    temporal_motion_threshold: float = 0.15,
) -> PipelineStats:
    """Process video with per-frame progress and cooperative cancellation.

    Progress includes an initial ``(0, total)`` notification. ``total`` is the
    source frame count limited by ``max_frames``, or None for unknown metadata.
    Cancellation is checked between expensive operations and before publishing
    transparent output. The caller owns any intermediate nontransparent MP4.
    """
    # Validate before opening a writer or modifying an existing output.
    started_at = perf_counter()
    check_cancelled(should_cancel)
    prepared = validate_video_request(
        input_path, output_path, mode, max_frames=max_frames, background_path=background_path,
        edge_feather=edge_feather, mask_open=mask_open, mask_close=mask_close,
        temporal_smooth=temporal_smooth, temporal_blend=temporal_blend,
        enable_depth=enable_depth, depth_interval=depth_interval, resize_short=resize_short,
        depth_model=depth_model, depth_checkpoint=depth_checkpoint,
        depth_near=depth_near, depth_mid=depth_mid,
        background_color_bgr=background_color_bgr, temporal_mode=temporal_mode,
        temporal_motion_threshold=temporal_motion_threshold,
    )
    info, output = prepared.info, prepared.output_path
    total_frames, background_bgr = prepared.total_frames, prepared.background_bgr
    if progress_callback is not None:
        progress_callback(0, total_frames)
    check_cancelled(should_cancel)
    temporal_smoother = (
        TemporalAlphaSmoother(blend=temporal_blend, mode=temporal_mode,
                             motion_threshold=temporal_motion_threshold)
        if temporal_smooth else None
    )
    depth_options = {"device": device, "model_type": depth_model,
                     "near_threshold": depth_near, "mid_threshold": depth_mid}
    if depth_checkpoint is not None:
        depth_options["checkpoint_path"] = depth_checkpoint
    depth_estimator = (
        DepthEstimator(**depth_options)
        if enable_depth and mode not in {"alpha", "transparent"} else None
    )
    reset = getattr(runner, "reset", None)
    if callable(reset):
        reset()
    check_cancelled(should_cancel)
    capture = cv2.VideoCapture(str(info.path))
    writer = None
    staging = None
    processed = 0
    last_depth_map = None
    source_shape = None
    frame_size = None
    rendered_shape = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Failed to open input video: {info.path}")
        if mode == "transparent":
            ensure_parent_dir(output)
            staging = TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent)
        while max_frames is None or processed < max_frames:
            check_cancelled(should_cancel)
            ok, frame = capture.read()
            if not ok:
                break
            if source_shape is None:
                source_shape = frame.shape[:2]
                frame_size = _video_frame_size(frame, resize_short)
            elif frame.shape[:2] != source_shape:
                raise ValueError("Input video frame dimensions changed during processing.")
            if (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            if depth_estimator is not None and processed % depth_interval == 0:
                last_depth_map = depth_estimator.estimate(frame)
            check_cancelled(should_cancel)
            rendered, _ = render_matting_frame(
                frame, runner=runner, mode=mode, background_bgr=background_bgr,
                edge_feather=edge_feather, mask_open=mask_open, mask_close=mask_close,
                temporal_smoother=temporal_smoother, depth_estimator=depth_estimator,
                depth_map=last_depth_map,
                background_color_bgr=background_color_bgr,
            )
            check_cancelled(should_cancel)
            # OpenCV's MP4 writer silently truncates odd rows/columns. Pad only
            # the final video frame so source pixels, alpha and depth stay aligned.
            # PNG sequences retain their exact rendered dimensions.
            if mode != "transparent":
                height, width = rendered.shape[:2]
                if height % 2 or width % 2:
                    rendered = cv2.copyMakeBorder(
                        rendered, 0, height % 2, 0, width % 2,
                        cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    )
            if processed == 0:
                rendered_shape = rendered.shape
                if mode != "transparent":
                    ensure_parent_dir(output)
                    height, width = rendered.shape[:2]
                    codec = "FFV1" if output.suffix.lower() == ".mkv" else "mp4v"
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*codec),
                        info.fps, (width, height),
                    )
                    if not writer.isOpened():
                        raise ValueError(f"Failed to open output video writer: {output}")
            elif rendered.shape != rendered_shape:
                raise ValueError("Rendered frame dimensions changed during processing.")
            if mode == "transparent":
                _write_transparent_frame(Path(staging.name), processed, rendered)
            else:
                writer.write(rendered)
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total_frames)
        check_cancelled(should_cancel)
        if processed == 0:
            raise ValueError(f"Input video has no readable frames: {info.path}")
        if total_frames is not None and processed < total_frames:
            raise ValueError(
                f"Input video decoding ended early: decoded {processed} of {total_frames} "
                f"expected frames. This may indicate truncated decoding or inaccurate "
                f"frame-count metadata; verify/re-encode source: {info.path}"
            )
        if staging is not None:
            _commit_png_sequence(Path(staging.name), output, should_cancel)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if staging is not None:
            staging.cleanup()
    if mode != "transparent":
        check_cancelled(should_cancel)
        _verify_written_frame_count(output, processed)
    elapsed = perf_counter() - started_at
    return PipelineStats(
        output_path=output, frames=processed, elapsed_seconds=elapsed,
        fps=processed / elapsed if elapsed > 0 else 0.0,
    )
