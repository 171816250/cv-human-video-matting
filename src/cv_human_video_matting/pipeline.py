"""Video-level matting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from cv_human_video_matting.depth import DepthEstimator, draw_distance_label
from cv_human_video_matting.io.video import probe_video
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
from cv_human_video_matting.utils.paths import ensure_dir, ensure_parent_dir


@dataclass(frozen=True)
class PipelineStats:
    output_path: Path
    frames: int
    elapsed_seconds: float
    fps: float


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
    ensure_dir(output_dir)
    if any(output_dir.glob("frame_*.png")):
        raise ValueError(f"PNG sequence already exists; choose a new output path: {output_dir}")
    return output_dir


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


def _render_frame(frame_bgr, alpha, mode: str, background_bgr=None, foreground_bgr=None):
    foreground = frame_bgr if foreground_bgr is None else foreground_bgr
    if mode == "alpha":
        return alpha_to_bgr(alpha)
    if mode == "green":
        return compose_on_color(foreground, alpha, color_bgr=(0, 255, 0))
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
        composed = compose_on_color(foreground, alpha, color_bgr=(0, 255, 0))
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

    Returns:
        Tuple of (rendered_frame, alpha_matte).
    """
    result = runner.predict(frame_bgr)
    alpha = _postprocess_alpha(
        result.alpha,
        edge_feather=edge_feather,
        mask_open=mask_open,
        mask_close=mask_close,
        temporal_smoother=temporal_smoother,
    )
    rendered = _render_frame(frame_bgr, alpha, mode, background_bgr, result.foreground)

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
) -> PipelineStats:
    # Validate before opening a writer or modifying an existing output.
    started_at = perf_counter()
    if mode not in {"alpha", "green", "foreground", "blur", "transparent", "image", "side_by_side"}:
        raise ValueError(f"Unsupported render mode: {mode}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")
    if min(edge_feather, mask_open, mask_close) < 0:
        raise ValueError("Postprocessing radii must be non-negative.")
    requested_output = Path(output_path)
    if Path(input_path).resolve() == requested_output.resolve():
        raise ValueError("Input and output paths must be different.")
    background_bgr = _load_background(background_path)
    if mode == "image" and background_bgr is None:
        raise ValueError("Render mode 'image' requires a background image.")
    info = probe_video(input_path)
    temporal_smoother = (
        TemporalAlphaSmoother(blend=temporal_blend) if temporal_smooth else None
    )
    depth_estimator = (
        DepthEstimator(device=device) if enable_depth and mode not in {"alpha", "transparent"}
        else None
    )
    depth_interval = max(1, depth_interval)
    reset = getattr(runner, "reset", None)
    if callable(reset):
        reset()
    output = (
        _transparent_output_dir(requested_output) if mode == "transparent" else requested_output
    )
    capture = cv2.VideoCapture(str(info.path))
    writer = None
    processed = 0
    last_depth_map = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Failed to open input video: {info.path}")
        while max_frames is None or processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if depth_estimator is not None and processed % depth_interval == 0:
                last_depth_map = depth_estimator.estimate(frame)
            rendered, _ = render_matting_frame(
                frame, runner=runner, mode=mode, background_bgr=background_bgr,
                edge_feather=edge_feather, mask_open=mask_open, mask_close=mask_close,
                temporal_smoother=temporal_smoother, depth_estimator=depth_estimator,
                depth_map=last_depth_map,
            )
            if processed == 0:
                if mode == "transparent":
                    _prepare_png_sequence_dir(output)
                else:
                    ensure_parent_dir(output)
                    height, width = rendered.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                        info.fps if info.fps > 0 else 30.0, (width, height),
                    )
                    if not writer.isOpened():
                        raise ValueError(f"Failed to open output video writer: {output}")
            if mode == "transparent":
                _write_transparent_frame(output, processed, rendered)
            else:
                writer.write(rendered)
            processed += 1
        if processed == 0:
            raise ValueError(f"Input video has no readable frames: {info.path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    elapsed = perf_counter() - started_at
    return PipelineStats(
        output_path=output, frames=processed, elapsed_seconds=elapsed,
        fps=processed / elapsed if elapsed > 0 else 0.0,
    )
