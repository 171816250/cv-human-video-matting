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
from cv_human_video_matting.utils.paths import ensure_parent_dir


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


def load_background_image(path: str | Path | None) -> np.ndarray | None:
    """Load an optional BGR background image for video or live-frame rendering."""
    return _load_background(path)


def _render_frame(frame_bgr, alpha, mode: str, background_bgr=None):
    if mode == "alpha":
        return alpha_to_bgr(alpha)
    if mode == "green":
        return compose_on_color(frame_bgr, alpha, color_bgr=(0, 255, 0))
    if mode == "foreground":
        return compose_on_color(frame_bgr, alpha, color_bgr=(0, 0, 0))
    if mode == "blur":
        return compose_on_blur(frame_bgr, alpha, blur_ksize=45)
    if mode == "transparent":
        return compose_transparent(frame_bgr, alpha)
    if mode == "image":
        if background_bgr is None:
            raise ValueError("Render mode 'image' requires a background image.")
        return compose_on_image(frame_bgr, alpha, background_bgr)
    if mode == "side_by_side":
        composed = compose_on_color(frame_bgr, alpha, color_bgr=(0, 255, 0))
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
    rendered = _render_frame(frame_bgr, alpha, mode, background_bgr)

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
    info = probe_video(input_path)
    output = Path(output_path)
    ensure_parent_dir(output)
    background_bgr = _load_background(background_path)
    temporal_smoother = (
        TemporalAlphaSmoother(blend=temporal_blend) if temporal_smooth else None
    )

    # Initialize depth estimator if enabled
    depth_estimator = DepthEstimator(device=device) if enable_depth else None
    last_depth_map = None
    frame_idx = 0

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open input video: {info.path}")

    first_ok, first_frame = capture.read()
    if not first_ok:
        capture.release()
        raise ValueError(f"Input video has no readable frames: {info.path}")

    # Compute depth for first frame if enabled
    if depth_estimator is not None:
        last_depth_map = depth_estimator.estimate(first_frame)

    first_rendered, _ = render_matting_frame(
        first_frame,
        runner=runner,
        mode=mode,
        background_bgr=background_bgr,
        edge_feather=edge_feather,
        mask_open=mask_open,
        mask_close=mask_close,
        temporal_smoother=temporal_smoother,
        depth_estimator=depth_estimator,
        depth_map=last_depth_map,
    )
    out_height, out_width = first_rendered.shape[:2]

    fps = info.fps if info.fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Failed to open output video writer: {output}")

    processed = 0
    started_at = perf_counter()
    try:
        writer.write(first_rendered)
        processed += 1
        frame_idx += 1

        while max_frames is None or processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            # Update depth map at intervals
            if depth_estimator is not None and frame_idx % depth_interval == 0:
                last_depth_map = depth_estimator.estimate(frame)

            rendered, _ = render_matting_frame(
                frame,
                runner=runner,
                mode=mode,
                background_bgr=background_bgr,
                edge_feather=edge_feather,
                mask_open=mask_open,
                mask_close=mask_close,
                temporal_smoother=temporal_smoother,
                depth_estimator=depth_estimator,
                depth_map=last_depth_map,
            )
            writer.write(rendered)
            processed += 1
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    elapsed = perf_counter() - started_at
    effective_fps = processed / elapsed if elapsed > 0 else 0.0
    return PipelineStats(
        output_path=output,
        frames=processed,
        elapsed_seconds=elapsed,
        fps=effective_fps,
    )
