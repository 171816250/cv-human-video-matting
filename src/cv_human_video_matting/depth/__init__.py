"""Depth estimation and relative distance labeling utilities."""

from cv_human_video_matting.depth.estimator import (
    DEPTH_ANYTHING_V2_SMALL,
    DepthEstimator,
    validate_depth_checkpoint,
)
from cv_human_video_matting.depth.visualizer import (
    draw_depth_heatmap,
    draw_distance_label,
    overlay_depth_on_frame,
)

__all__ = [
    "DEPTH_ANYTHING_V2_SMALL",
    "DepthEstimator",
    "draw_depth_heatmap",
    "draw_distance_label",
    "overlay_depth_on_frame",
    "validate_depth_checkpoint",
]
