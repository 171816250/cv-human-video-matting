"""Depth estimation and relative distance labeling utilities."""

from cv_human_video_matting.depth.estimator import DepthEstimator
from cv_human_video_matting.depth.visualizer import (
    draw_depth_heatmap,
    draw_distance_label,
    overlay_depth_on_frame,
)

__all__ = [
    "DepthEstimator",
    "draw_distance_label",
    "draw_depth_heatmap",
    "overlay_depth_on_frame",
]
