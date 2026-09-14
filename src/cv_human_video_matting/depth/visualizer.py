"""Visualization utilities for distance labeling on video frames."""

from __future__ import annotations

import cv2
import numpy as np
from cv_human_video_matting.render.compositor import normalize_alpha


def draw_distance_label(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    distance: float | None,
    label: str,
    thresh: float = 0.5,
) -> np.ndarray:
    """Draw distance label and bounding box on a person in the frame.

    Args:
        frame_bgr: Input frame in BGR format (HxWx3, uint8).
        alpha: Alpha matte (HxW or HxWx1, float32, 0~1).
        distance: Relative distance value (0~1, higher=farther), or None.
        label: Distance label text ("Near", "Mid", "Far", etc.).
        thresh: Alpha threshold to consider as person region.

    Returns:
        Frame with distance annotation drawn (HxWx3, uint8).
    """
    # Create a copy to avoid modifying the input
    output = frame_bgr.copy()

    # Ensure alpha is 2D
    alpha = normalize_alpha(alpha)[:, :, 0]

    # Create person mask
    mask = (alpha > thresh).astype(np.uint8)

    # Check if person region exists
    if mask.sum() < 50:
        return output

    # Find person bounding box
    ys, xs = np.where(mask)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    # Draw bounding box (green)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Format label text
    if distance is not None:
        text = f"{label}  d~{distance:.2f}"
    else:
        text = label

    # Draw label above bounding box
    text_y = max(20, y1 - 8)
    cv2.putText(
        output,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def draw_depth_heatmap(
    depth: np.ndarray,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Convert depth map to BGR heatmap for visualization.

    Args:
        depth: Normalized depth map (HxW, float32, 0~1).
        colormap: OpenCV colormap ID (default: VIRIDIS).

    Returns:
        BGR heatmap image (HxWx3, uint8).
    """
    # Convert to uint8 range
    depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(depth_u8, colormap)

    return heatmap


def overlay_depth_on_frame(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray,
    opacity: float = 0.5,
) -> np.ndarray:
    """Overlay depth heatmap on frame, masked by alpha.

    Args:
        frame_bgr: Input frame in BGR format (HxWx3, uint8).
        depth: Normalized depth map (HxW, float32, 0~1).
        alpha: Alpha matte to mask the overlay (HxW or HxWx1, float32, 0~1).
        opacity: Overlay opacity (0~1).

    Returns:
        Frame with depth overlay (HxWx3, uint8).
    """
    # Generate depth heatmap
    heatmap = draw_depth_heatmap(depth)

    # Ensure alpha is 3D
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    # Blend frame and heatmap using alpha mask
    frame_f = frame_bgr.astype(np.float32)
    heatmap_f = heatmap.astype(np.float32)

    # Apply overlay only in person region (alpha > 0)
    overlay = frame_f * (1 - opacity * alpha) + heatmap_f * (opacity * alpha)

    return np.clip(overlay, 0, 255).astype(np.uint8)
