"""Alpha matte compositing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim == 3 and alpha.shape[2] == 3:
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    if alpha.ndim == 3 and alpha.shape[2] == 1:
        alpha = alpha[:, :, 0]
    if alpha.ndim != 2:
        raise ValueError(f"Alpha must be HxW or HxWx1, got shape {alpha.shape}")

    if alpha.dtype == np.uint8:
        alpha_f32 = alpha.astype(np.float32) / 255.0
    else:
        alpha_f32 = alpha.astype(np.float32)
    alpha_f32 = np.clip(alpha_f32, 0.0, 1.0)
    return alpha_f32[:, :, None]


def compose_on_color(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    alpha_f32 = normalize_alpha(alpha)
    foreground = frame_bgr.astype(np.float32)
    background = np.empty_like(foreground)
    background[:, :] = np.array(color_bgr, dtype=np.float32)
    composed = foreground * alpha_f32 + background * (1.0 - alpha_f32)
    return np.clip(composed, 0, 255).astype(np.uint8)


def resize_cover(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    height, width = image_bgr.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Background image has invalid shape: {image_bgr.shape}")

    scale = max(target_width / width, target_height / height)
    resized_width = max(int(round(width * scale)), target_width)
    resized_height = max(int(round(height * scale)), target_height)
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    return resized[top : top + target_height, left : left + target_width]


def compose_on_image(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    background_bgr: np.ndarray,
) -> np.ndarray:
    alpha_f32 = normalize_alpha(alpha)
    foreground = frame_bgr.astype(np.float32)
    background = resize_cover(
        background_bgr,
        size=(frame_bgr.shape[1], frame_bgr.shape[0]),
    ).astype(np.float32)
    composed = foreground * alpha_f32 + background * (1.0 - alpha_f32)
    return np.clip(composed, 0, 255).astype(np.uint8)


def alpha_to_bgr(alpha: np.ndarray) -> np.ndarray:
    alpha_f32 = normalize_alpha(alpha)
    alpha_u8 = np.clip(alpha_f32[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2BGR)


def side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    if left_bgr.shape[0] != right_bgr.shape[0]:
        right_bgr = cv2.resize(
            right_bgr,
            (right_bgr.shape[1], left_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.concatenate([left_bgr, right_bgr], axis=1)


def compose_on_blur(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    blur_ksize: int = 45,
    foreground_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """Compose foreground on blurred background (bokeh effect).

    Args:
        frame_bgr: Input frame in BGR format (HxWx3, uint8).
        alpha: Alpha matte (HxW or HxWx1, float32, 0~1).
        blur_ksize: Gaussian blur kernel size (must be positive odd).

    Returns:
        Composed image with blurred background (HxWx3, uint8).
    """
    alpha_f32 = normalize_alpha(alpha)
    foreground = (frame_bgr if foreground_bgr is None else foreground_bgr).astype(np.float32)

    # Ensure kernel size is odd
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    k = max(1, k)  # Ensure positive

    # Blur the entire frame
    background = cv2.GaussianBlur(frame_bgr, (k, k), 0).astype(np.float32)

    # Blend: sharp foreground + blurred background
    composed = foreground * alpha_f32 + background * (1.0 - alpha_f32)
    return np.clip(composed, 0, 255).astype(np.uint8)


def compose_transparent(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Create BGRA image with transparent background.

    Args:
        frame_bgr: Input frame in BGR format (HxWx3, uint8).
        alpha: Alpha matte (HxW or HxWx1, float32, 0~1).

    Returns:
        BGRA image with alpha channel (HxWx4, uint8).
    """
    alpha_f32 = normalize_alpha(alpha)
    alpha_u8 = (alpha_f32[:, :, 0] * 255).astype(np.uint8)

    # Stack BGR + alpha to create BGRA
    return np.dstack([frame_bgr, alpha_u8])
