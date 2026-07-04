"""Alpha matte edge refinement utilities."""

from __future__ import annotations

import cv2
import numpy as np


def _as_uint8_alpha(alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim == 3 and alpha.shape[2] == 3:
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    if alpha.ndim == 3 and alpha.shape[2] == 1:
        alpha = alpha[:, :, 0]
    if alpha.ndim != 2:
        raise ValueError(f"Alpha must be HxW or HxWx1, got shape {alpha.shape}")

    if alpha.dtype == np.uint8:
        return alpha.copy()

    alpha_f32 = alpha.astype(np.float32)
    if alpha_f32.size > 0 and alpha_f32.max() <= 1.0:
        alpha_f32 = alpha_f32 * 255.0
    return np.clip(alpha_f32, 0, 255).astype(np.uint8)


def _kernel(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Blur hard alpha edges with a small, explicit radius."""
    alpha_u8 = _as_uint8_alpha(alpha)
    if radius <= 0:
        return alpha_u8
    kernel_size = radius * 2 + 1
    return cv2.GaussianBlur(alpha_u8, (kernel_size, kernel_size), sigmaX=0)


def clean_alpha(
    alpha: np.ndarray,
    open_radius: int = 0,
    close_radius: int = 0,
) -> np.ndarray:
    """Remove small foreground specks and fill small background holes."""
    cleaned = _as_uint8_alpha(alpha)
    if open_radius > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _kernel(open_radius))
    elif open_radius < 0:
        _kernel(open_radius)
    if close_radius > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, _kernel(close_radius))
    elif close_radius < 0:
        _kernel(close_radius)
    return cleaned


def refine_alpha(
    alpha: np.ndarray,
    feather_radius: int = 0,
    open_radius: int = 0,
    close_radius: int = 0,
) -> np.ndarray:
    """Apply morphology cleanup first, then edge feathering."""
    cleaned = clean_alpha(alpha, open_radius=open_radius, close_radius=close_radius)
    return feather_alpha(cleaned, radius=feather_radius)
