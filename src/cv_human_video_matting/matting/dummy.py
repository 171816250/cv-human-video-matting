"""Deterministic matting runner used for tests and pipeline debugging."""

from __future__ import annotations

import cv2
import numpy as np

from cv_human_video_matting.matting.base import MattingResult


class DummyMattingRunner:
    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        height, width = frame_bgr.shape[:2]
        alpha = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (max(width // 5, 1), max(height // 3, 1))
        cv2.ellipse(alpha, center, axes, 0, 0, 360, 255, -1)
        return MattingResult(alpha=alpha, foreground=None)
