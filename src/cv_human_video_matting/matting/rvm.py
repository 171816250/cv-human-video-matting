"""Robust Video Matting runner wrapper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from cv_human_video_matting.matting.base import MattingResult


class RVMRunner:
    def __init__(
        self,
        model_name: str = "mobilenetv3",
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        downsample_ratio: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.downsample_ratio = downsample_ratio
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting:master",
            model_name,
            pretrained=True,
            trust_repo=True,
        )
        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(str(checkpoint), map_location=self.device)
            self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self._rec = [None, None, None, None]

    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor, *self._rec, downsample_ratio=self.downsample_ratio)

        foreground, alpha, *rec = output
        self._rec = rec[:4]

        alpha_np = alpha[0, 0].detach().cpu().numpy()
        alpha_u8 = np.clip(alpha_np * 255.0, 0, 255).astype(np.uint8)

        foreground_np = foreground[0].detach().cpu().permute(1, 2, 0).numpy()
        foreground_u8 = np.clip(foreground_np * 255.0, 0, 255).astype(np.uint8)
        foreground_bgr = cv2.cvtColor(foreground_u8, cv2.COLOR_RGB2BGR)

        return MattingResult(alpha=alpha_u8, foreground=foreground_bgr)
