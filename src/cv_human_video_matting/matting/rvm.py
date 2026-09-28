"""Robust Video Matting runner wrapper."""

from __future__ import annotations

from pathlib import Path
import sys
from threading import RLock

import cv2
import numpy as np
import torch

from cv_human_video_matting.matting.base import MattingResult

# Explicit upstream branch reference, not an immutable commit. Shared so
# download tooling and the runtime select the same torch hub cache directory.
RVM_HUB_REF = "PeterL1n/RobustVideoMatting:master"
_MODEL_IMPORT_LOCK = RLock()


def _load_bundle_model(bundle, model_name):
    # RVM imports a top-level `model` package. Keep a previously loaded cache or
    # another bundle from silently supplying the code for this verified bundle.
    with _MODEL_IMPORT_LOCK:
        previous = {name: module for name, module in sys.modules.copy().items()
                    if name == "model" or name.startswith("model.")}
        for name in previous:
            sys.modules.pop(name, None)
        try:
            return torch.hub.load(bundle["code_path"], model_name, source="local", pretrained=False)
        finally:
            for name in list(sys.modules):
                if name == "model" or name.startswith("model."):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)


class RVMRunner:
    def __init__(
        self,
        model_name: str = "mobilenetv3",
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        downsample_ratio: float | str = 1.0,
        bundle_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.downsample_ratio = (
            "auto" if downsample_ratio == "auto" else float(downsample_ratio)
        )
        if self.downsample_ratio != "auto" and not 0 < self.downsample_ratio <= 1:
            raise ValueError("downsample_ratio must be 'auto' or in (0, 1].")
        if bundle_path is not None and checkpoint is not None:
            raise ValueError("RVM bundle_path and checkpoint cannot be configured together.")
        checkpoint_path = Path(checkpoint) if checkpoint is not None else None
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise FileNotFoundError(f"RVM checkpoint does not exist or is not a file: {checkpoint_path}")
        self.bundle_info = None
        if bundle_path is not None:
            from cv_human_video_matting.utils.model_bundle import verify_model_bundle
            self.bundle_info = verify_model_bundle(bundle_path, expected_model=model_name)
            self.model = _load_bundle_model(self.bundle_info, model_name)
            state = torch.load(self.bundle_info["weights_file"], map_location=self.device,
                               weights_only=True)
            self.model.load_state_dict(state)
        else:
            with _MODEL_IMPORT_LOCK:
                self.model = torch.hub.load(
                    RVM_HUB_REF, model_name, pretrained=checkpoint_path is None, trust_repo=True,
                )
            if checkpoint_path is not None:
                state = torch.load(str(checkpoint_path), map_location=self.device)
                self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self._rec = [None, None, None, None]
        self._frame_shape = None

    def reset(self) -> None:
        """Discard recurrent state between videos or resolution changes."""
        self._rec = [None, None, None, None]
        self._frame_shape = None

    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        if self._frame_shape != frame_bgr.shape:
            self.reset()
            self._frame_shape = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        with torch.no_grad():
            ratio = (
                min(1.0, 512.0 / max(frame_bgr.shape[:2]))
                if self.downsample_ratio == "auto" else self.downsample_ratio
            )
            output = self.model(tensor, *self._rec, downsample_ratio=ratio)

        foreground, alpha, *rec = output
        self._rec = rec[:4]

        alpha_np = alpha[0, 0].detach().cpu().numpy()
        alpha_u8 = np.clip(alpha_np * 255.0, 0, 255).astype(np.uint8)

        foreground_np = foreground[0].detach().cpu().permute(1, 2, 0).numpy()
        foreground_u8 = np.clip(foreground_np * 255.0, 0, 255).astype(np.uint8)
        foreground_bgr = cv2.cvtColor(foreground_u8, cv2.COLOR_RGB2BGR)

        return MattingResult(alpha=alpha_u8, foreground=foreground_bgr)
