"""BiRefNet high-precision matting runner for SOTA edge quality.

BiRefNet (Bilateral Reference Network) is a state-of-the-art image segmentation
model specialized for salient object detection with exceptional edge precision.
It processes single frames (no temporal state) and is more compute-intensive
than RVM, making it suitable for high-quality offline rendering.

Reference: https://github.com/ZhengPeng7/BiRefNet
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cv_human_video_matting.matting.base import MattingResult, MattingRunner


class BiRefNetRunner(MattingRunner):
    """BiRefNet matting runner for high-precision alpha matte extraction.

    This runner provides SOTA edge quality but is slower than RVM.
    Use for high-quality offline rendering where speed is not critical.

    Args:
        device: Torch device ("cpu" or "cuda").
        model_id: HuggingFace model ID or local path.

    Attributes:
        device: Inference device.
        model: Loaded BiRefNet model.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_id: str = "ZhengPeng7/BiRefNet",
    ):
        self.device = device
        self.model_id = model_id
        self.model = self._load_model()

    def _load_model(self):
        """Load BiRefNet model with HuggingFace mirror support for China."""
        local_path = Path("weights/BiRefNet")

        # Temporarily disable proxy to avoid blocking mirror access
        old_http_proxy = os.environ.pop("HTTP_PROXY", None)
        old_https_proxy = os.environ.pop("HTTPS_PROXY", None)

        # Use HuggingFace mirror for faster downloads in China
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        try:
            from transformers import AutoModelForImageSegmentation

            # Try loading from local cache first
            if local_path.exists():
                model = AutoModelForImageSegmentation.from_pretrained(
                    str(local_path),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            else:
                # Download from HuggingFace (or mirror)
                model = AutoModelForImageSegmentation.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                # Cache locally for offline use
                local_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(local_path))

            return model.eval().to(self.device)

        finally:
            # Restore original proxy settings
            if old_http_proxy is not None:
                os.environ["HTTP_PROXY"] = old_http_proxy
            if old_https_proxy is not None:
                os.environ["HTTPS_PROXY"] = old_https_proxy

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        """Run BiRefNet inference on a BGR frame.

        Args:
            frame_bgr: Input frame in BGR format (HxWx3, uint8).

        Returns:
            MattingResult with alpha matte (no foreground or state).
        """
        h, w = frame_bgr.shape[:2]

        # Convert BGR to RGB
        rgb = frame_bgr[:, :, ::-1].copy()

        # Convert to tensor and normalize to [0, 1]
        src = torch.from_numpy(rgb).float().div(255.0)
        src = src.permute(2, 0, 1).unsqueeze(0)  # CHW -> NCHW

        # Resize to fixed input size (1024x1024 for BiRefNet)
        src = F.interpolate(
            src,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        src = ((src - mean) / std).to(self.device)

        # Inference
        output = self.model(src)

        # Extract prediction (model may return list or single tensor)
        if isinstance(output, (list, tuple)):
            pred = output[-1]  # Use last output
        else:
            pred = output

        # Apply sigmoid and extract single-channel alpha
        alpha = pred.sigmoid()[0, 0]

        # Upsample back to original resolution
        alpha = F.interpolate(
            alpha.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        # Convert to numpy
        alpha_np = alpha.clamp(0, 1).cpu().numpy().astype(np.float32)

        return MattingResult(
            alpha=alpha_np,
            foreground=None,  # BiRefNet only outputs alpha
            state=None,  # No temporal state
        )

    def reset(self) -> None:
        """Reset temporal state (no-op for BiRefNet, included for protocol compliance)."""
        pass
