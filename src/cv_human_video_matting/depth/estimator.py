"""Distance estimation module using MiDaS for monocular depth prediction.

MiDaS outputs relative inverse depth (higher values = closer to camera).
We compute median depth within the person's alpha region and classify into
Near/Mid/Far distance tiers for labeling.

Note: This is relative distance estimation, not absolute metric depth.
For precise distance measurement, stereo cameras or ToF sensors are required.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from cv_human_video_matting.render.compositor import normalize_alpha


class DepthEstimator:
    """Monocular depth estimator using MiDaS for person distance labeling.

    Args:
        device: Torch device ("cpu" or "cuda").
        model_type: MiDaS model variant ("MiDaS_small" for speed, "DPT_Large" for quality).

    Attributes:
        device: Inference device.
        model: Loaded MiDaS model.
        transform: Input preprocessing transform.
    """

    def __init__(self, device: str = "cpu", model_type: str = "MiDaS_small"):
        self.device = device
        self.model_type = model_type

        # Load MiDaS model from torch.hub
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.eval().to(device)

        # Load corresponding transform
        transforms_module = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if "small" in model_type.lower():
            self.transform = transforms_module.small_transform
        else:
            self.transform = transforms_module.dpt_transform

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate normalized inverse depth map for a BGR frame.

        Args:
            frame_bgr: Input frame in BGR format (HxWx3, uint8).

        Returns:
            Normalized inverse depth map (HxW, float32, 0~1 range).
            Higher values indicate closer distance to camera.
        """
        import cv2

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform and inference
        input_tensor = self.transform(rgb).to(self.device)
        prediction = self.model(input_tensor)

        # Upsample to original resolution
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalize to 0~1 range (inverse depth)
        depth_min = prediction.min()
        depth_max = prediction.max()
        normalized = (prediction - depth_min) / (depth_max - depth_min + 1e-6)

        return normalized.astype(np.float32)

    def person_distance(
        self,
        depth: np.ndarray,
        alpha: np.ndarray,
        thresh: float = 0.5,
    ) -> tuple[float | None, str]:
        """Compute person distance tier from depth map and alpha matte.

        Args:
            depth: Normalized inverse depth map (HxW, float32, 0~1).
            alpha: Alpha matte (HxW or HxWx1, float32, 0~1).
            thresh: Alpha threshold to consider as person region.

        Returns:
            Tuple of (distance_value, distance_label):
            - distance_value: Relative distance (0~1, higher=farther), or None if no person.
            - distance_label: "Near", "Mid", "Far", or "No person detected".
        """
        # Ensure alpha is 2D
        alpha = normalize_alpha(alpha)[:, :, 0]
        if depth.shape != alpha.shape:
            raise ValueError("Depth and alpha must have matching spatial dimensions.")

        # Create person mask
        mask = (alpha > thresh) & np.isfinite(depth)

        # Check if person region is large enough
        if mask.sum() < 50:
            return None, "No person detected"

        # Compute median inverse depth in person region (robust to outliers)
        inverse_depth = float(np.median(depth[mask]))

        # Convert to distance feeling (1 - inv_depth, so higher = farther)
        distance_value = 1.0 - inverse_depth

        # Classify into distance tiers
        if inverse_depth > 0.66:
            label = "Near"
        elif inverse_depth > 0.33:
            label = "Mid"
        else:
            label = "Far"

        return distance_value, label
