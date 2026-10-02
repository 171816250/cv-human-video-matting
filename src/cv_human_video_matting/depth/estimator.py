"""Relative distance estimation using MiDaS or experimental Depth Anything V2 Small.

Both backends output scene-relative inverse depth (higher values = closer to camera).
We compute median depth within the person's alpha region and classify into
Near/Mid/Far distance tiers for labeling.

Note: This is relative distance estimation, not absolute metric depth.
For precise distance measurement, stereo cameras or ToF sensors are required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cv_human_video_matting.render.compositor import normalize_alpha
from cv_human_video_matting.utils.config import DEPTH_MODELS

DEPTH_ANYTHING_V2_SMALL = "DepthAnythingV2_small"
_DEPTH_ANYTHING_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def validate_depth_checkpoint(
    model_type: str,
    checkpoint_path: str | Path | None,
    enabled: bool = True,
) -> None:
    """Check local V2 Small assets before output is opened; never load or download a model.

    ``checkpoint_path`` is a local snapshot of the official Hugging Face
    ``depth-anything/Depth-Anything-V2-Small-hf`` repository, not a model ID or
    the original project's ``.pth`` checkpoint. MiDaS keeps its existing hub path.
    """
    if not enabled or model_type != DEPTH_ANYTHING_V2_SMALL:
        return
    hint = (
        "Depth Anything V2 Small is experimental and requires a complete local "
        "depth-anything/Depth-Anything-V2-Small-hf directory; set depth.checkpoint "
        "or --depth-checkpoint. No model is downloaded automatically."
    )
    if checkpoint_path is None or not isinstance(checkpoint_path, (str, Path)):
        raise FileNotFoundError(hint)
    if isinstance(checkpoint_path, str) and not checkpoint_path.strip():
        raise FileNotFoundError(hint)
    directory = Path(checkpoint_path).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"{hint} Directory not found: {directory}")
    missing = [name for name in _DEPTH_ANYTHING_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{hint} Missing in {directory}: {', '.join(missing)}"
        )
    with (directory / "model.safetensors").open("rb") as handle:
        lfs_pointer = handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    if lfs_pointer:
        raise FileNotFoundError(
            f"{hint} {directory / 'model.safetensors'} is a Git LFS pointer, not downloaded weights."
        )
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Depth Anything V2 config.json in {directory}: {exc}") from exc
    if not isinstance(config, dict) or config.get("model_type") != "depth_anything":
        raise ValueError(
            f"{directory} is not a Transformers Depth Anything checkpoint. "
            "Use depth-anything/Depth-Anything-V2-Small-hf."
        )
    if config.get("depth_estimation_type", "relative") != "relative":
        raise ValueError("Depth Anything V2 checkpoint must estimate relative, not metric depth.")
    if config.get("use_pretrained_backbone", False):
        raise ValueError(
            "Depth Anything V2 local checkpoint must include its backbone weights; "
            "use_pretrained_backbone must be false to avoid separate model loading."
        )
    # Transformers can fetch a nested backbone even when the outer
    # from_pretrained() call uses local_files_only=True. The official Small HF
    # snapshot embeds a DINOv2 config and has no external backbone reference.
    backbone_config = config.get("backbone_config")
    if (config.get("backbone") is not None or config.get("use_timm_backbone", False)
            or config.get("backbone_kwargs")
            or not isinstance(backbone_config, dict)
            or backbone_config.get("model_type") != "dinov2"
            or backbone_config.get("hidden_size") != 384
            or backbone_config.get("patch_size") != 14):
        raise ValueError(
            "Depth Anything V2 Small requires an embedded Small DINOv2 backbone_config "
            "with no external backbone, use_timm_backbone or backbone_kwargs."
        )


class DepthEstimator:
    """Monocular depth estimator for scene-relative person distance labeling.

    Args:
        device: Torch device ("cpu" or "cuda").
        model_type: MiDaS variant or experimental ``DepthAnythingV2_small``.
        near_threshold: Normalized inverse depth above which a region is labeled Near.
        mid_threshold: Normalized inverse depth above which a region is labeled Mid.
        checkpoint_path: Local Hugging Face V2 Small directory, required only for that backend.

    Attributes:
        device: Inference device.
        model: Loaded depth model.
        transform: Input preprocessing transform or image processor.
    """

    near_threshold = 0.66
    mid_threshold = 0.33

    def __init__(
        self,
        device: str = "cpu",
        model_type: str = "MiDaS_small",
        near_threshold: float = 0.66,
        mid_threshold: float = 0.33,
        checkpoint_path: str | Path | None = None,
    ):
        if not isinstance(model_type, str) or model_type not in DEPTH_MODELS | {
            DEPTH_ANYTHING_V2_SMALL
        }:
            raise ValueError(f"Unsupported depth model: {model_type}")
        if (
            isinstance(near_threshold, bool) or isinstance(mid_threshold, bool)
            or not isinstance(near_threshold, (int, float))
            or not isinstance(mid_threshold, (int, float))
            or not math.isfinite(near_threshold) or not math.isfinite(mid_threshold)
            or not 0 <= mid_threshold < near_threshold <= 1
        ):
            raise ValueError("Depth thresholds must satisfy 0 <= mid < near <= 1.")
        self.device = device
        self.model_type = model_type
        self.near_threshold = float(near_threshold)
        self.mid_threshold = float(mid_threshold)
        self.checkpoint_path = checkpoint_path

        if model_type == DEPTH_ANYTHING_V2_SMALL:
            validate_depth_checkpoint(model_type, checkpoint_path)
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            except ImportError as exc:
                raise RuntimeError(
                    "Experimental Depth Anything V2 Small requires the optional "
                    "Transformers dependency (see requirements-optional.txt)."
                ) from exc
            local_path = str(Path(checkpoint_path).expanduser())
            self.transform = AutoImageProcessor.from_pretrained(
                local_path, local_files_only=True, trust_remote_code=False,
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                local_path, local_files_only=True, trust_remote_code=False,
                use_safetensors=True,
            )
            self.model.eval().to(device)
            return

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

        if self.model_type == DEPTH_ANYTHING_V2_SMALL:
            # The official V2 Small HF processor accepts RGB NumPy images.
            inputs = self.transform(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            prediction = self.model(**inputs).predicted_depth
        else:
            # Preserve MiDaS preprocessing and inference behavior.
            input_tensor = self.transform(rgb).to(self.device)
            prediction = self.model(input_tensor)

        # Upsample to original resolution
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        )[0, 0].cpu().numpy()

        if self.model_type == DEPTH_ANYTHING_V2_SMALL:
            # V2 Small's relative output has disparity orientation (higher = nearer).
            # Keep the existing 0..1 distance-label contract across backends.
            if not np.isfinite(prediction).all():
                raise ValueError("Depth Anything V2 returned non-finite depth values.")
            depth_min = float(prediction.min())
            depth_max = float(prediction.max())
            if depth_max - depth_min <= 1e-6:
                return np.zeros(frame_bgr.shape[:2], dtype=np.float32)
            return ((prediction - depth_min) / (depth_max - depth_min)).astype(np.float32)

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
        if inverse_depth > self.near_threshold:
            label = "Near"
        elif inverse_depth > self.mid_threshold:
            label = "Mid"
        else:
            label = "Far"

        return distance_value, label
