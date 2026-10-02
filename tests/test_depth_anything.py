"""Offline contract tests for the optional Depth Anything V2 Small backend."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from cv_human_video_matting.depth import (
    DEPTH_ANYTHING_V2_SMALL,
    DepthEstimator,
    validate_depth_checkpoint,
)


@pytest.fixture
def local_checkpoint(tmp_path):
    """Only a fake local snapshot; tests never fetch or load real weights."""
    directory = tmp_path / "depth-anything-v2-small"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"model_type": "depth_anything",
                    "backbone": None, "backbone_config": {"model_type": "dinov2",
                                                          "hidden_size": 384, "patch_size": 14}}),
        encoding="utf-8",
    )
    (directory / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"fake offline weights")
    return directory


def test_missing_local_checkpoint_is_clear_and_does_not_load(monkeypatch, tmp_path):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("preflight must not load a model")

    monkeypatch.setattr(torch.hub, "load", forbidden_load)
    with pytest.raises(FileNotFoundError, match="depth.checkpoint.*No model is downloaded"):
        DepthEstimator(model_type=DEPTH_ANYTHING_V2_SMALL)
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        validate_depth_checkpoint(DEPTH_ANYTHING_V2_SMALL, tmp_path / "missing")
    validate_depth_checkpoint(DEPTH_ANYTHING_V2_SMALL, None, enabled=False)
    validate_depth_checkpoint("MiDaS_small", None)


def test_incomplete_snapshot_and_lfs_pointer_are_rejected(tmp_path, local_checkpoint):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json.*preprocessor_config.json.*model.safetensors"):
        validate_depth_checkpoint(DEPTH_ANYTHING_V2_SMALL, incomplete)

    (local_checkpoint / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        validate_depth_checkpoint(DEPTH_ANYTHING_V2_SMALL, local_checkpoint)


def test_metric_checkpoint_is_rejected(local_checkpoint):
    (local_checkpoint / "config.json").write_text(
        json.dumps({"model_type": "depth_anything", "depth_estimation_type": "metric"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative, not metric"):
        validate_depth_checkpoint(DEPTH_ANYTHING_V2_SMALL, local_checkpoint)


def test_external_backbone_loading_is_rejected(local_checkpoint):
    (local_checkpoint / "config.json").write_text(
        json.dumps({"model_type": "depth_anything", "use_pretrained_backbone": True,
                    "backbone_config": {"model_type": "dinov2",
                                        "hidden_size": 384, "patch_size": 14}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="use_pretrained_backbone must be false"):
        validate_depth_checkpoint(DEPTH_ANYTHING_V2_SMALL, local_checkpoint)


@pytest.mark.parametrize("change", [
    {"backbone": "remote/model"},
    {"use_timm_backbone": True},
    {"backbone_kwargs": {"pretrained": True}},
    {"backbone_config": None},
    {"backbone_config": {"model_type": "dinov2", "hidden_size": 768, "patch_size": 14}},
])
def test_nested_backbone_download_routes_are_rejected_before_model_load(
    local_checkpoint, monkeypatch, change,
):
    config_path = local_checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(change)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    fake_transformers = ModuleType("transformers")

    class ForbiddenModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pytest.fail("No model load or nested download is allowed")

    fake_transformers.AutoImageProcessor = ForbiddenModel
    fake_transformers.AutoModelForDepthEstimation = ForbiddenModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    with pytest.raises(ValueError, match="DINOv2 backbone_config"):
        DepthEstimator(model_type=DEPTH_ANYTHING_V2_SMALL, checkpoint_path=local_checkpoint)


def test_optional_dependency_error_after_local_preflight(local_checkpoint, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="optional Transformers dependency"):
        DepthEstimator(model_type=DEPTH_ANYTHING_V2_SMALL, checkpoint_path=local_checkpoint)


def test_offline_adapter_preserves_rgb_shape_and_relative_distance(
    local_checkpoint, monkeypatch,
):
    calls = []
    seen_images = []

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", path, kwargs))
            return cls()

        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            seen_images.append(images.copy())
            return {"pixel_values": torch.zeros((1, 3, 12, 12))}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return cls()

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, **inputs):
            assert inputs["pixel_values"].shape == (1, 3, 12, 12)
            raw = torch.zeros((1, 12, 12))
            raw[:, :, 6:] = 10
            return SimpleNamespace(predicted_depth=raw)

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = Processor
    fake_transformers.AutoModelForDepthEstimation = Model
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def forbidden_hub(*args, **kwargs):
        raise AssertionError("Depth Anything must not use torch.hub")

    monkeypatch.setattr(torch.hub, "load", forbidden_hub)
    estimator = DepthEstimator(
        device="cpu", model_type=DEPTH_ANYTHING_V2_SMALL,
        checkpoint_path=local_checkpoint,
    )
    frame = np.full((12, 12, 3), (10, 20, 30), dtype=np.uint8)
    depth = estimator.estimate(frame)

    assert depth.shape == frame.shape[:2]
    assert depth.dtype == np.float32
    assert depth.min() == 0.0 and depth.max() == 1.0
    np.testing.assert_array_equal(seen_images[0][0, 0], [30, 20, 10])
    assert all(path == str(local_checkpoint) for _, path, _ in calls)
    assert all(kwargs["local_files_only"] is True for _, _, kwargs in calls)
    assert all(kwargs["trust_remote_code"] is False for _, _, kwargs in calls)
    assert calls[1][2]["use_safetensors"] is True

    person_alpha = np.zeros((12, 12), dtype=np.float32)
    person_alpha[:, 6:] = 1.0
    distance, label = estimator.person_distance(depth, person_alpha)
    assert label == "Near"
    assert distance == 0.0
