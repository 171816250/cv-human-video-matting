"""Exercise BiRefNet prediction and rendering without downloading weights."""

import numpy as np
import pytest
import torch

from cv_human_video_matting.matting.base import MattingResult
from cv_human_video_matting.matting.birefnet import BiRefNetRunner
from cv_human_video_matting.pipeline import render_matting_frame


class _StubBiRefNet:
    def __init__(self):
        self.output = torch.zeros((1, 1, 3, 5), dtype=torch.float32)
        self.last_input = None

    def __call__(self, tensor):
        self.last_input = {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": tensor.device.type,
            "grad_enabled": torch.is_grad_enabled(),
            "pixel": tensor[0, :, 0, 0].cpu().numpy().copy(),
        }
        return self.output


@pytest.fixture
def stub_model(monkeypatch):
    model = _StubBiRefNet()
    monkeypatch.setattr(BiRefNetRunner, "_load_model", lambda self: model)
    return model


@pytest.mark.parametrize("output_kind", ["tensor", "list", "tuple"])
def test_predict_returns_alpha_at_original_resolution(stub_model, output_kind):
    if output_kind != "tensor":
        # Earlier decoder outputs deliberately disagree with the final prediction.
        outputs = [torch.full((1, 1, 1, 1), 9.0), stub_model.output]
        stub_model.output = outputs if output_kind == "list" else tuple(outputs)
    runner = BiRefNetRunner(device="cpu")
    frame = np.full((11, 17, 3), (10, 80, 200), dtype=np.uint8)

    result = runner.predict(frame)

    assert isinstance(result, MattingResult)
    assert result.foreground is None
    assert result.alpha.shape == (11, 17)
    assert result.alpha.dtype == np.float32
    # Zero logits give 0.5, including after resizing a non-square prediction.
    np.testing.assert_allclose(result.alpha, 0.5, rtol=0, atol=1e-6)


def test_predict_normalizes_rgb_input_without_modifying_frame(stub_model):
    runner = BiRefNetRunner(device="cpu")
    frame = np.full((7, 13, 3), (10, 80, 200), dtype=np.uint8)
    original = frame.copy()

    runner.predict(frame)

    received = stub_model.last_input
    assert received["shape"] == (1, 3, 1024, 1024)
    assert received["dtype"] == torch.float32
    assert received["device"] == "cpu"
    assert received["grad_enabled"] is False
    expected_rgb = np.array([200, 80, 10], dtype=np.float32) / 255
    expected_pixel = (expected_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(received["pixel"], expected_pixel, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(frame, original)


def test_predict_keeps_extreme_logits_finite_and_alpha_bounded(stub_model):
    stub_model.output = torch.tensor([[[[-1000.0, 1000.0]]]])
    runner = BiRefNetRunner(device="cpu")

    result = runner.predict(np.zeros((9, 15, 3), dtype=np.uint8))

    assert result.alpha.shape == (9, 15)
    assert np.isfinite(result.alpha).all()
    assert result.alpha.min() == 0.0
    assert result.alpha.max() == 1.0
    assert np.any((result.alpha > 0) & (result.alpha < 1))


@pytest.mark.parametrize("mode", ["green", "transparent"])
def test_birefnet_alpha_renders_with_original_frame(stub_model, mode):
    runner = BiRefNetRunner(device="cpu")
    frame = np.full((11, 17, 3), (0, 0, 255), dtype=np.uint8)

    rendered, alpha = render_matting_frame(frame, runner, mode=mode)

    assert alpha.dtype == np.uint8
    np.testing.assert_array_equal(alpha, np.full((11, 17), 127, dtype=np.uint8))
    assert rendered.dtype == np.uint8
    if mode == "transparent":
        assert rendered.shape == (11, 17, 4)
        np.testing.assert_array_equal(rendered[:, :, :3], frame)
        np.testing.assert_array_equal(rendered[:, :, 3], alpha)
    else:
        assert rendered.shape == frame.shape
        expected = np.full_like(frame, (0, 128, 127))
        np.testing.assert_allclose(rendered, expected, rtol=0, atol=1)
