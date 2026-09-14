import numpy as np
import torch
import pytest

from cv_human_video_matting.matting.rvm import RVMRunner


class _FakeRVMModel:
    def __init__(self):
        self.downsample_ratio = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, tensor, *rec, downsample_ratio=1):
        self.downsample_ratio = downsample_ratio
        batch, _channels, height, width = tensor.shape
        foreground = torch.zeros((batch, 3, height, width), dtype=torch.float32)
        alpha = torch.ones((batch, 1, height, width), dtype=torch.float32)
        return [foreground, alpha, None, None, None, None]


def test_rvm_runner_uses_numeric_downsample_ratio(monkeypatch):
    fake_model = _FakeRVMModel()

    def fake_load(*args, **kwargs):
        return fake_model

    monkeypatch.setattr(torch.hub, "load", fake_load)
    runner = RVMRunner(device="cpu")

    result = runner.predict(np.zeros((8, 10, 3), dtype=np.uint8))

    assert fake_model.downsample_ratio == 1.0
    assert result.alpha.shape == (8, 10)


def test_rvm_runner_pins_torch_hub_ref_to_avoid_default_branch_probe(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeRVMModel()

    monkeypatch.setattr(torch.hub, "load", fake_load)

    RVMRunner(device="cpu")

    args, kwargs = calls[0]
    assert args[0] == "PeterL1n/RobustVideoMatting:master"
    assert kwargs["trust_repo"] is True


def test_rvm_resets_recurrent_state_for_new_resolution(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: _FakeRVMModel())
    runner = RVMRunner(device="cpu")
    runner.predict(np.zeros((8, 10, 3), np.uint8))
    runner._rec = ["old"] * 4
    runner.reset()
    assert runner._rec == [None] * 4
    runner.predict(np.zeros((12, 16, 3), np.uint8))
    assert runner._frame_shape == (12, 16, 3)


def test_rvm_auto_scale_limits_internal_long_side(monkeypatch):
    model = _FakeRVMModel()
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: model)
    runner = RVMRunner(downsample_ratio="auto")
    result = runner.predict(np.zeros((576, 1024, 3), np.uint8))
    assert model.downsample_ratio == 0.5
    assert result.alpha.shape == (576, 1024)


@pytest.mark.parametrize("ratio", [0, -0.5, 2, "invalid"])
def test_rvm_rejects_invalid_scale_before_loading_model(ratio):
    with pytest.raises(ValueError):
        RVMRunner(downsample_ratio=ratio)
