"""Explicit RVM checkpoint selection must be deterministic and work offline."""

import pytest
import torch

from cv_human_video_matting.matting.rvm import RVM_HUB_REF, RVMRunner


class _Model:
    def __init__(self):
        self.state = None
        self.device = None
        self.evaluating = False

    def load_state_dict(self, state):
        self.state = state

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self


@pytest.mark.parametrize("is_directory", [False, True])
def test_invalid_explicit_checkpoint_is_rejected_before_hub_load(tmp_path, is_directory):
    checkpoint = tmp_path / "model.pth"
    if is_directory:
        checkpoint.mkdir()

    # The autouse fixture rejects any hub call, including implicit downloads.
    with pytest.raises(FileNotFoundError, match="RVM checkpoint"):
        RVMRunner(checkpoint=checkpoint)


def test_explicit_checkpoint_loads_local_weights_without_pretrained_download(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"mock checkpoint")
    model = _Model()
    state = {"custom.weight": object()}
    hub_calls = []
    checkpoint_calls = []

    def load_model(*args, **kwargs):
        hub_calls.append((args, kwargs))
        return model

    def load_checkpoint(path, *, map_location):
        checkpoint_calls.append((path, map_location))
        return state

    monkeypatch.setattr(torch.hub, "load", load_model)
    monkeypatch.setattr(torch, "load", load_checkpoint)

    runner = RVMRunner(model_name="resnet50", checkpoint=checkpoint, device="cpu")

    assert hub_calls == [
        ((RVM_HUB_REF, "resnet50"), {"pretrained": False, "trust_repo": True})
    ]
    assert checkpoint_calls == [(str(checkpoint), torch.device("cpu"))]
    assert runner.model is model
    assert model.state is state
    assert model.device == torch.device("cpu")
    assert model.evaluating


def test_default_checkpoint_uses_upstream_pretrained_weights(monkeypatch):
    model = _Model()
    calls = []

    def load_model(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    def unexpected_checkpoint_load(*args, **kwargs):
        raise AssertionError("No local checkpoint was selected")

    monkeypatch.setattr(torch.hub, "load", load_model)
    monkeypatch.setattr(torch, "load", unexpected_checkpoint_load)

    runner = RVMRunner(checkpoint=None)

    assert calls == [
        ((RVM_HUB_REF, "mobilenetv3"), {"pretrained": True, "trust_repo": True})
    ]
    assert runner.model is model
    assert model.state is None
