"""Offline tests must not silently download large models."""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"

import pytest
import torch


@pytest.fixture(autouse=True)
def block_model_downloads(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Mock torch.hub.load in unit tests; real models require a separate smoke run.")
    monkeypatch.setattr(torch.hub, "load", blocked)
