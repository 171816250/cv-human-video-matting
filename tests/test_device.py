import pytest

from cv_human_video_matting.utils import device as device_utils


def test_resolve_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: False)

    assert device_utils.resolve_device("auto") == "cpu"


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: True)

    assert device_utils.resolve_device("auto") == "cuda"


@pytest.mark.parametrize("requested", ["cuda", "cuda:0", "CUDA:1"])
def test_explicit_cuda_does_not_silently_fall_back_when_unavailable(monkeypatch, requested):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="explicitly requested.*unavailable"):
        device_utils.resolve_device(requested)


@pytest.mark.parametrize("requested,expected", [
    ("cuda", "cuda"), ("cuda:0", "cuda:0"), (" CUDA:1 ", "cuda:1"), ("cuda:01", "cuda:1"),
])
def test_resolve_device_cuda_available_and_index_valid(monkeypatch, requested, expected):
    import torch

    monkeypatch.setattr(device_utils, "cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert device_utils.resolve_device(requested) == expected


@pytest.mark.parametrize("requested", ["cuda:2", "cuda:999"])
def test_cuda_index_out_of_range_is_rejected_before_inference(monkeypatch, requested):
    import torch

    monkeypatch.setattr(device_utils, "cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="out of range: 2 visible"):
        device_utils.resolve_device(requested)


@pytest.mark.parametrize("requested", [
    "cudabanana", "cuda:-1", "cuda:", "cuda:abc", "cuda:1:2", "cuda:1.0", "cuda: 0",
    "cpu:0", "mps", "", "  ", 0, False,
])
def test_invalid_device_is_rejected_even_without_cuda(monkeypatch, requested):
    def must_not_probe():
        raise AssertionError("Validate the device syntax before probing CUDA")

    monkeypatch.setattr(device_utils, "cuda_available", must_not_probe)
    with pytest.raises(ValueError, match="[Dd]evice"):
        device_utils.resolve_device(requested)


def test_explicit_cpu_never_probes_cuda(monkeypatch):
    def must_not_probe():
        raise AssertionError("CPU selection must not depend on CUDA")

    monkeypatch.setattr(device_utils, "cuda_available", must_not_probe)
    assert device_utils.resolve_device(" CPU ") == "cpu"


def test_none_retains_auto_selection(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: False)
    assert device_utils.resolve_device(None) == "cpu"


@pytest.mark.parametrize("failure", [ImportError("missing torch"), OSError("[WinError 126] DLL load failed")])
def test_torch_import_failure_allows_auto_but_rejects_explicit_cuda(monkeypatch, failure):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise failure
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # A broken torch install (non-ImportError) must not crash the caller.
    assert device_utils.cuda_available() is False
    assert device_utils.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        device_utils.resolve_device("cuda:0")


def test_cuda_probe_failure_allows_auto_but_rejects_explicit_cuda(monkeypatch):
    import torch

    def broken_probe():
        raise RuntimeError("driver initialization failed")

    monkeypatch.setattr(torch.cuda, "is_available", broken_probe)
    assert device_utils.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="failed to initialize"):
        device_utils.resolve_device("cuda")


def test_cuda_enumeration_failure_is_reported(monkeypatch):
    import torch

    def broken_count():
        raise RuntimeError("device enumeration failed")

    monkeypatch.setattr(device_utils, "cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", broken_count)
    with pytest.raises(RuntimeError, match="Unable to enumerate") as raised:
        device_utils.resolve_device("cuda:0")
    assert str(raised.value.__cause__) == "device enumeration failed"

