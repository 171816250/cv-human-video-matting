from cv_human_video_matting.utils import device as device_utils


def test_resolve_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: False)

    assert device_utils.resolve_device("auto") == "cpu"


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: True)

    assert device_utils.resolve_device("auto") == "cuda"


def test_resolve_device_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: False)

    assert device_utils.resolve_device("cuda") == "cpu"


def test_resolve_device_cuda_indexed_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: False)

    # "cuda:0" and other CUDA-family strings must also degrade to CPU.
    assert device_utils.resolve_device("cuda:0") == "cpu"
    assert device_utils.resolve_device("CUDA:1") == "cpu"


def test_resolve_device_cuda_indexed_preserved_when_available(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_available", lambda: True)

    assert device_utils.resolve_device("cuda:0") == "cuda:0"


def test_cuda_available_returns_false_when_torch_import_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("[WinError 126] DLL load failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # A broken torch install (non-ImportError) must not crash the caller.
    assert device_utils.cuda_available() is False

