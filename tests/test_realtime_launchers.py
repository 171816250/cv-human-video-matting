"""Realtime entrypoints preserve explicit paths and cannot hide CUDA failures."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from cv_human_video_matting import cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_service(monkeypatch):
    calls = []
    module = ModuleType("cv_human_video_matting.realtime.service")
    module.create_realtime_app = lambda *args, **kwargs: calls.append((args, kwargs)) or "app"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    server = ModuleType("uvicorn")
    server.run = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", server)
    monkeypatch.setattr(cli, "config_with_model_bundle", lambda config, path, **kwargs:
                        {"matting": {"bundle": str(Path(path).absolute())}, "device": {"mode": "cpu"}})
    return calls


def test_realtime_cli_caller_paths_explicit_cuda_and_bounded_server(fake_service, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli.main(["realtime", "--model-bundle", "my bundle", "--device", "cuda", "--port", "7992",
              "--max-width", "1280", "--mode", "alpha"])
    create, run = fake_service
    assert create == ((str(tmp_path / "my bundle"),), {"device": "cuda", "max_width": 1280, "mode": "alpha"})
    assert run[0] == ("app",)
    assert run[1] == {"host": "127.0.0.1", "port": 7992, "workers": 1, "proxy_headers": False,
                      "ws_max_size": 2 * 1024 * 1024, "ws_max_queue": 1, "access_log": False}


@pytest.mark.parametrize("args", [[], ["--port", "0"], ["--max-width", "1920"],
    ["--model-bundle", "x", "--host", "0.0.0.0", "--trust-local-proxy"]])
def test_realtime_cli_invalid_request_never_serves(fake_service, args):
    with pytest.raises(SystemExit) as error:
        cli.main(["realtime", *args])
    assert error.value.code == 2
    assert fake_service == []


def test_realtime_proxy_trust_is_explicit_and_limited_to_loopback(fake_service):
    cli.main(["realtime", "--model-bundle", "x", "--trust-local-proxy"])
    assert fake_service[-1][1]["proxy_headers"] is True
    assert fake_service[-1][1]["forwarded_allow_ips"] == "127.0.0.1,::1"


def load_starter():
    spec = importlib.util.spec_from_file_location("start_realtime_test", ROOT / "scripts/start_realtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_starter_checks_then_starts_without_touching_cpu_environment(monkeypatch, tmp_path):
    starter = load_starter()
    project = tmp_path / "project"
    project.mkdir()
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.setattr(starter, "PROJECT_ROOT", project)
    monkeypatch.setattr(starter, "project_python", lambda: Path(sys.executable))
    monkeypatch.chdir(caller)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")
    bundles, calls = [], []
    def select(config, bundle):
        bundles.append(bundle)
        return {**config, "matting": {"bundle": str(bundle)}}
    def capture(args):
        config = yaml.safe_load(Path(args[1]).read_text(encoding="utf-8"))
        calls.append((args, config, Path.cwd()))
    monkeypatch.setattr(cli, "config_with_model_bundle", select)
    monkeypatch.setattr(cli, "main", capture)
    assert starter.main(["--model-bundle", "../模型", "--port", "7998", "--max-width", "960"]) == 0
    assert bundles == [tmp_path / "模型"]
    assert len(calls) == 2 and calls[0][0][2:] == ["doctor"]
    assert calls[1][0][2:] == ["realtime", "--host", "127.0.0.1", "--port", "7998",
                               "--max-width", "960", "--device", "cuda"]
    assert all(call[1]["device"]["mode"] == "cuda" and call[2] == project for call in calls)
    assert not Path(calls[0][0][1]).exists()
    assert Path.cwd() == caller


def test_gpu_starter_diagnostic_failure_prevents_launch(monkeypatch, tmp_path):
    starter = load_starter()
    monkeypatch.setattr(starter, "project_python", lambda: Path(sys.executable))
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(cli, "config_with_model_bundle", lambda config, bundle:
                        {**config, "matting": {"bundle": str(bundle)}})
    calls = []
    def fail(args):
        calls.append(args)
        raise SystemExit(1)
    monkeypatch.setattr(cli, "main", fail)
    with pytest.raises(SystemExit) as error:
        starter.main(["--model-bundle", str(tmp_path)])
    assert error.value.code == 1
    assert len(calls) == 1
    assert not Path(calls[0][1]).exists()


def test_gpu_starter_missing_environment_never_installs(monkeypatch, tmp_path):
    starter = load_starter()
    monkeypatch.setattr(starter, "project_python", lambda: tmp_path / "missing.exe")
    monkeypatch.setattr(starter.subprocess, "run", lambda *args, **kwargs: pytest.fail("no subprocess"))
    with pytest.raises(SystemExit) as error:
        starter.main([])
    assert error.value.code == 2


def test_gpu_starter_reexecutes_gpu_interpreter_only(monkeypatch, tmp_path):
    starter = load_starter()
    target = tmp_path / "GPU 环境.exe"
    target.touch()
    monkeypatch.setattr(starter, "project_python", lambda: target)
    calls = []
    monkeypatch.setattr(starter.subprocess, "run", lambda args, **kwargs:
                        calls.append(args) or SimpleNamespace(returncode=7))
    assert starter.main(["--port", "7990"]) == 7
    assert calls[0] == [str(target), "-B", str(ROOT / "scripts/start_realtime.py"), "--port", "7990"]
