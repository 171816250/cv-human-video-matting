"""Portable local launchers resolve paths and fail closed without starting services."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from cv_human_video_matting import cli, diagnostics
from cv_human_video_matting.utils import model_bundle


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_ui(monkeypatch):
    calls = []
    module = ModuleType("cv_human_video_matting.ui.gradio_app")
    module.launch_demo = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.fixture
def make_bundle(tmp_path):
    def create(destination, family="mobilenetv3"):
        cache = tmp_path / ("cache-" + family)
        code = cache / model_bundle.CACHE_CODE_DIR
        for name in model_bundle._REQUIRED_CODE:
            path = code / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# local fixture\n", encoding="utf-8")
        weights = cache / "checkpoints" / f"rvm_{family}.pth"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"small fixture, never loaded")
        return model_bundle.prepare_model_bundle(destination, family, cache)
    return create


def test_cli_default_config_is_independent_of_cwd(tmp_path, monkeypatch, fake_ui):
    monkeypatch.chdir(tmp_path)
    cli.main(["ui", "--port", "7991"])
    assert fake_ui[0]["config"]["project"]["name"] == "cv-human-video-matting"
    assert fake_ui[0]["server_port"] == 7991
    assert Path.cwd() == tmp_path


def test_cli_explicit_config_still_uses_caller_cwd(tmp_path, monkeypatch, fake_ui):
    (tmp_path / "custom.yaml").write_text("project:\n  name: caller-config\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cli.main(["--config", "custom.yaml", "ui"])
    assert fake_ui[0]["config"]["project"]["name"] == "caller-config"


def test_doctor_forwards_selected_requirements(tmp_path, monkeypatch):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(diagnostics, "check_environment", lambda config, **kwargs:
                        calls.append((config, kwargs)) or {"checks": [], "ready": True})
    cli.main(["--config", "relative.yaml", "doctor", "--requirements", "runtime.txt"])
    assert calls == [("relative.yaml", {"requirements_path": "runtime.txt"})]


def test_ui_bundle_flag_selects_manifest_model_and_clears_checkpoint(tmp_path, monkeypatch, fake_ui, make_bundle):
    bundle = make_bundle(tmp_path / "模型 包", "resnet50")
    config = tmp_path / "old.yaml"
    config.write_text("matting:\n  model: rvm_mobilenetv3\n  checkpoint: old.pth\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cli.main(["--config", "old.yaml", "ui", "--model-bundle", bundle.name])
    assert fake_ui[0]["config"]["matting"] == {
        "model": "rvm_resnet50", "bundle": str(bundle), "checkpoint": None}
    assert "old.pth" in config.read_text(encoding="utf-8")


def test_bad_bundle_blocks_ui_before_launch(tmp_path, fake_ui, make_bundle):
    bundle = make_bundle(tmp_path / "bundle")
    (bundle / "weights/rvm_mobilenetv3.pth").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        cli.main(["ui", "--model-bundle", str(bundle)])
    assert fake_ui == []


def test_configured_bundle_model_mismatch_does_not_silently_switch(tmp_path, fake_ui, make_bundle):
    bundle = make_bundle(tmp_path / "bundle", "resnet50")
    config = tmp_path / "wrong.yaml"
    config.write_text(yaml.safe_dump({"matting": {"model": "rvm_mobilenetv3", "bundle": str(bundle)}}))
    with pytest.raises(ValueError, match="model mismatch"):
        cli.main(["--config", str(config), "ui"])
    assert not fake_ui


@pytest.mark.parametrize("port", ["0", "65536", "-2"])
def test_invalid_ui_port_fails_before_launch(port, fake_ui):
    with pytest.raises(SystemExit):
        cli.main(["ui", "--port", port])
    assert not fake_ui


def test_legacy_launcher_resolves_explicit_paths_and_forwards_options(tmp_path, monkeypatch):
    launcher = load_script("launch_gradio")
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "main", lambda args: calls.append((args, Path.cwd())))
    launcher.main(["--config", "my config.yaml", "--model-bundle", "模型 包",
                   "--host", "localhost", "--port", "7992", "--share"])
    args, cwd = calls[0]
    assert args == ["--config", str(tmp_path / "my config.yaml"), "ui", "--host", "localhost",
                    "--port", "7992", "--share", "--model-bundle", str(tmp_path / "模型 包")]
    assert cwd == ROOT
    assert Path.cwd() == tmp_path


def test_legacy_launcher_rejects_unknown_arguments(monkeypatch):
    launcher = load_script("launch_gradio")
    monkeypatch.setattr(cli, "main", lambda args: pytest.fail("must not launch"))
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--unknown"])
    assert exc.value.code == 2


@pytest.fixture
def starter(tmp_path, monkeypatch):
    launcher = load_script("start_demo")
    project = tmp_path / "交付 项目'" / "source"
    (project / "configs").mkdir(parents=True)
    (project / "configs/default.yaml").write_text("device:\n  mode: cpu\ndepth:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "PROJECT_ROOT", project)
    monkeypatch.setattr(launcher, "project_python", lambda: Path(sys.executable))
    return launcher


def test_starter_chooses_unique_verified_evidence_bundle(starter, make_bundle):
    target = starter.PROJECT_ROOT.parent / "evidence" / "portable bundle"
    make_bundle(target)
    assert starter.discover_model_bundle() == target


def test_starter_rejects_ambiguous_bundles(starter, make_bundle):
    make_bundle(starter.PROJECT_ROOT / "weights/a")
    make_bundle(starter.PROJECT_ROOT.parent / "evidence/b")
    with pytest.raises(ValueError, match="2 verified"):
        starter.discover_model_bundle()


def test_starter_does_not_accept_unrelated_or_damaged_manifest(starter, make_bundle):
    unrelated = starter.PROJECT_ROOT / "weights/unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "manifest.json").write_text(json.dumps({"kind": "something_else"}))
    damaged = make_bundle(starter.PROJECT_ROOT / "weights/damaged")
    (damaged / "weights/rvm_mobilenetv3.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="No verified"):
        starter.discover_model_bundle()


def test_starter_checks_exact_effective_config_then_starts_local_ui(starter, tmp_path, monkeypatch, make_bundle):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")
    bundle = make_bundle(tmp_path / "selected 模型")
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    calls = []

    def capture(args):
        assert starter.os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
        config_path = Path(args[1])
        calls.append((args, yaml.safe_load(config_path.read_text(encoding="utf-8")), Path.cwd(), config_path))

    monkeypatch.setattr(cli, "main", capture)
    assert starter.main(["--model-bundle", "../selected 模型", "--port", "7993",
                         "--requirements", "runtime requirements.txt"]) == 0
    doctor, ui = calls
    assert doctor[0][2:] == ["doctor", "--requirements", str(caller / "runtime requirements.txt")]
    assert ui[0][2:] == ["ui", "--host", "127.0.0.1", "--port", "7993", "--no-share"]
    assert doctor[1] == ui[1]
    assert ui[1]["matting"]["bundle"] == str(bundle)
    assert ui[1]["device"]["mode"] == "cpu"
    assert ui[1]["depth"]["enabled"] is False
    assert doctor[2] == ui[2] == starter.PROJECT_ROOT
    assert not doctor[3].exists()
    assert Path.cwd() == caller


def test_starter_failed_doctor_blocks_ui_and_cleans_config(starter, tmp_path, monkeypatch, make_bundle):
    bundle = make_bundle(tmp_path / "bundle")
    calls = []
    previous = Path.cwd()

    def fail(args):
        calls.append(args)
        raise SystemExit(1)

    monkeypatch.setattr(cli, "main", fail)
    with pytest.raises(SystemExit) as exc:
        starter.main(["--model-bundle", str(bundle)])
    assert exc.value.code == 1
    assert len(calls) == 1 and calls[0][2] == "doctor"
    assert not Path(calls[0][1]).exists()
    assert Path.cwd() == previous


def test_starter_missing_project_environment_does_not_call_cli(starter, monkeypatch):
    monkeypatch.setattr(starter, "project_python", lambda: starter.PROJECT_ROOT / ".venv/Scripts/python.exe")
    monkeypatch.setattr(cli, "main", lambda args: pytest.fail("must not launch"))
    with pytest.raises(SystemExit) as exc:
        starter.main([])
    assert exc.value.code == 2


def test_starter_reexecutes_only_project_python_preserving_caller_cwd(starter, tmp_path, monkeypatch):
    interpreter = tmp_path / "中文 环境" / "python.exe"
    interpreter.parent.mkdir()
    interpreter.touch()
    monkeypatch.setattr(starter, "project_python", lambda: interpreter)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(starter.subprocess, "run", lambda args, **kwargs:
                        calls.append((args, kwargs, Path.cwd())) or SimpleNamespace(returncode=5))
    assert starter.main(["--model-bundle", "relative bundle"]) == 5
    assert calls[0][0] == [str(interpreter), "-B", str(Path(starter.__file__).resolve()),
                           "--model-bundle", "relative bundle"]
    assert calls[0][1] == {"check": False}
    assert calls[0][2] == tmp_path


def test_windows_entrypoint_quotes_local_interpreter_and_script():
    content = (ROOT / "Start-Demo.cmd").read_text(encoding="utf-8")
    assert '"%~dp0.venv\\Scripts\\python.exe" -B "%~dp0scripts\\start_demo.py" %*' in content
    assert "pip install" not in content
