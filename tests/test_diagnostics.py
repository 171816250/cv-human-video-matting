"""Local preflight reports remain useful when dependencies or export tools fail."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cv_human_video_matting import diagnostics
from cv_human_video_matting.utils import config as config_module


@pytest.fixture
def environment(monkeypatch, tmp_path):
    state = SimpleNamespace(config={"matting": {"model": "dummy"}}, errors={}, versions={},
                            ffmpeg_error=None,
                            encoders=" V....D libx264 H.264\n A..... aac AAC\n")

    def never_load(*args, **kwargs):
        pytest.fail("Doctor must never load or download models.")

    state.torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0,
                             get_device_name=lambda index: "Test GPU"),
        from_numpy=lambda array: SimpleNamespace(numpy=lambda: array),
        hub=SimpleNamespace(get_dir=lambda: str(tmp_path / "hub"), load=never_load),
    )
    state.modules = {
        "torch": state.torch,
        "numpy": SimpleNamespace(zeros=lambda shape, dtype: [[0, 0], [0, 0]]),
        "imageio_ffmpeg": SimpleNamespace(get_ffmpeg_exe=lambda: "local-ffmpeg"),
    }

    def import_module(name):
        if name in state.errors:
            raise state.errors[name]
        return state.modules.get(name, SimpleNamespace())

    def version(name):
        if name in state.versions:
            raise state.versions[name]
        return "1.2.3"

    def load_config(path):
        if isinstance(state.config, Exception):
            raise state.config
        return state.config

    def run(command, **kwargs):
        assert command == ["local-ffmpeg", "-nostdin", "-hide_banner", "-encoders"]
        assert kwargs["timeout"] == 20
        if state.ffmpeg_error:
            raise state.ffmpeg_error
        return SimpleNamespace(stdout=state.encoders)

    monkeypatch.setattr(diagnostics.importlib, "import_module", import_module)
    monkeypatch.setattr(diagnostics, "version", version)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    monkeypatch.setattr(config_module, "load_config", load_config)
    monkeypatch.setattr(diagnostics.sys, "version_info", (3, 10, 0))
    return state


def checks_by_name(report):
    return {item["name"]: item for item in report["checks"]}


def test_cpu_preflight_is_json_serializable_without_model_loading(environment):
    report = diagnostics.check_environment()
    assert report["ready"]
    assert report["schema_version"] == 1
    checks = checks_by_name(report)
    assert "CPU selected" in checks["device"]["detail"]
    assert checks["video-export"]["status"] == "pass"
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("module,error", [
    ("torch", ImportError("missing torch")),
    ("torchvision", RuntimeError("operator torchvision::nms does not exist")),
    ("cv2", OSError("DLL load failed")),
    ("yaml", ImportError("missing yaml")),
])
def test_broken_dependencies_are_reported_and_other_checks_continue(environment, module, error):
    environment.errors[module] = error
    report = diagnostics.check_environment()
    assert not report["ready"]
    assert any(item["status"] == "fail" and str(error) in item["detail"] for item in report["checks"])
    assert checks_by_name(report)["video-export"]["status"] == "pass"


def test_package_metadata_missing_does_not_prevent_import_check(environment):
    environment.versions["torch"] = diagnostics.PackageNotFoundError("torch")
    report = diagnostics.check_environment()
    assert report["ready"]
    assert checks_by_name(report)["torch"]["detail"] == "version unavailable"


@pytest.mark.parametrize("error", [FileNotFoundError("missing config"), ValueError("bad interval")])
def test_bad_config_is_recorded_without_suppressing_environment_checks(environment, error):
    environment.config = error
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks["config"]["status"] == "fail"
    assert checks["numpy-torch"]["status"] == "pass"
    assert checks["video-export"]["status"] == "pass"
    assert "model-cache" not in checks


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:2"])
def test_explicit_cuda_request_fails_when_unavailable(environment, device):
    environment.config["device"] = {"mode": device}
    report = diagnostics.check_environment()
    assert not report["ready"]
    assert checks_by_name(report)["device"]["status"] == "fail"


def test_cpu_choice_remains_cpu_even_if_cuda_is_available(environment):
    environment.config["device"] = {"mode": "cpu"}
    environment.torch.cuda.is_available = lambda: True
    report = diagnostics.check_environment()
    assert "CPU selected" in checks_by_name(report)["device"]["detail"]


def test_invalid_cuda_index_is_reported(environment):
    environment.config["device"] = {"mode": "cuda:2"}
    environment.torch.cuda.is_available = lambda: True
    environment.torch.cuda.device_count = lambda: 1
    checks = checks_by_name(diagnostics.check_environment())
    assert checks["device"]["status"] == "fail"
    assert "only 1" in checks["device"]["detail"]


def test_cuda_runtime_error_does_not_skip_abi_check(environment):
    def broken_cuda():
        raise RuntimeError("CUDA driver error")

    environment.torch.cuda.is_available = broken_cuda
    checks = checks_by_name(diagnostics.check_environment())
    assert checks["device"]["status"] == "fail"
    assert checks["numpy-torch"]["status"] == "pass"
    assert checks["video-export"]["status"] == "pass"


def test_numpy_torch_abi_failure_and_absent_cache_are_both_reported(environment):
    def broken_conversion(array):
        raise RuntimeError("Numpy is not available")

    environment.torch.from_numpy = broken_conversion
    environment.config["matting"]["model"] = "rvm_mobilenetv3"
    checks = checks_by_name(diagnostics.check_environment())
    assert checks["numpy-torch"]["status"] == "fail"
    assert checks["model-cache"]["status"] == "warn"


def test_checkpoint_is_checked_even_when_torch_import_fails(environment, tmp_path):
    environment.errors["torch"] = ImportError("missing")
    environment.config["matting"]["checkpoint"] = str(tmp_path / "missing.pth")
    checks = checks_by_name(diagnostics.check_environment())
    assert checks["checkpoint"]["status"] == "fail"


def test_rvm_cache_presence_does_not_claim_inference_success(environment, tmp_path):
    environment.config["matting"]["model"] = "rvm_mobilenetv3"
    hub = tmp_path / "hub"
    code = hub / "PeterL1n_RobustVideoMatting_master" / "hubconf.py"
    weight = hub / "checkpoints" / "rvm_mobilenetv3.pth"
    code.parent.mkdir(parents=True)
    weight.parent.mkdir(parents=True)
    code.write_text("# local cache fixture", encoding="utf-8")
    weight.write_bytes(b"cache fixture")
    check = checks_by_name(diagnostics.check_environment())["model-cache"]
    assert check["status"] == "pass"
    assert "not an inference test" in check["detail"]
    weight.write_bytes(b"")
    assert checks_by_name(diagnostics.check_environment())["model-cache"]["status"] == "warn"


def test_optional_backend_dependencies_are_checked(environment):
    environment.config = {"matting": {"model": "birefnet"}, "depth": {"enabled": True}}
    environment.errors["timm"] = ImportError("missing timm")
    checks = checks_by_name(diagnostics.check_environment())
    assert checks["timm"]["status"] == "fail"
    assert checks["transformers"]["status"] == "pass"
    assert checks["depth-model"]["status"] == "warn"


@pytest.mark.parametrize("error", [
    subprocess.TimeoutExpired("ffmpeg", 20),
    subprocess.CalledProcessError(1, "ffmpeg"),
    FileNotFoundError("missing ffmpeg"),
])
def test_ffmpeg_failures_return_actionable_failed_check(environment, error):
    environment.ffmpeg_error = error
    report = diagnostics.check_environment()
    assert not report["ready"]
    assert checks_by_name(report)["video-export"]["status"] == "fail"


def test_encoder_description_does_not_count_as_missing_encoder(environment):
    environment.encoders = " V....D other description mentions libx264\n A..... aac AAC\n"
    check = checks_by_name(diagnostics.check_environment())["video-export"]
    assert check["status"] == "fail"
    assert "libx264" in check["detail"]


def test_unsupported_python_is_a_failure(environment, monkeypatch):
    monkeypatch.setattr(diagnostics.sys, "version_info", (3, 12, 0))
    assert checks_by_name(diagnostics.check_environment())["python"]["status"] == "fail"


def test_save_report_preserves_unicode_and_replaces_atomically(tmp_path):
    target = tmp_path / "reports" / "环境检查.json"
    report = {"ready": True, "message": "环境检查完成"}
    assert diagnostics.save_diagnostics(report, target) == target
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert list(target.parent.glob("*.tmp")) == []


def test_serialization_failure_keeps_previous_report(tmp_path):
    target = tmp_path / "doctor.json"
    target.write_text("previous report", encoding="utf-8")
    with pytest.raises(ValueError):
        diagnostics.save_diagnostics({"value": float("nan")}, target)
    assert target.read_text(encoding="utf-8") == "previous report"


def test_report_requires_json_extension(tmp_path):
    with pytest.raises(ValueError, match=".json"):
        diagnostics.save_diagnostics({}, tmp_path / "config.yaml")


def test_cli_help_and_doctor_work_without_site_packages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    help_result = subprocess.run([sys.executable, "-S", "app.py", "--help"], cwd=root,
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0, help_result.stderr
    assert "doctor" in help_result.stdout
    target = tmp_path / "no-dependencies.json"
    doctor = subprocess.run(
        [sys.executable, "-S", "app.py", "doctor", "--report", str(target)], cwd=root,
        capture_output=True, text=True, timeout=30,
    )
    assert doctor.returncode == 1, doctor.stderr
    report = json.loads(target.read_text(encoding="utf-8"))
    assert not report["ready"]
    checks = checks_by_name(report)
    assert checks["torch"]["status"] == "fail"
    assert checks["config"]["status"] == "fail"
