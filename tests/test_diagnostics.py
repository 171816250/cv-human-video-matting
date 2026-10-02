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
                            vision_requires=["torch==2.5.1"],
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
            result = state.versions[name]
            if isinstance(result, Exception):
                raise result
            return result
        return {
            "torch": "2.5.1", "torchvision": "0.20.1", "numpy": "1.26.4",
            "opencv-python": "4.10.0.84", "pillow": "10.4.0", "pyyaml": "6.0.2",
            "gradio": "4.37.2", "imageio-ffmpeg": "0.6.0", "transformers": "4.41.2",
            "timm": "1.0.27", "safetensors": "0.4.3", "tqdm": "4.66.5", "pytest": "8.3.3",
            "huggingface-hub": "0.23.2", "fastapi": "0.111.1", "starlette": "0.37.2",
            "pydantic": "2.8.2", "gradio-client": "1.0.2", "packaging": "25.0",
        }.get(name.lower(), "1.2.3")

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
    monkeypatch.setattr(diagnostics, "distribution_requires", lambda package: state.vision_requires)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    monkeypatch.setattr(diagnostics, "_probe_ffv1", lambda cv2, np: "FFV1 round trip passed")
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
    assert not report["ready"]
    assert checks_by_name(report)["torch"]["detail"] == "version unavailable"
    assert checks_by_name(report)["requirement:torch"]["status"] == "fail"


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
    assert checks["optional-requirements"]["status"] == "pass"
    assert checks["depth-model"]["status"] == "warn"


def test_base_rvm_does_not_require_optional_models_or_test_tools(environment):
    environment.config = {"matting": {"model": "rvm_mobilenetv3"}}
    optional = ("transformers", "timm", "safetensors", "pytest", "pytest-cov")
    for name in optional:
        environment.versions[name] = diagnostics.PackageNotFoundError(name)
        environment.errors[name] = ImportError(name)
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert report["ready"]
    assert report["optional_requirements_profile"] is None
    assert report["checked_optional_packages"] == []
    assert "optional-requirements" not in checks
    assert all(name not in checks and f"requirement:{name}" not in checks for name in optional)


@pytest.mark.parametrize("package", ["timm", "safetensors"])
def test_enabled_depth_checks_optional_versions_without_transformers(environment, package):
    environment.config = {"matting": {"model": "rvm_mobilenetv3"}, "depth": {"enabled": True}}
    environment.versions[package] = "0.1.0"
    environment.versions["transformers"] = diagnostics.PackageNotFoundError("transformers")
    environment.errors["transformers"] = ImportError("transformers")
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks[f"requirement:{package}"]["status"] == "fail"
    assert checks["optional-requirements"]["status"] == "fail"
    assert "transformers" not in checks
    assert "requirement:transformers" not in checks
    assert report["checked_optional_packages"] == ["safetensors", "timm"]
    assert Path(report["optional_requirements_profile"]).name == "requirements-optional.txt"


def test_depth_anything_doctor_checks_local_files_and_relevant_dependencies(environment, tmp_path):
    checkpoint = tmp_path / "depth-anything"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type":"depth_anything","depth_estimation_type":"relative",'
        '"backbone":null,"backbone_config":{"model_type":"dinov2",'
        '"hidden_size":384,"patch_size":14}}',
        encoding="utf-8",
    )
    (checkpoint / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"local-test-weights")
    environment.config = {
        "matting": {"model": "dummy"},
        "depth": {"enabled": True, "model": "DepthAnythingV2_small",
                  "checkpoint": str(checkpoint)},
    }
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert checks["depth-checkpoint"]["status"] == "pass"
    assert checks["depth-model"]["status"] == "warn"
    assert report["checked_optional_packages"] == ["safetensors", "transformers"]
    assert "timm" not in checks
    assert report["ready"]

    (checkpoint / "model.safetensors").unlink()
    checks = checks_by_name(diagnostics.check_environment())
    assert checks["depth-checkpoint"]["status"] == "fail"


def test_enabled_birefnet_checks_transformers_version_even_when_import_works(environment):
    environment.config = {"matting": {"model": "birefnet"}}
    environment.versions["transformers"] = "4.33.1"
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks["transformers"]["status"] == "pass"
    assert checks["requirement:transformers"]["status"] == "fail"
    assert checks["optional-requirements"]["status"] == "fail"


def test_optional_profile_remains_checked_with_explicit_base_profile(environment, tmp_path):
    profile = tmp_path / "base.txt"
    profile.write_text("numpy>=1.24\n", encoding="utf-8")
    environment.config = {"matting": {"model": "birefnet"}}
    environment.versions["timm"] = diagnostics.PackageNotFoundError("timm")
    checks = checks_by_name(diagnostics.check_environment(requirements_path=profile))
    assert checks["requirements"]["status"] == "pass"
    assert checks["optional-requirements"]["status"] == "fail"


@pytest.mark.parametrize("torch_version,vision_version", [
    ("2.5.1", "0.20.1"), ("2.5.1+cpu", "0.20.1+cpu"),
    ("2.5.1+cu121", "0.20.1+cu121"),
])
def test_declared_torchvision_pair_accepts_matching_builds(environment, torch_version, vision_version):
    environment.versions.update(torch=torch_version, torchvision=vision_version)
    checks = checks_by_name(diagnostics.check_environment())
    assert checks["torchvision-pair"]["status"] == "pass"
    assert checks["requirement:torch"]["status"] == "pass"
    assert checks["requirement:torchvision"]["status"] == "pass"


@pytest.mark.parametrize("requires,torch_version,vision_version,detail", [
    (["torch==2.4.1"], "2.5.1", "0.20.1", "declares"),
    (["torch==2.5.1"], "2.5.1+cpu", "0.20.1+cu121", "Build mismatch"),
    ([], "2.5.1", "0.20.1", "does not declare"),
])
def test_importable_but_inconsistent_torchvision_pair_fails(
    environment, requires, torch_version, vision_version, detail,
):
    environment.vision_requires = requires
    environment.versions.update(torch=torch_version, torchvision=vision_version)
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks["torch"]["status"] == "pass"
    assert checks["torchvision"]["status"] == "pass"
    assert checks["torchvision-pair"]["status"] == "fail"
    assert detail in checks["torchvision-pair"]["detail"]


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


@pytest.mark.parametrize("package,installed", [
    ("numpy", "1.22.4"), ("opencv-python", "4.7.0.72"), ("pillow", "9.5.0"),
    ("torch", "1.13.0+cu117"), ("gradio", "5.0.0"), ("numpy", "2.0.0"),
])
def test_importable_but_out_of_range_dependencies_make_doctor_fail(environment, package, installed):
    environment.versions[package] = installed
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks["requirement:" + package]["status"] == "fail"
    assert installed in checks["requirement:" + package]["detail"]
    assert checks["requirements"]["status"] == "fail"
    assert checks["video-export"]["status"] == "pass"


def test_requirements_include_constraints_are_combined_and_profile_is_recorded(environment, tmp_path):
    base = tmp_path / "base.txt"
    base.write_text("numpy>=1.24\n", encoding="utf-8")
    profile = tmp_path / "demo.txt"
    profile.write_text("-r base.txt\nnumpy<2\n", encoding="utf-8")
    environment.versions["numpy"] = "1.23.5"
    report = diagnostics.check_environment(requirements_path=profile)
    checks = checks_by_name(report)
    assert report["requirements_profile"] == str(profile.resolve())
    assert checks["requirement:numpy"]["status"] == "fail"
    assert "base.txt" in checks["requirement:numpy"]["detail"]
    assert "demo.txt" in checks["requirement:numpy"]["detail"]


def test_unavailable_requirements_file_does_not_claim_compatibility(environment, tmp_path):
    report = diagnostics.check_environment(requirements_path=tmp_path / "missing.txt")
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks["requirements"]["status"] == "fail"
    assert "missing.txt" in checks["requirements"]["detail"]


def test_cyclic_requirements_includes_are_rejected(environment, tmp_path):
    profile = tmp_path / "cycle.txt"
    profile.write_text("-r cycle.txt\n", encoding="utf-8")
    checks = checks_by_name(diagnostics.check_environment(requirements_path=profile))
    assert checks["requirements"]["status"] == "fail"
    assert "Cyclic" in checks["requirements"]["detail"]


def test_bundle_is_verified_even_without_torch_and_does_not_query_hub_cache(environment, monkeypatch):
    from cv_human_video_matting.utils import model_bundle
    environment.config = {"matting": {"model": "rvm_mobilenetv3", "bundle": "offline/rvm"},
                          "depth": {"enabled": True}}
    environment.errors["torch"] = ImportError("torch not installed")
    calls = []

    def verify(path, expected_model):
        calls.append((path, expected_model))
        return {"content_sha256": "a" * 64}

    monkeypatch.setattr(model_bundle, "verify_model_bundle", verify)
    checks = checks_by_name(diagnostics.check_environment())
    assert calls == [("offline/rvm", "mobilenetv3")]
    assert checks["model-bundle"]["status"] == "pass"
    assert "model-cache" not in checks
    assert "MiDaS" in checks["depth-model"]["detail"]
    assert checks["torch"]["status"] == "fail"


def test_valid_bundle_skips_broken_global_torch_hub_cache(environment, monkeypatch):
    from cv_human_video_matting.utils import model_bundle
    environment.config = {"matting": {"model": "rvm_resnet50", "bundle": "offline/resnet"}}

    def no_global_cache():
        pytest.fail("Bundle mode must not read the global torch hub cache.")

    environment.torch.hub.get_dir = no_global_cache
    monkeypatch.setattr(model_bundle, "verify_model_bundle",
                        lambda path, expected_model: {"content_sha256": "b" * 64})
    report = diagnostics.check_environment()
    assert report["ready"]
    assert checks_by_name(report)["model-bundle"]["status"] == "pass"
    assert "model-cache" not in checks_by_name(report)


def test_changed_bundle_is_a_failure_without_cache_fallback(environment, monkeypatch):
    from cv_human_video_matting.utils import model_bundle
    environment.config = {"matting": {"model": "rvm_mobilenetv3", "bundle": "changed"}}

    def changed(*args, **kwargs):
        raise ValueError("RVM bundle file hash/size mismatch: code/hubconf.py")

    monkeypatch.setattr(model_bundle, "verify_model_bundle", changed)
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks["model-bundle"]["status"] == "fail"
    assert "hash/size mismatch" in checks["model-bundle"]["detail"]
    assert "model-cache" not in checks


def test_ffv1_error_is_recorded_even_when_export_encoder_exists(environment, monkeypatch):
    def broken(cv2, np):
        raise RuntimeError("FFV1 writer unavailable")

    monkeypatch.setattr(diagnostics, "_probe_ffv1", broken)
    report = diagnostics.check_environment()
    checks = checks_by_name(report)
    assert not report["ready"]
    assert checks["lossless-intermediate"]["status"] == "fail"
    assert checks["video-export"]["status"] == "pass"


def test_actual_ffv1_matroska_round_trip():
    import cv2
    import numpy as np
    assert "Matroska" in diagnostics._probe_ffv1(cv2, np)


def test_ffv1_unavailable_writer_releases_resource():
    import numpy as np
    released = []
    cv2 = SimpleNamespace(
        VideoWriter_fourcc=lambda *args: 1,
        VideoWriter=lambda *args: SimpleNamespace(isOpened=lambda: False,
                                                  release=lambda: released.append(True)),
    )
    with pytest.raises(RuntimeError, match="FFV1/Matroska writer"):
        diagnostics._probe_ffv1(cv2, np)
    assert released == [True]
