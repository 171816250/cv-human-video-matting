"""Portable RVM bundles use recorded local bytes and never fetch missing models."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from cv_human_video_matting.matting.rvm import RVMRunner
from cv_human_video_matting.utils import model_bundle


@pytest.fixture
def cached_model(tmp_path):
    hub = tmp_path / "torch-hub"
    code = hub / "PeterL1n_RobustVideoMatting_master"
    for name in (
        "hubconf.py", "LICENSE", "model/__init__.py", "model/model.py", "model/mobilenetv3.py",
        "model/resnet.py", "model/lraspp.py", "model/decoder.py", "model/fast_guided_filter.py",
        "model/deep_guided_filter.py", "README.md",
    ):
        file = code / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"# local fixture: {name}\n", encoding="utf-8")
    for name in (".git/config", "model/__pycache__/model.cpython-39.pyc", "stray.pyc"):
        file = code / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"must not be exported")
    (hub / "checkpoints").mkdir()
    (hub / "checkpoints" / "rvm_mobilenetv3.pth").write_bytes(b"local mobilenet state fixture")
    (hub / "checkpoints" / "rvm_resnet50.pth").write_bytes(b"local resnet state fixture")
    return hub


@pytest.fixture
def bundle(cached_model, tmp_path):
    return model_bundle.prepare_model_bundle(tmp_path / "bundle", hub_dir=cached_model)


def test_bundle_copies_exact_model_and_license_and_excludes_cache_artifacts(bundle):
    info = model_bundle.verify_model_bundle(bundle, expected_model="mobilenetv3")
    assert info["schema_version"] == 1
    assert info["kind"] == "rvm_model_bundle"
    assert info["model_name"] == "mobilenetv3"
    assert Path(info["code_path"]) == bundle / "code"
    assert Path(info["weights_file"]) == bundle / "weights" / "rvm_mobilenetv3.pth"
    assert (bundle / "code" / "LICENSE").is_file()
    assert not (bundle / "code" / ".git").exists()
    assert not (bundle / "code" / "model" / "__pycache__").exists()
    assert not (bundle / "code" / "stray.pyc").exists()
    assert not (bundle / "weights" / "rvm_resnet50.pth").exists()
    assert info["upstream"]["requested_ref"] == "master"
    assert info["upstream"]["ref_is_immutable"] is False
    assert info["upstream"]["resolved_commit"] is None
    assert len(info["manifest_sha256"]) == len(info["content_sha256"]) == 64


def test_same_cache_export_has_same_content_identity(cached_model, tmp_path):
    one = model_bundle.prepare_model_bundle(tmp_path / "one", hub_dir=cached_model)
    two = model_bundle.prepare_model_bundle(tmp_path / "two", hub_dir=cached_model)
    first, second = model_bundle.verify_model_bundle(one), model_bundle.verify_model_bundle(two)
    assert first["content_sha256"] == second["content_sha256"]
    assert first["files"] == second["files"]


def test_resnet_bundle_uses_only_resnet_weights(cached_model, tmp_path):
    target = model_bundle.prepare_model_bundle(tmp_path / "resnet", "resnet50", cached_model)
    info = model_bundle.verify_model_bundle(target, expected_model="resnet50")
    assert Path(info["weights_file"]).name == "rvm_resnet50.pth"
    with pytest.raises(ValueError, match="model mismatch"):
        model_bundle.verify_model_bundle(target, expected_model="mobilenetv3")


@pytest.mark.parametrize("name", ["code/hubconf.py", "weights/rvm_mobilenetv3.pth"])
def test_changed_code_or_weights_fail_verification(bundle, name):
    file = bundle / name
    before = file.read_bytes()
    file.write_bytes(b"X" + before[1:])
    with pytest.raises(ValueError, match="hash/size mismatch"):
        model_bundle.verify_model_bundle(bundle)


@pytest.mark.parametrize("change", ["missing", "extra", "empty_weights"])
def test_file_inventory_and_empty_weights_are_not_marked_ready(bundle, change):
    if change == "missing":
        (bundle / "code" / "hubconf.py").unlink()
    elif change == "extra":
        (bundle / "code" / "unrecorded.py").write_text("# injected", encoding="utf-8")
    else:
        (bundle / "weights" / "rvm_mobilenetv3.pth").write_bytes(b"")
    with pytest.raises(ValueError):
        model_bundle.verify_model_bundle(bundle)


def test_generated_bytecode_cache_does_not_change_recorded_source_identity(bundle):
    previous = model_bundle.verify_model_bundle(bundle)["content_sha256"]
    cache = bundle / "code" / "model" / "__pycache__" / "model.cpython-39.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"generated cache fixture")
    assert model_bundle.verify_model_bundle(bundle)["content_sha256"] == previous


@pytest.mark.parametrize("bad_path", ["../outside.py", "/absolute.py", "C:/outside.py",
                                      "code/../escape.py", "code\\hubconf.py"])
def test_manifest_paths_cannot_escape_bundle(bundle, bad_path):
    file = bundle / "manifest.json"
    manifest = json.loads(file.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = bad_path
    file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="relative manifest path"):
        model_bundle.verify_model_bundle(bundle)


def test_manifest_must_match_its_recorded_content_hash(bundle):
    file = bundle / "manifest.json"
    manifest = json.loads(file.read_text(encoding="utf-8"))
    manifest["content_sha256"] = "0" * 64
    file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        model_bundle.verify_model_bundle(bundle)


def test_duplicate_case_variant_manifest_paths_are_rejected(bundle):
    file = bundle / "manifest.json"
    manifest = json.loads(file.read_text(encoding="utf-8"))
    duplicate = dict(manifest["files"][0])
    duplicate["path"] = duplicate["path"].upper()
    manifest["files"].append(duplicate)
    file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        model_bundle.verify_model_bundle(bundle)


def test_missing_bundle_and_invalid_manifest_have_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory"):
        model_bundle.verify_model_bundle(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="manifest"):
        model_bundle.verify_model_bundle(tmp_path)
    (tmp_path / "manifest.json").write_text("{bad JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        model_bundle.verify_model_bundle(tmp_path)


@pytest.mark.parametrize("missing", ["code", "weight", "required_module"])
def test_prepare_fails_locally_when_cache_is_missing(cached_model, tmp_path, missing):
    if missing == "code":
        hub = tmp_path / "no-cache"
    else:
        hub = cached_model
        target = (hub / "checkpoints" / "rvm_mobilenetv3.pth" if missing == "weight"
                  else hub / model_bundle.CACHE_CODE_DIR / "model" / "decoder.py")
        target.unlink()
    output = tmp_path / "not-created"
    with pytest.raises((FileNotFoundError, ValueError)):
        model_bundle.prepare_model_bundle(output, hub_dir=hub)
    assert not output.exists()


def test_prepare_does_not_overwrite_existing_directory(cached_model, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    keep = target / "keep.txt"
    keep.write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        model_bundle.prepare_model_bundle(target, hub_dir=cached_model)
    assert keep.read_bytes() == b"user file"


def test_prepare_refuses_output_inside_original_cache(cached_model):
    with pytest.raises(ValueError, match="outside"):
        model_bundle.prepare_model_bundle(cached_model / "export", hub_dir=cached_model)


def test_copy_failure_cleans_staging_without_publishing(cached_model, tmp_path, monkeypatch):
    def broken_copy(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(model_bundle.shutil, "copyfile", broken_copy)
    target = tmp_path / "failed-bundle"
    with pytest.raises(OSError, match="disk full"):
        model_bundle.prepare_model_bundle(target, hub_dir=cached_model)
    assert not target.exists()
    assert list(tmp_path.glob(".failed-bundle-*")) == []


def test_source_change_during_copy_is_rejected(cached_model, tmp_path, monkeypatch):
    original = model_bundle.shutil.copyfile

    def changing_copy(source, destination):
        if Path(source).name == "hubconf.py":
            Path(source).write_text("# changed during copy", encoding="utf-8")
        return original(source, destination)

    monkeypatch.setattr(model_bundle.shutil, "copyfile", changing_copy)
    output = tmp_path / "changed-bundle"
    with pytest.raises(ValueError, match="changed while being copied"):
        model_bundle.prepare_model_bundle(output, hub_dir=cached_model)
    assert not output.exists()


class FakeModel:
    def __init__(self):
        self.state = None

    def load_state_dict(self, state):
        self.state = state

    def to(self, device):
        return self

    def eval(self):
        return self


def test_runner_verifies_bundle_and_loads_only_local_code_and_weights(bundle, monkeypatch):
    calls, weights = [], []
    model = FakeModel()
    previous_model_module = SimpleNamespace(identity="previous cached model")
    monkeypatch.setitem(sys.modules, "model", previous_model_module)

    def load(path, name, **kwargs):
        assert "model" not in sys.modules
        calls.append((path, name, kwargs))
        sys.modules["model"] = SimpleNamespace(identity="bundled model")
        return model

    def load_weights(path, **kwargs):
        weights.append((path, kwargs))
        return {"fixture": "local state"}

    monkeypatch.setattr(torch.hub, "load", load)
    monkeypatch.setattr(torch, "load", load_weights)
    runner = RVMRunner(bundle_path=bundle, device="cpu")
    assert calls == [(str(bundle / "code"), "mobilenetv3", {"source": "local", "pretrained": False})]
    assert Path(weights[0][0]) == bundle / "weights" / "rvm_mobilenetv3.pth"
    assert weights[0][1]["weights_only"] is True
    assert str(weights[0][1]["map_location"]) == "cpu"
    assert model.state == {"fixture": "local state"}
    assert runner.bundle_info["content_sha256"]
    assert sys.modules["model"] is previous_model_module


def test_failed_local_model_import_restores_previous_module(bundle, monkeypatch):
    previous = SimpleNamespace(identity="previous")
    monkeypatch.setitem(sys.modules, "model", previous)

    def failing_import(*args, **kwargs):
        sys.modules["model.bad"] = SimpleNamespace()
        raise RuntimeError("local dependency missing")

    monkeypatch.setattr(torch.hub, "load", failing_import)
    with pytest.raises(RuntimeError, match="local dependency missing"):
        RVMRunner(bundle_path=bundle)
    assert sys.modules["model"] is previous
    assert "model.bad" not in sys.modules


def test_bundle_checkpoint_conflict_fails_before_loading_models(bundle):
    with pytest.raises(ValueError, match="cannot be configured together"):
        RVMRunner(bundle_path=bundle, checkpoint="missing.pth")


def test_runner_rejects_changed_bundle_before_any_model_code_is_executed(bundle):
    (bundle / "code" / "hubconf.py").write_text("# modified", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        RVMRunner(bundle_path=bundle)


def test_preparation_script_works_without_torch_when_cache_path_is_given(cached_model, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", "scripts/prepare_offline_model.py", "--hub-dir", str(cached_model),
         "--output", str(tmp_path / "script-bundle")], cwd=root,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Content SHA256:" in result.stdout
    assert model_bundle.verify_model_bundle(tmp_path / "script-bundle")["model_name"] == "mobilenetv3"
