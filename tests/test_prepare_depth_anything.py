"""Offline checks for the pinned Depth Anything V2 Small preparation command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts import prepare_depth_anything as prepare


FAKE_WEIGHTS = b"test-only depth weights, never a real model"


@pytest.fixture
def fake_snapshot(monkeypatch):
    monkeypatch.setattr(prepare, "WEIGHTS_SHA256", hashlib.sha256(FAKE_WEIGHTS).hexdigest())
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        (destination / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "depth_anything",
                    "depth_estimation_type": "relative",
                    "use_pretrained_backbone": False,
                    "backbone": None,
                    "use_timm_backbone": False,
                    "backbone_kwargs": None,
                    "backbone_config": {
                        "model_type": "dinov2",
                        "hidden_size": 384,
                        "patch_size": 14,
                    },
                }
            ),
            encoding="utf-8",
        )
        (destination / "preprocessor_config.json").write_text(
            json.dumps({"image_processor_type": "DPTImageProcessor"}), encoding="utf-8"
        )
        (destination / "model.safetensors").write_bytes(FAKE_WEIGHTS)
        # Hugging Face may create local metadata. It must stay out of the bundle.
        (destination / ".cache").mkdir()
        (destination / ".cache" / "metadata").write_text("cache", encoding="utf-8")
        return str(destination)

    return download, calls


def test_prepare_pins_official_source_and_records_reproducible_hashes(tmp_path, fake_snapshot):
    download, calls = fake_snapshot
    first = tmp_path / "weights" / "first"
    second = tmp_path / "weights" / "second"
    manifest = prepare.prepare_depth_anything(first, snapshot_download_fn=download)
    other = prepare.prepare_depth_anything(second, snapshot_download_fn=download)

    assert len(calls) == 2
    for call in calls:
        assert call["repo_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
        assert call["revision"] == "5426e4f0f36572d16453bbda7a8389317b1bef99"
        assert call["repo_type"] == "model"
        assert call["allow_patterns"] == list(prepare.REQUIRED_FILES)
        assert Path(call["cache_dir"]).parent == Path(call["local_dir"])
        assert call["token"] is False
    assert manifest == other == prepare.verify_depth_anything(first)
    assert manifest["source"]["repo_id"] == prepare.REPO_ID
    assert manifest["source"]["revision"] == prepare.REVISION
    assert manifest["license"]["id"] == "apache-2.0"
    assert manifest["files"]["model.safetensors"]["sha256"] == prepare.WEIGHTS_SHA256
    assert set(manifest["files"]) == set(prepare.REQUIRED_FILES)
    for name in prepare.REQUIRED_FILES:
        data = (first / name).read_bytes()
        assert manifest["files"][name] == {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    assert {path.name for path in first.iterdir()} == {*prepare.REQUIRED_FILES, "manifest.json"}
    assert not any(path.name.startswith(".depth-anything-") for path in first.parent.iterdir())
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


@pytest.mark.parametrize("name", ["config.json", "preprocessor_config.json", "model.safetensors"])
def test_verify_rejects_changed_file_hash(tmp_path, fake_snapshot, name):
    download, _ = fake_snapshot
    output = tmp_path / "weights" / "model"
    prepare.prepare_depth_anything(output, snapshot_download_fn=download)
    file = output / name
    file.write_bytes(file.read_bytes() + (b"\n" if name.endswith(".json") else b"tamper"))
    with pytest.raises(ValueError, match="SHA256|manifest"):
        prepare.verify_depth_anything(output)


@pytest.mark.parametrize(
    "change",
    ["model_type", "architecture", "remote_backbone", "timm_backbone", "backbone_kwargs",
     "processor", "missing", "lfs"],
)
def test_invalid_download_never_becomes_the_final_bundle(tmp_path, fake_snapshot, change):
    download, _ = fake_snapshot
    output = tmp_path / "weights" / "model"

    def invalid_download(**kwargs):
        download(**kwargs)
        base = Path(kwargs["local_dir"])
        if change == "model_type":
            config = json.loads((base / "config.json").read_text(encoding="utf-8"))
            config["model_type"] = "other"
            (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
        elif change == "architecture":
            config = json.loads((base / "config.json").read_text(encoding="utf-8"))
            config["backbone_config"]["hidden_size"] = 768
            (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
        elif change in {"remote_backbone", "timm_backbone", "backbone_kwargs"}:
            config = json.loads((base / "config.json").read_text(encoding="utf-8"))
            updates = {
                "remote_backbone": ("backbone", "facebook/dinov2-small"),
                "timm_backbone": ("use_timm_backbone", True),
                "backbone_kwargs": ("backbone_kwargs", {"out_indices": [1, 2]}),
            }
            key, value = updates[change]
            config[key] = value
            (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
        elif change == "processor":
            (base / "preprocessor_config.json").write_text(
                '{"image_processor_type": "WrongImageProcessor"}', encoding="utf-8"
            )
        elif change == "missing":
            (base / "preprocessor_config.json").unlink()
        else:
            (base / "model.safetensors").write_bytes(
                b"version https://git-lfs.github.com/spec/v1\n"
            )

    with pytest.raises((FileNotFoundError, ValueError)):
        prepare.prepare_depth_anything(output, snapshot_download_fn=invalid_download)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_existing_invalid_directory_is_preserved_without_download(tmp_path, fake_snapshot):
    download, _ = fake_snapshot
    output = tmp_path / "weights" / "model"
    prepare.prepare_depth_anything(output, snapshot_download_fn=download)
    (output / "manifest.json").unlink()
    marker = output / "user-note.txt"
    marker.write_text("keep", encoding="utf-8")

    def unexpected_download(**_kwargs):
        pytest.fail("existing directory must never trigger a download")

    with pytest.raises(FileNotFoundError, match="manifest"):
        prepare.prepare_depth_anything(output, snapshot_download_fn=unexpected_download)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert (output / "model.safetensors").read_bytes() == FAKE_WEIGHTS


def test_existing_valid_bundle_is_reused_without_download(tmp_path, fake_snapshot):
    download, calls = fake_snapshot
    output = tmp_path / "weights" / "model"
    manifest = prepare.prepare_depth_anything(output, snapshot_download_fn=download)
    assert prepare.prepare_depth_anything(output, snapshot_download_fn=download) == manifest
    assert len(calls) == 1


def test_verify_rejects_changed_source_in_manifest(tmp_path, fake_snapshot):
    download, _ = fake_snapshot
    output = tmp_path / "weights" / "model"
    prepare.prepare_depth_anything(output, snapshot_download_fn=download)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source"]["revision"] = "main"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        prepare.verify_depth_anything(output)


def test_verify_only_uses_no_hub_import_or_network(tmp_path, fake_snapshot, monkeypatch, capsys):
    download, _ = fake_snapshot
    root = tmp_path / "weights"
    output = root / "model"
    prepare.prepare_depth_anything(output, snapshot_download_fn=download)
    monkeypatch.setattr(prepare, "WEIGHTS_ROOT", root)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    prepare.main(["--output", str(output), "--verify-only"])
    assert "Depth Anything V2 Small ready" in capsys.readouterr().out


def test_cli_rejects_destination_outside_ignored_weights(tmp_path, monkeypatch):
    root = tmp_path / "weights"
    monkeypatch.setattr(prepare, "WEIGHTS_ROOT", root)
    outside = tmp_path / "tracked" / "model"
    with pytest.raises(SystemExit) as exc:
        prepare.main(["--output", str(outside)])
    assert exc.value.code == 2
    assert not outside.exists()
    assert not root.exists()


def test_cli_accepts_new_relative_child_under_weights(monkeypatch):
    monkeypatch.chdir(prepare.PROJECT_ROOT)
    assert prepare._cli_output("weights/new-depth-model") == (
        prepare.WEIGHTS_ROOT / "new-depth-model"
    )


def test_optional_hub_import_has_clear_error_and_leaves_no_bundle(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    output = tmp_path / "weights" / "model"
    with pytest.raises(RuntimeError, match="optional huggingface_hub"):
        prepare.prepare_depth_anything(output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []
