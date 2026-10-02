"""Migration archives must survive a new host without overwriting user data."""

import hashlib
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "server_portable", Path(__file__).resolve().parents[1] / "scripts/server/portable.py",
)
portable = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portable)


def test_deployment_archive_round_trip_and_tamper_detection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    app = source / "app.py"
    app.write_text("print('ready')\n", encoding="utf-8")
    model = source / "model.pth"
    model.write_bytes(b"model bytes")
    archive = tmp_path / "deployment.zip"
    files = {"app.py": app, "weights/portable-rvm/weights/model.pth": model}
    portable.create_archive(archive, files, "deployment", model_content_sha256="a" * 64)
    assert portable.verify_archive(archive, "deployment")["model_content_sha256"] == "a" * 64

    restored = tmp_path / "new server" / "source"
    portable.extract_archive(archive, restored)
    assert (restored / "app.py").read_bytes() == app.read_bytes()
    assert (restored / "weights/portable-rvm/weights/model.pth").read_bytes() == model.read_bytes()
    with pytest.raises(portable.PortableError, match="already exists"):
        portable.extract_archive(archive, restored)

    with ZipFile(archive, "a") as handle:
        handle.writestr("../escape", "bad")
    with pytest.raises(portable.PortableError, match="SHA256"):
        portable.verify_archive(archive)
    portable._sidecar(archive).write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n", encoding="ascii",
    )
    with pytest.raises(portable.PortableError, match="Unsafe archive path"):
        portable.verify_archive(archive)
    assert not (tmp_path / "escape").exists()


def test_backup_restores_user_files_and_rejects_conflicts(tmp_path):
    old = tmp_path / "old" / "shared"
    (old / "data/input").mkdir(parents=True)
    (old / "data/input/person.mp4").write_bytes(b"authorized source")
    (old / "data/output").mkdir(parents=True)
    (old / "data/output/result.json").write_text('{"passed": true}', encoding="utf-8")
    (old / "logs").mkdir()
    (old / "logs/server.log").write_text("started", encoding="utf-8")
    (old / "weights/offline-rvm/weights").mkdir(parents=True)
    (old / "weights/offline-rvm/weights/rvm.pth").write_bytes(b"checkpoint")
    backup = tmp_path / "migration.zip"
    portable.create_archive(backup, portable._backup_files(old), "data_backup")

    new = tmp_path / "new" / "shared"
    (new / "weights/offline-rvm/weights").mkdir(parents=True)
    # A fresh deployment may already have the same model bytes under this name.
    (new / "weights/offline-rvm/weights/rvm.pth").write_bytes(b"checkpoint")
    portable.restore_backup(backup, new)
    assert (new / "data/input/person.mp4").read_bytes() == b"authorized source"
    assert (new / "data/output/result.json").read_text(encoding="utf-8") == '{"passed": true}'
    assert (new / "weights/offline-rvm/weights/rvm.pth").read_bytes() == b"checkpoint"
    portable.restore_backup(backup, new)  # An identical retry is safe.
    (new / "data/output/result.json").write_text("newer result", encoding="utf-8")
    with pytest.raises(portable.PortableError, match="overwrite different data"):
        portable.restore_backup(backup, new)
    assert (new / "data/output/result.json").read_text(encoding="utf-8") == "newer result"


def test_empty_backup_is_a_valid_checkpoint(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    backup = tmp_path / "empty.zip"
    portable.create_archive(backup, portable._backup_files(shared), "data_backup")
    assert portable.verify_archive(backup, "data_backup")["files"] == []


def test_backup_skips_only_known_current_model_link(tmp_path):
    shared = tmp_path / "shared"
    model = shared / "weights/offline-rvm"
    model.mkdir(parents=True)
    (model / "weights.pth").write_bytes(b"checkpoint")
    pointer = shared / "weights/current-rvm"
    other = shared / "weights/offline-rvm/current-rvm"
    try:
        pointer.symlink_to(model, target_is_directory=True)
        other.symlink_to(model, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    with pytest.raises(portable.PortableError, match="Linked files"):
        portable._backup_files(shared)
    other.unlink()
    files = portable._backup_files(shared)
    assert "weights/current-rvm" not in files
    assert files["weights/offline-rvm/weights.pth"].read_bytes() == b"checkpoint"


def test_backup_link_filter_is_scoped_to_weights_root(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    pointer = shared / "weights/current-rvm"
    nested = shared / "weights/offline-rvm/current-rvm"
    pointer.mkdir(parents=True)
    nested.mkdir(parents=True)
    (shared / "weights/offline-rvm/weights.pth").write_bytes(b"checkpoint")
    original_is_symlink = Path.is_symlink
    original_reject = portable._reject_link

    def is_symlink(path):
        return path in {pointer, nested} or original_is_symlink(path)

    def reject(path):
        if path == nested:
            raise portable.PortableError(f"Linked files are not portable: {path}")
        return original_reject(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setattr(portable, "_reject_link", reject)
    with pytest.raises(portable.PortableError, match="Linked files"):
        portable._backup_files(shared)
    nested.rmdir()
    files = portable._backup_files(shared)
    assert "weights/current-rvm" not in files
    assert "weights/offline-rvm/weights.pth" in files
