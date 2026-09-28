"""Exercise packaging with tiny disposable repositories, without model/network access."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "package_release", Path(__file__).resolve().parents[1] / "scripts" / "package_release.py",
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True,
    ).stdout.decode("utf-8").strip()


@pytest.fixture
def tiny_repo(tmp_path):
    repo = tmp_path / "中文 user's repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Package Test")
    git(repo, "config", "user.email", "package@example.invalid")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "tiny"\nversion = "0.2.0rc1"\n', encoding="utf-8",
    )
    (repo / "README.md").write_bytes(b"first\n")
    (repo / ".gitignore").write_text("weights/\noutput/\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "initial")
    (repo / "README.md").write_bytes(b"current\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "second")
    return repo


def test_candidate_contains_committed_source_explicit_evidence_and_recoverable_history(tiny_repo, tmp_path):
    evidence = tmp_path / "答辩证据"
    evidence.mkdir()
    (evidence / "result.json").write_text('{"passed": true}\n', encoding="utf-8")
    (tiny_repo / "_git_private.txt").write_text("private scratch", encoding="utf-8")
    (tiny_repo / "weights").mkdir()
    (tiny_repo / "weights" / "fake.pth").write_bytes(b"not a real model")
    output = release.package_release(tiny_repo, tmp_path / "候选包", [evidence])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == git(tiny_repo, "rev-parse", "HEAD")
    assert manifest["branch"] == "main"
    assert manifest["package_version"] == "0.2.0rc1"
    assert manifest["untracked_files_excluded"] == 1
    assert manifest["remote_release"] is False
    with ZipFile(output / "candidate.zip") as archive:
        assert archive.read("source/README.md") == b"current\n"
        assert archive.read("evidence/答辩证据/result.json") == (evidence / "result.json").read_bytes()
        assert "repository.bundle" in archive.namelist()
        assert not any("_git_private" in name or "fake.pth" in name for name in archive.namelist())
        for name, metadata in manifest["zip_entries"].items():
            contents = archive.read(name)
            assert metadata == {"sha256": hashlib.sha256(contents).hexdigest(), "size_bytes": len(contents)}
    for name, metadata in manifest["files"].items():
        contents = (output / name).read_bytes()
        assert metadata == {"sha256": hashlib.sha256(contents).hexdigest(), "size_bytes": len(contents)}
    restored = tmp_path / "restored"
    git(tmp_path, "clone", str(output / "repository.bundle"), str(restored))
    assert git(restored, "rev-parse", "HEAD") == manifest["commit"]
    assert git(restored, "rev-list", "--count", "HEAD") == "2"
    assert (tiny_repo / "_git_private.txt").read_text() == "private scratch"


@pytest.mark.parametrize("staged", [False, True])
def test_dirty_tracked_changes_refused(tiny_repo, tmp_path, staged):
    (tiny_repo / "README.md").write_text("uncommitted", encoding="utf-8")
    if staged:
        git(tiny_repo, "add", "README.md")
    output = tmp_path / "package"
    with pytest.raises(release.PackagingError, match="uncommitted"):
        release.package_release(tiny_repo, output)
    assert not output.exists()


def test_existing_package_is_untouched(tiny_repo, tmp_path):
    output = tmp_path / "package"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(release.PackagingError, match="already exists"):
        release.package_release(tiny_repo, output)
    assert marker.read_text() == "keep"


def test_evidence_name_collision_is_rejected(tiny_repo, tmp_path):
    first = tmp_path / "one" / "report.json"
    second = tmp_path / "two" / "REPORT.json"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(release.PackagingError, match="collide"):
        release.package_release(tiny_repo, tmp_path / "package", [first, second])


def test_output_inside_evidence_is_rejected(tiny_repo, tmp_path):
    with pytest.raises(release.PackagingError, match="contain one another"):
        release.package_release(tiny_repo, tmp_path / "package", [tmp_path])


def test_directory_symlink_evidence_rejected(tiny_repo, tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    link = evidence / "linked"
    try:
        link.symlink_to(tiny_repo, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires permission on this Windows host")
    with pytest.raises(release.PackagingError, match="links/junctions"):
        release.package_release(tiny_repo, tmp_path / "package", [evidence])


def test_failed_verification_cleans_only_own_staging(tiny_repo, tmp_path, monkeypatch):
    unrelated = tmp_path / ".package.staging-other"
    unrelated.mkdir()
    marker = unrelated / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    original = release._verify_zip

    def fail_final(path, expected=None):
        if expected is not None:
            raise release.PackagingError("simulated verification failure")
        return original(path)

    monkeypatch.setattr(release, "_verify_zip", fail_final)
    with pytest.raises(release.PackagingError, match="simulated"):
        release.package_release(tiny_repo, tmp_path / "package")
    assert not (tmp_path / "package").exists()
    assert list(tmp_path.glob(".package.staging-*")) == [unrelated]
    assert marker.read_text() == "keep"


def test_changed_evidence_fails_content_hash_verification(tiny_repo, tmp_path, monkeypatch):
    evidence = tmp_path / "report.txt"
    evidence.write_text("before", encoding="utf-8")
    original = release._append_file

    def mutate_then_append(archive, path, name, date_time):
        if path == evidence:
            path.write_text("after", encoding="utf-8")
        return original(archive, path, name, date_time)

    monkeypatch.setattr(release, "_append_file", mutate_then_append)
    with pytest.raises(release.PackagingError, match="SHA256"):
        release.package_release(tiny_repo, tmp_path / "package", [evidence])
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "source/../bad.txt", "C:/bad.txt"])
def test_unsafe_zip_paths_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as target:
        target.writestr(name, b"bad")
    with pytest.raises(release.PackagingError, match="Unsafe"):
        release._verify_zip(archive)


def test_backslash_path_rejected_before_windows_zipfile_can_normalize_it():
    with pytest.raises(release.PackagingError, match="Unsafe"):
        release._archive_name("source\\bad.txt")


def test_repeated_package_has_identical_content(tiny_repo, tmp_path):
    first = release.package_release(tiny_repo, tmp_path / "first")
    second = release.package_release(tiny_repo, tmp_path / "second")
    for name in ("candidate.zip", "START_HERE.md", "manifest.json", "repository.bundle"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_git_trust_is_command_scoped_and_timeout_is_friendly(tmp_path, monkeypatch):
    calls = []

    def timeout(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(release.subprocess, "run", timeout)
    with pytest.raises(release.PackagingError, match="timed out.*no package was published"):
        release._git(tmp_path, "status", "--porcelain=v1")
    command, kwargs = calls[0]
    assert command[:5] == [
        "git", "-c", f"safe.directory={tmp_path.resolve().as_posix()}", "-C", str(tmp_path.resolve()),
    ]
    assert "--global" not in command
    assert 0 < kwargs["timeout"] <= 300
