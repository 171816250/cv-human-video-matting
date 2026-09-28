"""Portable, hash-verified RVM code and weights copied from an existing local cache.

No function in this module downloads models. The manifest records local bytes;
it is an integrity record, not an upstream signature or an inference guarantee.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from tempfile import TemporaryDirectory


MODEL_NAMES = {"mobilenetv3", "resnet50"}
CACHE_CODE_DIR = "PeterL1n_RobustVideoMatting_master"
_REQUIRED_CODE = {
    "hubconf.py", "LICENSE", "model/__init__.py", "model/model.py", "model/mobilenetv3.py",
    "model/resnet.py", "model/lraspp.py", "model/decoder.py", "model/fast_guided_filter.py",
    "model/deep_guided_filter.py",
}


def _model_name(name) -> str:
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise ValueError(f"Unsupported RVM model name: {name}; choose mobilenetv3 or resnet50.")
    return name


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file_record(path: Path) -> dict:
    initial = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    final = path.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError(f"Model file changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size_bytes": final.st_size}


def _excluded(relative: PurePosixPath) -> bool:
    return any(part in {".git", "__pycache__"} for part in relative.parts) or relative.suffix in {".pyc", ".pyo"}


def _inventory(root: Path) -> dict[str, Path]:
    """Inventory regular files while excluding caches and rejecting linked escapes."""
    root = root.resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _excluded(PurePosixPath(relative.as_posix())):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Model bundle/cache must contain regular local files, not links: {path}")
        if path.is_file():
            result[relative.as_posix()] = path
    return result


def _safe_relative(value) -> str:
    if (
        not isinstance(value, str) or not value or "\\" in value or ":" in value
        or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"Invalid relative manifest path: {value!r}")
    return value


def verify_model_bundle(path, expected_model=None) -> dict:
    """Verify all recorded files and return manifest plus resolved paths and identity.

    Added source/weight files, missing files, size changes and SHA256 changes fail.
    Python bytecode caches created by a previous local run are ignored. A bundle
    may be moved without changing its stable ``content_sha256`` identity.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RVM model bundle directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"RVM bundle manifest is missing or not a regular file: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read RVM bundle manifest {manifest_path}: {exc}") from exc
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("kind") != "rvm_model_bundle"):
        raise ValueError(f"Unsupported RVM model bundle manifest: {manifest_path}")
    model_name = _model_name(manifest.get("model_name"))
    if expected_model is not None and _model_name(expected_model) != model_name:
        raise ValueError(f"RVM bundle model mismatch: expected {expected_model}, contains {model_name}.")
    weights_relative = f"weights/rvm_{model_name}.pth"
    if manifest.get("code_dir") != "code" or manifest.get("weights_path") != weights_relative:
        raise ValueError("RVM bundle must use code/ and its matching weights/rvm_<model>.pth file.")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("RVM bundle manifest must contain a nonempty files list.")
    expected, folded = {}, set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid RVM bundle file record.")
        name = _safe_relative(item.get("path"))
        if name == "manifest.json" or _excluded(PurePosixPath(name)):
            raise ValueError(f"Manifest cannot track itself or excluded cache artifacts: {name}")
        if name in expected or name.casefold() in folded:
            raise ValueError(f"Duplicate RVM bundle manifest path: {name}")
        if (not isinstance(item.get("sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) is None
                or type(item.get("size_bytes")) is not int or item["size_bytes"] < 0):
            raise ValueError(f"Invalid SHA256 or size in RVM bundle manifest: {name}")
        expected[name] = item
        folded.add(name.casefold())
    required = {"code/" + name for name in _REQUIRED_CODE} | {weights_relative}
    missing_required = required - expected.keys()
    if missing_required:
        raise ValueError(f"RVM manifest lacks required files: {', '.join(sorted(missing_required))}")
    inventory = _inventory(root)
    inventory.pop("manifest.json", None)
    missing, extra = expected.keys() - inventory.keys(), inventory.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"RVM bundle file inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    for name, item in expected.items():
        if _file_record(inventory[name]) != {"sha256": item["sha256"], "size_bytes": item["size_bytes"]}:
            raise ValueError(f"RVM bundle file hash/size mismatch: {name}")
    if expected[weights_relative]["size_bytes"] == 0:
        raise ValueError(f"RVM bundle weights file is empty: {weights_relative}")
    normalized = [{"path": name, "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
                  for name, item in sorted(expected.items())]
    if manifest.get("content_sha256") != _canonical_hash(normalized):
        raise ValueError("RVM bundle manifest content_sha256 does not match its file records.")
    return {
        **manifest, "bundle_path": str(root), "code_path": str(root / "code"),
        "weights_file": str(root / weights_relative),
        "manifest_sha256": _file_record(manifest_path)["sha256"],
    }


def prepare_model_bundle(output_path, model_name="mobilenetv3", hub_dir=None) -> Path:
    """Copy cached upstream code and the exact matching weight filename, without network.

    Existing output directories are never overwritten. Preparation takes place in
    a sibling temporary directory and publishes only after complete verification.
    """
    model_name = _model_name(model_name)
    if hub_dir is None:
        import torch
        hub_dir = torch.hub.get_dir()
    hub = Path(hub_dir).resolve()
    source = hub / CACHE_CODE_DIR
    weight = hub / "checkpoints" / f"rvm_{model_name}.pth"
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing RVM bundle target: {output}")
    if output.is_relative_to(hub):
        raise ValueError("Export the RVM bundle outside the source torch hub cache.")
    if not source.is_dir():
        raise FileNotFoundError(f"Cached RVM code is missing: {source}. No download was attempted.")
    if not weight.is_file() or weight.is_symlink() or weight.stat().st_size == 0:
        raise FileNotFoundError(f"Cached RVM weights are missing or empty: {weight}. No download was attempted.")
    source_files = _inventory(source)
    if _REQUIRED_CODE - source_files.keys():
        raise ValueError(f"Cached RVM source is incomplete: {sorted(_REQUIRED_CODE - source_files.keys())}")
    source_records = {name: _file_record(path) for name, path in source_files.items()}
    weight_record = _file_record(weight)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        files = []
        for name, path in source_files.items():
            destination = staged / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if _file_record(destination) != source_records[name]:
                raise ValueError(f"Cached RVM source changed while being copied: {path}")
            files.append({"path": "code/" + name, **source_records[name]})
        weights_relative = f"weights/rvm_{model_name}.pth"
        destination = staged / weights_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(weight, destination)
        if _file_record(destination) != weight_record:
            raise ValueError(f"Cached RVM weights changed while being copied: {weight}")
        files.append({"path": weights_relative, **weight_record})
        files.sort(key=lambda item: item["path"])
        manifest = {
            "schema_version": 1, "kind": "rvm_model_bundle", "model_name": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_dir": "code", "weights_path": weights_relative,
            "upstream": {"repository": "PeterL1n/RobustVideoMatting", "requested_ref": "master",
                         "resolved_commit": None, "ref_is_immutable": False,
                         "note": "master is a mutable upstream reference; local code/weights are identified by SHA256, not an assumed upstream commit."},
            "integrity_scope": "Local file hashes only; not an upstream signature, dependency bundle, or inference/quality guarantee.",
            "files": files, "content_sha256": _canonical_hash(files),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_model_bundle(staged, expected_model=model_name)
        if output.exists():
            raise FileExistsError(f"RVM bundle target appeared during preparation: {output}")
        staged.rename(output)
    return output
