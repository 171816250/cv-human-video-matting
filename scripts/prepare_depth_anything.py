"""Prepare and verify the official Depth Anything V2 Small Hugging Face snapshot.

The CLI writes only below this repository's Git-ignored ``weights/`` directory.
``huggingface_hub`` is needed for preparation, but verification uses stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = PROJECT_ROOT / "weights"
DEFAULT_OUTPUT = WEIGHTS_ROOT / "depth-anything-v2-small-hf"
REPO_ID = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
LICENSE_ID = "apache-2.0"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
MODEL_URL = f"https://huggingface.co/{REPO_ID}/tree/{REVISION}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_model_files(directory: Path) -> dict[str, dict[str, int | str]]:
    metadata = {}
    for name in REQUIRED_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing regular model file: {path}")
        size = path.stat().st_size
        if size == 0:
            raise ValueError(f"Empty model file: {path}")
        metadata[name] = {"size": size, "sha256": _sha256(path)}

    config = _read_json_object(directory / "config.json")
    if config.get("model_type") != "depth_anything":
        raise ValueError("config.json is not a Depth Anything checkpoint")
    if config.get("depth_estimation_type", "relative") != "relative":
        raise ValueError("config.json must specify relative depth estimation")
    if config.get("use_pretrained_backbone", False):
        raise ValueError("config.json must contain local backbone weights")
    if config.get("backbone") is not None:
        raise ValueError("config.json must not name a separately loaded backbone")
    if config.get("use_timm_backbone", False):
        raise ValueError("config.json must not request a timm backbone")
    if config.get("backbone_kwargs"):
        raise ValueError("config.json must not request separate backbone options")
    backbone = config.get("backbone_config")
    if not isinstance(backbone, dict) or backbone.get("model_type") != "dinov2":
        raise ValueError("config.json must specify the official DINOv2 backbone")
    if backbone.get("hidden_size") != 384 or backbone.get("patch_size") != 14:
        raise ValueError("config.json is not the V2 Small architecture")

    processor = _read_json_object(directory / "preprocessor_config.json")
    if processor.get("image_processor_type") != "DPTImageProcessor":
        raise ValueError("preprocessor_config.json must specify DPTImageProcessor")

    with (directory / "model.safetensors").open("rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("model.safetensors is a Git LFS pointer, not model weights")
    if metadata["model.safetensors"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("model.safetensors SHA256 does not match the official pinned weights")
    return metadata


def _manifest(files: dict[str, dict[str, int | str]]) -> dict:
    return {
        "schema_version": 1,
        "kind": "depth_anything_v2_small_hf",
        "model_version": "Depth Anything V2 Small",
        "source": {"repo_id": REPO_ID, "revision": REVISION, "url": MODEL_URL},
        "license": {
            "id": LICENSE_ID,
            "source": f"https://huggingface.co/{REPO_ID}/blob/{REVISION}/README.md",
        },
        "files": files,
    }


def verify_depth_anything(directory: str | Path) -> dict:
    """Verify local assets, manifest, and the published official weights hash."""
    directory = Path(directory).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(f"Depth Anything directory not found: {directory}")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"Depth Anything manifest not found: {manifest_path}")
    actual = _read_json_object(manifest_path)
    files = _validate_model_files(directory)
    if actual != _manifest(files):
        raise ValueError("Depth Anything manifest does not match the pinned source or local file hashes")
    return actual


def _snapshot_download(**kwargs):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Preparing Depth Anything requires the optional huggingface_hub package"
        ) from exc
    return snapshot_download(**kwargs)


def prepare_depth_anything(
    directory: str | Path = DEFAULT_OUTPUT,
    *,
    snapshot_download_fn: Callable | None = None,
) -> dict:
    """Download the exact official revision without overwriting any existing path.

    ``snapshot_download_fn`` allows offline tests to supply deterministic files.
    The command-line entry point restricts ``directory`` to ``weights/``.
    """
    directory = Path(directory).expanduser()
    if directory.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {directory}")
    if directory.exists():
        return verify_depth_anything(directory)

    directory.parent.mkdir(parents=True, exist_ok=True)
    download_dir = Path(tempfile.mkdtemp(prefix=".depth-anything-download-", dir=directory.parent))
    staged_dir = Path(tempfile.mkdtemp(prefix=".depth-anything-bundle-", dir=directory.parent))
    try:
        downloader = snapshot_download_fn or _snapshot_download
        downloader(
            repo_id=REPO_ID,
            revision=REVISION,
            repo_type="model",
            allow_patterns=list(REQUIRED_FILES),
            local_dir=str(download_dir),
            cache_dir=str(download_dir / ".hub-cache"),
            token=False,
        )
        for name in REQUIRED_FILES:
            source = download_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Official snapshot is missing {name}")
            shutil.copyfile(source, staged_dir / name)

        files = _validate_model_files(staged_dir)
        (staged_dir / "manifest.json").write_text(
            json.dumps(_manifest(files), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = verify_depth_anything(staged_dir)
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Refusing to replace existing path: {directory}")
        os.rename(staged_dir, directory)
        return manifest
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)
        shutil.rmtree(staged_dir, ignore_errors=True)


def _cli_output(value: str) -> Path:
    if WEIGHTS_ROOT.is_symlink():
        raise ValueError(f"Refusing symlinked weights directory: {WEIGHTS_ROOT}")
    root = WEIGHTS_ROOT.resolve()
    output = Path(value).expanduser().absolute().resolve()
    if output.parent != root:
        raise ValueError(f"Output must be a direct child of the Git-ignored {WEIGHTS_ROOT}")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned official Depth Anything V2 Small model under weights/."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Destination directory directly below weights/ (default: weights/depth-anything-v2-small-hf).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the local directory and manifest without importing Hugging Face or using the network.",
    )
    args = parser.parse_args(argv)
    try:
        output = _cli_output(args.output)
        manifest = verify_depth_anything(output) if args.verify_only else prepare_depth_anything(output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Depth Anything V2 Small ready: {output}")
    print(f"Revision: {manifest['source']['revision']}")
    print(f"model.safetensors SHA256: {manifest['files']['model.safetensors']['sha256']}")


if __name__ == "__main__":
    main()
