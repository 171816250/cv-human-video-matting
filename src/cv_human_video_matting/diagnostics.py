"""Local preflight checks. This module never downloads or initializes model weights."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile


def _check_requirements(path: Path, record) -> None:
    """Evaluate declared package constraints against local metadata without pip/network."""
    try:
        from packaging.requirements import Requirement

        declarations = {}
        visited = set()

        def read_requirements(source: Path, stack: set) -> None:
            source = source.resolve()
            if source in stack:
                raise ValueError(f"Cyclic requirements include: {source}")
            if source in visited:
                return
            for raw in source.read_text(encoding="utf-8-sig").splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                include = re.fullmatch(r"(?:-r|--requirement)\s*(.+)", line)
                if include:
                    read_requirements(source.parent / include.group(1).strip(), stack | {source})
                    continue
                requirement = Requirement(line)
                if requirement.url:
                    raise ValueError(f"Cannot verify direct-URL requirement offline: {line}")
                if requirement.marker is not None and not requirement.marker.evaluate():
                    continue
                name = re.sub(r"[-_.]+", "-", requirement.name).lower()
                declarations.setdefault(name, []).append((requirement, source.name))
            visited.add(source)

        read_requirements(path, set())
        if not declarations:
            raise ValueError("Requirements profile contains no applicable package requirements.")
    except Exception as exc:
        record("requirements", "fail", f"Cannot verify declared dependencies in {path}: {exc}")
        return

    failed = []
    for name, entries in sorted(declarations.items()):
        constraints = "; ".join(f"{requirement} [{source}]" for requirement, source in entries)
        try:
            installed = version(name)
            compatible = all(requirement.specifier.contains(installed, prereleases=True)
                             for requirement, _ in entries)
            status = "pass" if compatible else "fail"
            detail = f"Installed {installed}; declared: {constraints}"
        except Exception as exc:
            status, detail = "fail", f"Cannot establish installed version: {exc}; declared: {constraints}"
        if status == "fail":
            failed.append(name)
        record(f"requirement:{name}", status, detail)
    record("requirements", "fail" if failed else "pass",
           f"Profile: {path.resolve()}; {len(declarations)} packages checked. "
           + (f"Unmet/unverified: {', '.join(failed)}" if failed else "All declared package constraints met."))


def _probe_ffv1(cv2, np) -> str:
    """Perform a tiny lossless encode/decode round trip using the delivery container."""
    frames = [np.full((16, 24, 3), color, dtype=np.uint8)
              for color in ((10, 80, 200), (240, 32, 5), (0, 255, 100))]
    with tempfile.TemporaryDirectory(prefix="matting-doctor-") as temporary:
        path = Path(temporary) / "lossless.mkv"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 12.0, (24, 16))
        try:
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open the FFV1/Matroska writer.")
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("OpenCV FFV1 writer produced no usable temporary file.")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not reopen the FFV1/Matroska file.")
            for expected in frames:
                ok, decoded = capture.read()
                if not ok or not np.array_equal(decoded, expected):
                    raise RuntimeError("FFV1 round trip lost frames or changed pixel values.")
            if capture.read()[0]:
                raise RuntimeError("FFV1 round trip produced an unexpected extra frame.")
        finally:
            capture.release()
    return "OpenCV FFV1/Matroska: 3 frames at 24x16 encoded and decoded losslessly; temporary file removed."


def check_environment(config_path="configs/default.yaml", requirements_path=None) -> dict:
    checks = []

    def record(name, status, detail):
        checks.append({"name": name, "status": status, "detail": str(detail)})

    record("python", "pass" if (3, 9) <= sys.version_info[:2] < (3, 12) else "fail",
           f"{platform.python_version()} ({sys.executable}); supported: 3.9–3.11")
    requirements = (Path(requirements_path) if requirements_path is not None
                    else Path(__file__).resolve().parents[2] / "requirements-demo.txt")
    _check_requirements(requirements, record)
    modules = {}

    def check_import(name, package):
        try:
            modules[name] = importlib.import_module(name)
            try:
                installed = version(package)
            except PackageNotFoundError:
                installed = "version unavailable"
            record(package, "pass", installed)
        except Exception as exc:
            record(package, "fail", f"{exc}. Check the runtime installation instructions.")

    for name, package in (("yaml", "PyYAML"), ("numpy", "numpy"), ("cv2", "opencv-python"), ("PIL", "Pillow"),
                          ("torch", "torch"), ("torchvision", "torchvision"),
                          ("gradio", "gradio"), ("imageio_ffmpeg", "imageio-ffmpeg")):
        check_import(name, package)

    config = {}
    config_valid = False
    try:
        from cv_human_video_matting.utils.config import load_config
        config = load_config(config_path)
        config_valid = True
        record("config", "pass", str(Path(config_path).resolve()))
    except Exception as exc:
        record("config", "fail", exc)

    if config_valid:
        matting = config.get("matting", {})
        model = matting.get("model", "rvm_mobilenetv3")
        checkpoint = matting.get("checkpoint")
        bundle = matting.get("bundle")
        if checkpoint is not None:
            try:
                path = Path(checkpoint)
                exists = path.is_file() and path.stat().st_size > 0
                record("checkpoint", "pass" if exists else "fail",
                       f"Configured checkpoint {'present' if exists else 'missing or empty'}: {path}")
            except (TypeError, OSError, ValueError) as exc:
                record("checkpoint", "fail", exc)
        if bundle is not None:
            try:
                from cv_human_video_matting.utils.model_bundle import verify_model_bundle
                if checkpoint is not None:
                    raise ValueError("matting.bundle and matting.checkpoint cannot be configured together.")
                if model not in {"rvm", "rvm_mobilenetv3", "rvm_resnet50", "mobilenetv3", "resnet50"}:
                    raise ValueError("matting.bundle supports only RVM models.")
                family = "resnet50" if "resnet50" in model else "mobilenetv3"
                identity = verify_model_bundle(bundle, expected_model=family)
                record("model-bundle", "pass",
                       f"RVM {family} local code and weight hashes verified: {identity['content_sha256']}; "
                       "does not use the global torch hub cache. Dependencies and MiDaS are not included; "
                       "hash verification is not an inference or source-trust guarantee.")
            except Exception as exc:
                record("model-bundle", "fail", exc)
        if model in {"birefnet", "ultra"}:
            for name in ("transformers", "timm", "safetensors"):
                check_import(name, name)
        elif config.get("depth", {}).get("enabled", False):
            check_import("timm", "timm")

    torch = modules.get("torch")
    if torch is not None:
        try:
            available = torch.cuda.is_available()
            requested = config.get("device", {}).get("mode", "auto")
            use_cuda = requested.startswith("cuda") or (requested == "auto" and available)
            if use_cuda and not available:
                record("device", "fail", f"Requested {requested}, but CUDA is unavailable; select cpu or auto.")
            elif use_cuda:
                index = int(requested.split(":", 1)[1]) if ":" in requested else 0
                count = torch.cuda.device_count()
                if index >= count:
                    record("device", "fail", f"Requested cuda:{index}; only {count} CUDA device(s) available.")
                else:
                    record("device", "pass", f"cuda:{index}: {torch.cuda.get_device_name(index)}")
            else:
                record("device", "pass", f"CPU selected; CUDA {'available' if available else 'unavailable'}")
        except Exception as exc:
            record("device", "fail", exc)

        if "numpy" in modules:
            # Catch ABI incompatibilities such as NumPy 2 with an older PyTorch.
            try:
                torch.from_numpy(modules["numpy"].zeros((2, 2), dtype="float32")).numpy()
                record("numpy-torch", "pass", "Array/tensor conversion works")
            except Exception as exc:
                record("numpy-torch", "fail", exc)

        if config_valid and bundle is None and (model.startswith("rvm") or model in {"mobilenetv3", "resnet50"}):
            try:
                hub = Path(torch.hub.get_dir())
                family = "resnet50" if "resnet50" in model else "mobilenetv3"
                code_cached = (hub / "PeterL1n_RobustVideoMatting_master" / "hubconf.py").is_file()
                weights = hub / "checkpoints" / f"rvm_{family}.pth"
                weights_cached = weights.is_file() and weights.stat().st_size > 0
                record("model-cache", "pass" if code_cached and weights_cached else "warn",
                       f"RVM {family}: code={code_cached}, weights={weights_cached}; {hub}. "
                       "Cache presence is not an inference test; run a sample before going offline.")
            except Exception as exc:
                record("model-cache", "fail", exc)
        elif config_valid and bundle is None and model != "dummy":
            record("model-cache", "warn", "Optional backend: verify its weights and a full run before the demo.")
    if config_valid and config.get("depth", {}).get("enabled", False):
        record("depth-model", "warn", "Depth is enabled; MiDaS code/weights are not included in the RVM bundle. "
               "Prepare depth separately and rehearse a full run before going offline.")

    if "cv2" in modules and "numpy" in modules:
        try:
            record("lossless-intermediate", "pass", _probe_ffv1(modules["cv2"], modules["numpy"]))
        except Exception as exc:
            record("lossless-intermediate", "fail", f"OpenCV FFV1/Matroska round trip failed: {exc}")

    ffmpeg = modules.get("imageio_ffmpeg")
    if ffmpeg is not None:
        try:
            result = subprocess.run(
                [ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-encoders"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=20, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            encoders = {fields[1] for line in result.stdout.splitlines()
                        if len(fields := line.split()) >= 2 and len(fields[0]) == 6
                        and fields[0][0] in "VAS"}
            missing = {"libx264", "aac"} - encoders
            if missing:
                raise RuntimeError(f"Required FFmpeg encoder(s) unavailable: {', '.join(sorted(missing))}")
            record("video-export", "pass", "FFmpeg H.264/AAC encoders available")
        except Exception as exc:
            record("video-export", "fail", exc)

    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "ready": not any(item["status"] == "fail" for item in checks),
            "requirements_profile": str(requirements.resolve()),
            "scope": "Local dependency versions/imports, config, model bundle integrity and codec checks only; warnings require a rehearsal. No model quality, source-trust or speed guarantee.",
            "checks": checks}


def save_diagnostics(report: dict, path: str | Path) -> Path:
    """Write UTF-8 JSON atomically, retaining any existing report on failure."""
    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError("Diagnostic report must use a .json extension.")
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}-",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
