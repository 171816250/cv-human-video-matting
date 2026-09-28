"""Local preflight checks. This module never downloads or initializes model weights."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def check_environment(config_path="configs/default.yaml") -> dict:
    checks = []

    def record(name, status, detail):
        checks.append({"name": name, "status": status, "detail": str(detail)})

    record("python", "pass" if (3, 9) <= sys.version_info[:2] < (3, 12) else "fail",
           f"{platform.python_version()} ({sys.executable}); supported: 3.9–3.11")
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

    for name, package in (("yaml", "PyYAML"), ("numpy", "numpy"), ("cv2", "opencv-python"),
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
        if checkpoint is not None:
            try:
                path = Path(checkpoint)
                exists = path.is_file() and path.stat().st_size > 0
                record("checkpoint", "pass" if exists else "fail",
                       f"Configured checkpoint {'present' if exists else 'missing or empty'}: {path}")
            except (TypeError, OSError, ValueError) as exc:
                record("checkpoint", "fail", exc)
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

        if config_valid and (model.startswith("rvm") or model in {"mobilenetv3", "resnet50"}):
            try:
                hub = Path(torch.hub.get_dir())
                family = "resnet50" if "resnet50" in model else "mobilenetv3"
                code_cached = (hub / "PeterL1n_RobustVideoMatting_master" / "hubconf.py").is_file()
                weights_cached = any(path.is_file() and path.stat().st_size > 0
                                     for path in (hub / "checkpoints").glob(f"rvm_{family}*.pth"))
                record("model-cache", "pass" if code_cached and weights_cached else "warn",
                       f"RVM {family}: code={code_cached}, weights={weights_cached}; {hub}. "
                       "Cache presence is not an inference test; run a sample before going offline.")
            except Exception as exc:
                record("model-cache", "fail", exc)
        elif config_valid and model != "dummy":
            record("model-cache", "warn", "Optional backend: verify its weights and a full run before the demo.")
    if config_valid and config.get("depth", {}).get("enabled", False):
        record("depth-model", "warn", "Depth is enabled; run a sample with this model before going offline.")

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
            "scope": "Local dependency/config checks only; warnings require a rehearsal. No model quality or speed guarantee.",
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
