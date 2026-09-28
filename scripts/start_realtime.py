"""Start the GPU camera workspace in its separate .venv-gpu environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def project_python():
    return PROJECT_ROOT / ".venv-gpu" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", type=Path)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--max-width", type=int, choices=[320, 640, 960, 1280], default=640)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if not 1 <= args.port <= 65535:
        parser.error("Server port must be between 1 and 65535.")
    interpreter = project_python()
    if not interpreter.is_file():
        parser.error(f"GPU environment missing: {interpreter}; see docs/REALTIME.md.")
    if Path(sys.executable).resolve() != interpreter.resolve():
        return subprocess.run([str(interpreter), "-B", str(Path(__file__).resolve()), *arguments],
                              check=False).returncode
    bundle = args.model_bundle.absolute().resolve() if args.model_bundle is not None else None
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        from cv_human_video_matting import cli
        from start_demo import discover_model_bundle
        import yaml

        config = cli.config_with_model_bundle(
            {"device": {"mode": "cuda"}, "depth": {"enabled": False}},
            bundle if bundle is not None else discover_model_bundle(),
        )
        print(f"GPU runtime: {sys.executable}", flush=True)
        print(f"Offline model: {config['matting']['bundle']}", flush=True)
        with TemporaryDirectory(prefix="matting-realtime-") as temporary:
            effective = Path(temporary) / "config.yaml"
            effective.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            cli.main(["--config", str(effective), "doctor"])
            cli.main(["--config", str(effective), "realtime", "--host", "127.0.0.1",
                      "--port", str(args.port), "--max-width", str(args.max_width), "--device", "cuda"])
        return 0
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    raise SystemExit(main())
