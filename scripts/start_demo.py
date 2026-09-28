"""Start the local offline demo using this project's .venv; never install packages.

The launcher runs from the project directory. Explicit argument paths are resolved
from the caller's directory first; paths within YAML use the project directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def project_python() -> Path:
    return PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def discover_model_bundle() -> Path:
    """Choose only one fully verified RVM bundle under the two delivery locations."""
    from cv_human_video_matting.utils.model_bundle import verify_model_bundle

    valid = {}
    for search in (PROJECT_ROOT / "weights", PROJECT_ROOT.parent / "evidence"):
        if not search.is_dir():
            continue
        for manifest in sorted(search.rglob("manifest.json")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("kind") != "rvm_model_bundle":
                    continue
                identity = verify_model_bundle(manifest.parent)
                path = Path(identity["bundle_path"])
                valid[path] = path
            except (OSError, ValueError) as exc:
                print(f"Ignoring invalid model bundle {manifest.parent}: {exc}", file=sys.stderr)
    if len(valid) != 1:
        if not valid:
            raise ValueError("No verified offline RVM bundle found in weights/ or ../evidence/. "
                             "Supply --model-bundle PATH; no model will be downloaded.")
        raise ValueError(f"Found {len(valid)} verified offline RVM bundles. "
                         "Supply --model-bundle PATH to choose one explicitly.")
    return next(iter(valid))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--model-bundle", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if not 1 <= args.port <= 65535:
        parser.error("Server port must be between 1 and 65535.")
    interpreter = project_python()
    if not interpreter.is_file():
        parser.error(f"Project environment is missing: {interpreter}. "
                     "Create .venv and install the documented dependencies first; nothing was installed.")
    if Path(sys.executable).resolve() != interpreter.resolve():
        # Keep the caller's cwd until the child has resolved explicit path options.
        return subprocess.run([str(interpreter), "-B", str(Path(__file__).resolve()), *arguments],
                              check=False).returncode

    config_path = args.config.absolute().resolve() if args.config is not None else PROJECT_ROOT / "configs/default.yaml"
    requirements = (args.requirements.absolute().resolve() if args.requirements is not None
                    else PROJECT_ROOT / "requirements-demo.txt")
    explicit_bundle = args.model_bundle.absolute().resolve() if args.model_bundle is not None else None
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        # Apply before Gradio imports, including the diagnostic import check.
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        from cv_human_video_matting import cli
        from cv_human_video_matting.utils.config import load_config
        import yaml

        config = load_config(config_path)
        configured_bundle = config.get("matting", {}).get("bundle")
        if explicit_bundle is not None:
            config = cli.config_with_model_bundle(config, explicit_bundle)
        elif configured_bundle is not None:
            config = cli.config_with_model_bundle(config, configured_bundle, select_model=False)
        else:
            config = cli.config_with_model_bundle(config, discover_model_bundle())
        config.setdefault("depth", {})["enabled"] = False
        print(f"Offline model: {config['matting']['bundle']}")
        print(f"Runtime: {sys.executable}")
        print("Working directory: " + str(PROJECT_ROOT))
        # The config is launcher-owned, never edits the user's YAML, and survives
        # throughout the blocking UI run. Doctor checks the exact UI configuration.
        with TemporaryDirectory(prefix="matting-start-") as temporary:
            effective = Path(temporary) / "effective.yaml"
            effective.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            cli.main(["--config", str(effective), "doctor", "--requirements", str(requirements)])
            cli.main(["--config", str(effective), "ui", "--host", "127.0.0.1",
                      "--port", str(args.port), "--no-share"])
        return 0
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    raise SystemExit(main())
