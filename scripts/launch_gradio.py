"""Compatibility UI entry point; config-relative assets use the project directory.

Explicit --config and --model-bundle paths are resolved from the caller's directory
before changing cwd. All UI options are forwarded to the main CLI.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cv_human_video_matting import cli  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-bundle", type=Path)
    args = parser.parse_args(argv)
    config = args.config.absolute().resolve() if args.config is not None else PROJECT_ROOT / "configs/default.yaml"
    bundle = args.model_bundle.absolute().resolve() if args.model_bundle is not None else None
    forwarded = ["--config", str(config), "ui", "--host", args.host, "--port", str(args.port),
                 "--share" if args.share else "--no-share"]
    if bundle is not None:
        forwarded.extend(["--model-bundle", str(bundle)])
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        return cli.main(forwarded)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
