"""Export a portable RVM model bundle from existing local cache files only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cv_human_video_matting.utils.model_bundle import prepare_model_bundle, verify_model_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export cached RVM code and weights. Never downloads models.")
    parser.add_argument("--output", required=True, help="New bundle directory; existing paths are refused.")
    parser.add_argument("--model", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    parser.add_argument("--hub-dir", help="Existing torch hub cache directory; defaults to torch.hub.get_dir().")
    args = parser.parse_args(argv)
    try:
        result = prepare_model_bundle(args.output, args.model, args.hub_dir)
        manifest = verify_model_bundle(result, expected_model=args.model)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"RVM bundle: {result}")
    print(f"Model: {manifest['model_name']}")
    print(f"Content SHA256: {manifest['content_sha256']}")
    print("Copied local cache only. Install runtime dependencies and rehearse inference before offline use.")


if __name__ == "__main__":
    main()
