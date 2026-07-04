"""Application entry point for the iterative video matting project."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cv_human_video_matting.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
