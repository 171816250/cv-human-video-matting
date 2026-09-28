"""Application entry point for the iterative video matting project."""

import os
from pathlib import Path
import sys


# Force a headless matplotlib backend before any import can trigger backend
# auto-selection. Gradio wraps every request in a MatplotlibBackendManager that
# calls matplotlib.get_backend(); with PyQt5 installed, matplotlib would try the
# Qt backend, which crashes on some PyQt5/matplotlib version combos and breaks
# live-camera streaming. Agg is non-interactive and always safe for a server.
os.environ.setdefault("MPLBACKEND", "Agg")


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cv_human_video_matting.cli import main  # noqa: E402


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Processing interrupted; temporary resources have been released.", file=sys.stderr)
        raise SystemExit(130)
