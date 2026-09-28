"""Launch the Gradio demo with the local src package path configured."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cv_human_video_matting.ui.gradio_app import launch_demo  # noqa: E402
from cv_human_video_matting.utils.config import load_config  # noqa: E402


if __name__ == "__main__":
    launch_demo(server_name="127.0.0.1", server_port=7860, share=False,
                config=load_config(PROJECT_ROOT / "configs" / "default.yaml"))
