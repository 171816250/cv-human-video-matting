"""设置运行环境并进入视频抠像程序。"""

import os
from pathlib import Path
import sys


# 未指定绘图后端时使用 Agg，避免网页请求触发 Qt 兼容问题。
# 在导入 Gradio 前设置；保留用户已指定的后端。
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
