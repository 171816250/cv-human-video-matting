"""Pre-download model code and weights used by the demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


MODEL_CHOICES = [
    "rvm_mobilenetv3",
    "rvm_resnet50",
    "depth_midas",
    "birefnet",
    "all",
]


def _download_rvm(model_name: str) -> None:
    import torch

    from cv_human_video_matting.matting.rvm import RVM_HUB_REF

    print(f"Downloading RVM {model_name} through torch.hub...")
    model = torch.hub.load(
        RVM_HUB_REF,
        model_name,
        pretrained=True,
        trust_repo=True,
    )
    model.eval()
    print(f"RVM {model_name} is ready.")


def _download_depth_midas() -> None:
    from cv_human_video_matting.depth import DepthEstimator

    print("Downloading MiDaS small model through torch.hub...")
    DepthEstimator(device="cpu", model_type="MiDaS_small")
    print("MiDaS small model is ready.")


def _download_birefnet() -> None:
    from cv_human_video_matting.matting.birefnet import BiRefNetRunner

    print("Downloading BiRefNet through HuggingFace/transformers...")
    BiRefNetRunner(device="cpu")
    print("BiRefNet is ready.")


def _expand_models(models: list[str]) -> list[str]:
    if "all" not in models:
        return models
    # Keep BiRefNet opt-in because it is larger and slower to download/load.
    expanded = ["rvm_mobilenetv3", "rvm_resnet50", "depth_midas"]
    return expanded + [model for model in models if model not in {"all", *expanded}]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-download model dependencies.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["rvm_mobilenetv3"],
        help=(
            "Models to prepare. 'all' includes RVM MobileNetV3, RVM ResNet50, "
            "and MiDaS; choose 'birefnet' explicitly for the heavy model."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    models = _expand_models(args.models)

    for model in models:
        if model == "rvm_mobilenetv3":
            _download_rvm("mobilenetv3")
        elif model == "rvm_resnet50":
            _download_rvm("resnet50")
        elif model == "depth_midas":
            _download_depth_midas()
        elif model == "birefnet":
            _download_birefnet()
        else:
            raise ValueError(f"Unsupported model selection: {model}")

    print("Requested model downloads completed.")


if __name__ == "__main__":
    main()
