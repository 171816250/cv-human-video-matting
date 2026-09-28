"""Run a local RVM rehearsal with an empty hub cache and Python network guards.

This verifies the supplied sample only, not real-person quality, an operating
system firewall, dependency installation, or optional depth/BiRefNet backends.
"""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; existing paths refused.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1 or args.max_frames < 1:
        parser.error("runs and max-frames must be positive integers")

    import torch
    from cv_human_video_matting.cli import main as run_cli
    from cv_human_video_matting.evaluation import write_evaluation_summary
    from cv_human_video_matting.utils.model_bundle import verify_model_bundle

    bundle = verify_model_bundle(args.bundle)
    if not Path(args.input).is_file():
        parser.error("input file does not exist")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    previous_hub = torch.hub.get_dir()
    attempts = []

    def deny_network(*unused_args, **unused_kwargs):
        attempts.append("network/download attempted")
        raise RuntimeError("Offline rehearsal blocked a network/download attempt.")

    reports = []
    try:
        with TemporaryDirectory(prefix="empty-rvm-hub-") as empty_hub:
            torch.hub.set_dir(empty_hub)
            with patch("socket.socket.connect", deny_network), \
                    patch("socket.create_connection", deny_network), \
                    patch("torch.hub.download_url_to_file", deny_network):
                for index in range(1, args.runs + 1):
                    report = output / f"run-{index}.json"
                    run_cli(["matte", "--model", f"rvm_{bundle['model_name']}",
                        "--model-bundle", str(Path(args.bundle).resolve()), "--device", args.device,
                        "--input", str(Path(args.input).resolve()),
                        "--output", str(output / f"run-{index}.mp4"), "--report", str(report),
                        "--max-frames", str(args.max_frames), "--mode", "side_by_side",
                        "--temporal-smooth", "--temporal-mode", "adaptive", "--no-enable-depth"])
                    reports.append(report)
            cache_entries = sorted(p.relative_to(empty_hub).as_posix() for p in Path(empty_hub).rglob("*"))
    finally:
        torch.hub.set_dir(previous_hub)

    records = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    frames_match = all(item["output"]["frame_count"] == item["result"]["frames"] for item in records)
    identities_match = all(item["assets"]["model_bundle"]["content_sha256"] == bundle["content_sha256"]
                           and item["assets"]["model_bundle"]["load_identity_confirmed"] for item in records)
    passed = not attempts and not cache_entries and frames_match and identities_match
    result = {
        "passed": passed, "scope": "Supplied sample only; no real-person quality or sustained speed claim.",
        "network_guard": "Python socket connect/create_connection and torch hub downloads blocked; not an OS firewall.",
        "blocked_attempts": attempts, "empty_hub_entries_after_runs": cache_entries,
        "model_content_sha256": bundle["content_sha256"], "run_count": len(records),
        "decoded_frame_counts_match": frames_match, "model_identities_match": identities_match,
        "output_hashes": [item["output"]["sha256"] for item in records],
        "pipeline_fps": [item["result"]["fps"] for item in records],
    }
    (output / "offline-rehearsal.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_evaluation_summary(reports, output / "summary.html")
    print(f"Offline rehearsal {'passed' if passed else 'failed'}: {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
