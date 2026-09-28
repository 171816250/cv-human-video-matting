"""Exercise Gradio's real progress/request injection and two HTTP sessions locally."""

import json
from pathlib import Path
import socket
from threading import Event

import cv2
import numpy as np
from gradio_client import Client, handle_file

from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.ui import gradio_app as ui
from cv_human_video_matting.ui.jobs import BatchJobs


def test_gradio_http_cancel_then_download_report(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    assert writer.isOpened()
    for _ in range(24):
        writer.write(np.full((48, 64, 3), 100, np.uint8))
    writer.release()
    entered, release = Event(), Event()
    class HeldRunner(DummyMattingRunner):
        def predict(self, frame):
            entered.set()
            assert release.wait(20)
            return super().predict(frame)
    monkeypatch.setattr(ui, "create_matting_runner", lambda **kwargs: HeldRunner())
    monkeypatch.setattr(ui, "_BATCH_JOBS", BatchJobs())
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {})
    output = tmp_path / "results"
    demo = ui.build_demo({"paths": {"output_dir": str(output)}})
    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        port = channel.getsockname()[1]
    clients = []
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, share=False,
                    prevent_thread_lock=True, quiet=True)
        owner = Client(f"http://127.0.0.1:{port}", verbose=False)
        other = Client(f"http://127.0.0.1:{port}", verbose=False)
        clients.extend([owner, other])
        args = [handle_file(str(source)), None, "rvm_mobilenetv3", "green", 0, "cpu",
                0, 0, 0, False, 0.75, False, 5, False, 0, "#00FF00"]
        job = owner.submit(*args, api_name="/process_video")
        assert entered.wait(20)
        assert owner.session_hash in ui._BATCH_JOBS._active
        assert "没有" in other.predict(api_name="/stop_processing")
        assert "收到停止" in owner.predict(api_name="/stop_processing")
        release.set()
        cancelled = job.result(timeout=20)
        assert "已停止" in cancelled[2]
        assert not list(output.iterdir())
        monkeypatch.setattr(ui, "create_matting_runner", lambda **kwargs: DummyMattingRunner())
        args[4] = 12
        succeeded = owner.predict(*args, api_name="/process_video")
        assert succeeded[0] and succeeded[3]
        downloaded = succeeded[3]["value"] if isinstance(succeeded[3], dict) else succeeded[3]
        report = json.loads(Path(downloaded).read_text(encoding="utf-8"))
        assert report["result"]["frames"] == 12
        assert Path(report["result"]["output_path"]).is_file()
        assert report["schema_version"] == 3
    finally:
        release.set()
        for client in clients:
            client.close()
        demo.close()
