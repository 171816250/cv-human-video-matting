# RVM MVP Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first real matting pipeline: input video -> alpha matte -> foreground/green/side-by-side output video, with RVM as the production runner and a deterministic dummy runner for tests.

**Architecture:** Keep the pipeline model-agnostic: video I/O stays in `io/`, alpha compositing stays in `render/`, model wrappers stay in `matting/`, and CLI only wires pieces together. Tests should validate the pipeline with a cheap deterministic runner so the project remains testable without downloading RVM weights.

**Tech Stack:** Python, OpenCV, NumPy, PyTorch, PyYAML, pytest.

**Execution Status:** Implemented and verified on 2026-06-23. Full test suite passed with 13 tests. Dummy runner smoke and RVM CPU smoke both generated output videos. Commit steps were skipped because this workspace is not currently a Git repository.

## Global Constraints

- Project root: `cv-human-video-matting/`.
- Main entry: `python app.py`.
- Model weights must not be committed; keep them under `weights/`.
- Generated videos must not be committed; keep them under `data/output/`.
- Smoke input video can be generated with `python scripts/create_smoke_video.py`.
- Current machine has GTX 1660 Ti, but the detected PyTorch build is CPU-only; MVP must still run on CPU at low resolution.
- Default production model is `rvm_mobilenetv3`.
- No report/论文 work in this plan.

---

## File Structure

- Modify: `requirements.txt`
  - Add `pytest` so every iteration can have a real test cycle.
- Create: `tests/test_render_compositor.py`
  - Unit tests for alpha normalization and image compositing.
- Create: `tests/test_matting_dummy.py`
  - Unit tests for deterministic dummy matting output.
- Create: `tests/test_pipeline.py`
  - Integration test for short-video processing with dummy runner.
- Create: `src/cv_human_video_matting/render/compositor.py`
  - Pure NumPy/OpenCV compositing helpers.
- Create: `src/cv_human_video_matting/matting/base.py`
  - Shared `MattingResult` dataclass and runner protocol.
- Create: `src/cv_human_video_matting/matting/dummy.py`
  - Cheap deterministic matting runner for tests and pipeline debugging.
- Create: `src/cv_human_video_matting/matting/rvm.py`
  - RVM wrapper, isolated from the rest of the pipeline.
- Create: `src/cv_human_video_matting/matting/factory.py`
  - Runner construction from CLI/config model names.
- Create: `src/cv_human_video_matting/pipeline.py`
  - Video-level frame loop and output rendering.
- Modify: `src/cv_human_video_matting/cli.py`
  - Add `matte` subcommand.
- Modify: `README.md`
  - Document the new command and expected outputs.
- Modify: `weights/README.md`
  - Document RVM weight placement and torch hub fallback.

---

### Task 1: Rendering Primitives

**Files:**
- Create: `src/cv_human_video_matting/render/compositor.py`
- Create: `tests/test_render_compositor.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: OpenCV frames as `np.ndarray` in BGR order.
- Produces:
  - `normalize_alpha(alpha: np.ndarray) -> np.ndarray`
  - `compose_on_color(frame_bgr: np.ndarray, alpha: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray`
  - `alpha_to_bgr(alpha: np.ndarray) -> np.ndarray`
  - `side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Add pytest dependency**

Modify `requirements.txt` to include:

```text
opencv-python>=4.9
numpy>=1.24
Pillow>=10.0
PyYAML>=6.0
gradio>=4.0
torch>=2.3
torchvision>=0.18
tqdm>=4.66
pytest>=8.0
```

- [ ] **Step 2: Write the failing compositor tests**

Create `tests/test_render_compositor.py`:

```python
import numpy as np

from cv_human_video_matting.render.compositor import (
    alpha_to_bgr,
    compose_on_color,
    normalize_alpha,
    side_by_side,
)


def test_normalize_alpha_accepts_uint8_mask():
    alpha = np.array([[0, 127, 255]], dtype=np.uint8)

    normalized = normalize_alpha(alpha)

    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 3, 1)
    np.testing.assert_allclose(normalized[0, :, 0], [0.0, 127 / 255, 1.0], rtol=1e-6)


def test_compose_on_color_blends_bgr_frame_with_alpha():
    frame = np.array([[[10, 20, 30], [100, 150, 200]]], dtype=np.uint8)
    alpha = np.array([[255, 0]], dtype=np.uint8)

    output = compose_on_color(frame, alpha, color_bgr=(0, 255, 0))

    assert output.dtype == np.uint8
    assert output.tolist() == [[[10, 20, 30], [0, 255, 0]]]


def test_alpha_to_bgr_returns_three_channel_visualization():
    alpha = np.array([[0, 255]], dtype=np.uint8)

    output = alpha_to_bgr(alpha)

    assert output.shape == (1, 2, 3)
    assert output.dtype == np.uint8
    assert output.tolist() == [[[0, 0, 0], [255, 255, 255]]]


def test_side_by_side_concatenates_matching_height_frames():
    left = np.zeros((2, 3, 3), dtype=np.uint8)
    right = np.full((2, 3, 3), 255, dtype=np.uint8)

    output = side_by_side(left, right)

    assert output.shape == (2, 6, 3)
    assert output[:, :3].sum() == 0
    assert output[:, 3:].sum() == 2 * 3 * 3 * 255
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
pytest tests/test_render_compositor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cv_human_video_matting.render.compositor'`.

- [ ] **Step 4: Implement compositor helpers**

Create `src/cv_human_video_matting/render/compositor.py`:

```python
"""Alpha matte compositing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim == 3 and alpha.shape[2] == 3:
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    if alpha.ndim == 3 and alpha.shape[2] == 1:
        alpha = alpha[:, :, 0]
    if alpha.ndim != 2:
        raise ValueError(f"Alpha must be HxW or HxWx1, got shape {alpha.shape}")

    if alpha.dtype == np.uint8:
        alpha_f32 = alpha.astype(np.float32) / 255.0
    else:
        alpha_f32 = alpha.astype(np.float32)
    alpha_f32 = np.clip(alpha_f32, 0.0, 1.0)
    return alpha_f32[:, :, None]


def compose_on_color(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    alpha_f32 = normalize_alpha(alpha)
    foreground = frame_bgr.astype(np.float32)
    background = np.empty_like(foreground)
    background[:, :] = np.array(color_bgr, dtype=np.float32)
    composed = foreground * alpha_f32 + background * (1.0 - alpha_f32)
    return np.clip(composed, 0, 255).astype(np.uint8)


def alpha_to_bgr(alpha: np.ndarray) -> np.ndarray:
    alpha_f32 = normalize_alpha(alpha)
    alpha_u8 = np.clip(alpha_f32[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2BGR)


def side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    if left_bgr.shape[0] != right_bgr.shape[0]:
        right_bgr = cv2.resize(
            right_bgr,
            (right_bgr.shape[1], left_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.concatenate([left_bgr, right_bgr], axis=1)
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
pytest tests/test_render_compositor.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/cv_human_video_matting/render/compositor.py tests/test_render_compositor.py
git commit -m "feat: add alpha compositing helpers"
```

---

### Task 2: Matting Interfaces and Dummy Runner

**Files:**
- Create: `src/cv_human_video_matting/matting/base.py`
- Create: `src/cv_human_video_matting/matting/dummy.py`
- Create: `tests/test_matting_dummy.py`

**Interfaces:**
- Consumes: BGR frame `np.ndarray`.
- Produces:
  - `MattingResult(alpha: np.ndarray, foreground: np.ndarray | None = None)`
  - `MattingRunner.predict(frame_bgr: np.ndarray) -> MattingResult`
  - `DummyMattingRunner.predict(frame_bgr: np.ndarray) -> MattingResult`

- [ ] **Step 1: Write the failing dummy runner tests**

Create `tests/test_matting_dummy.py`:

```python
import numpy as np

from cv_human_video_matting.matting.dummy import DummyMattingRunner


def test_dummy_runner_returns_center_ellipse_alpha():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    runner = DummyMattingRunner()

    result = runner.predict(frame)

    assert result.alpha.shape == (20, 30)
    assert result.alpha.dtype == np.uint8
    assert result.alpha[10, 15] == 255
    assert result.alpha[0, 0] == 0
    assert result.foreground is None


def test_dummy_runner_output_matches_input_frame_size():
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    runner = DummyMattingRunner()

    result = runner.predict(frame)

    assert result.alpha.shape == frame.shape[:2]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_matting_dummy.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cv_human_video_matting.matting.dummy'`.

- [ ] **Step 3: Implement matting base types**

Create `src/cv_human_video_matting/matting/base.py`:

```python
"""Shared interfaces for matting runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MattingResult:
    alpha: np.ndarray
    foreground: np.ndarray | None = None


class MattingRunner(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        """Return alpha matte and optional foreground for one BGR frame."""
```

- [ ] **Step 4: Implement dummy runner**

Create `src/cv_human_video_matting/matting/dummy.py`:

```python
"""Deterministic matting runner used for tests and pipeline debugging."""

from __future__ import annotations

import cv2
import numpy as np

from cv_human_video_matting.matting.base import MattingResult


class DummyMattingRunner:
    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        height, width = frame_bgr.shape[:2]
        alpha = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (max(width // 5, 1), max(height // 3, 1))
        cv2.ellipse(alpha, center, axes, 0, 0, 360, 255, -1)
        return MattingResult(alpha=alpha, foreground=None)
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
pytest tests/test_matting_dummy.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 6: Commit**

```bash
git add src/cv_human_video_matting/matting/base.py src/cv_human_video_matting/matting/dummy.py tests/test_matting_dummy.py
git commit -m "feat: add matting runner interface"
```

---

### Task 3: Video Matting Pipeline

**Files:**
- Create: `src/cv_human_video_matting/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes:
  - `MattingRunner.predict(frame_bgr: np.ndarray) -> MattingResult`
  - `probe_video(path: str | Path) -> VideoInfo`
- Produces:
  - `PipelineStats(output_path: Path, frames: int, elapsed_seconds: float, fps: float)`
  - `process_video(input_path: str | Path, output_path: str | Path, runner: MattingRunner, mode: str, max_frames: int | None = None) -> PipelineStats`

- [ ] **Step 1: Write the failing pipeline integration test**

Create `tests/test_pipeline.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from cv_human_video_matting.io.video import probe_video
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.pipeline import process_video


def _write_test_video(path: Path, size: tuple[int, int] = (64, 48), frames: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = index * 20
            frame[:, :, 1] = 80
            frame[:, :, 2] = 160
            writer.write(frame)
    finally:
        writer.release()


def test_process_video_writes_alpha_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "alpha.mp4"
    _write_test_video(input_path)

    stats = process_video(
        input_path=input_path,
        output_path=output_path,
        runner=DummyMattingRunner(),
        mode="alpha",
        max_frames=4,
    )

    assert output_path.exists()
    assert stats.output_path == output_path
    assert stats.frames == 4
    output_info = probe_video(output_path)
    assert output_info.width == 64
    assert output_info.height == 48


def test_process_video_writes_side_by_side_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "side.mp4"
    _write_test_video(input_path)

    stats = process_video(
        input_path=input_path,
        output_path=output_path,
        runner=DummyMattingRunner(),
        mode="side_by_side",
        max_frames=3,
    )

    assert stats.frames == 3
    output_info = probe_video(output_path)
    assert output_info.width == 128
    assert output_info.height == 48
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cv_human_video_matting.pipeline'`.

- [ ] **Step 3: Implement process_video**

Create `src/cv_human_video_matting/pipeline.py`:

```python
"""Video-level matting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from cv_human_video_matting.io.video import probe_video
from cv_human_video_matting.matting.base import MattingRunner
from cv_human_video_matting.render.compositor import alpha_to_bgr, compose_on_color, side_by_side
from cv_human_video_matting.utils.paths import ensure_parent_dir


@dataclass(frozen=True)
class PipelineStats:
    output_path: Path
    frames: int
    elapsed_seconds: float
    fps: float


def _render_frame(frame_bgr, alpha, mode: str):
    if mode == "alpha":
        return alpha_to_bgr(alpha)
    if mode == "green":
        return compose_on_color(frame_bgr, alpha, color_bgr=(0, 255, 0))
    if mode == "foreground":
        return compose_on_color(frame_bgr, alpha, color_bgr=(0, 0, 0))
    if mode == "side_by_side":
        return side_by_side(frame_bgr, compose_on_color(frame_bgr, alpha, color_bgr=(0, 255, 0)))
    raise ValueError(f"Unsupported render mode: {mode}")


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    runner: MattingRunner,
    mode: str,
    max_frames: int | None = None,
) -> PipelineStats:
    info = probe_video(input_path)
    output = Path(output_path)
    ensure_parent_dir(output)

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open input video: {info.path}")

    first_ok, first_frame = capture.read()
    if not first_ok:
        capture.release()
        raise ValueError(f"Input video has no readable frames: {info.path}")

    first_result = runner.predict(first_frame)
    first_rendered = _render_frame(first_frame, first_result.alpha, mode)
    out_height, out_width = first_rendered.shape[:2]

    fps = info.fps if info.fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Failed to open output video writer: {output}")

    processed = 0
    started_at = perf_counter()
    try:
        writer.write(first_rendered)
        processed += 1

        while max_frames is None or processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            result = runner.predict(frame)
            rendered = _render_frame(frame, result.alpha, mode)
            writer.write(rendered)
            processed += 1
    finally:
        capture.release()
        writer.release()

    elapsed = perf_counter() - started_at
    effective_fps = processed / elapsed if elapsed > 0 else 0.0
    return PipelineStats(
        output_path=output,
        frames=processed,
        elapsed_seconds=elapsed,
        fps=effective_fps,
    )
```

- [ ] **Step 4: Run pipeline tests**

Run:

```bash
pytest tests/test_pipeline.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Run all tests from Tasks 1-3**

Run:

```bash
pytest tests/test_render_compositor.py tests/test_matting_dummy.py tests/test_pipeline.py -v
```

Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add src/cv_human_video_matting/pipeline.py tests/test_pipeline.py
git commit -m "feat: add video matting pipeline"
```

---

### Task 4: RVM Runner Wrapper

**Files:**
- Create: `src/cv_human_video_matting/matting/rvm.py`
- Create: `src/cv_human_video_matting/matting/factory.py`
- Create: `tests/test_matting_factory.py`
- Modify: `weights/README.md`

**Interfaces:**
- Consumes:
  - `DummyMattingRunner.predict(frame_bgr: np.ndarray) -> MattingResult`
  - PyTorch model compatible with RVM output tuple `(fgr, pha, *rec)`
- Produces:
  - `RVMRunner(model_name: str = "mobilenetv3", checkpoint: str | Path | None = None, device: str = "cpu")`
  - `RVMRunner.predict(frame_bgr: np.ndarray) -> MattingResult`
  - `create_matting_runner(model: str, checkpoint: str | Path | None = None, device: str = "cpu") -> MattingRunner`

- [ ] **Step 1: Write the failing factory tests**

Create `tests/test_matting_factory.py`:

```python
import numpy as np
import pytest

from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.matting.factory import create_matting_runner


def test_factory_creates_dummy_runner():
    runner = create_matting_runner("dummy")

    assert isinstance(runner, DummyMattingRunner)
    result = runner.predict(np.zeros((16, 16, 3), dtype=np.uint8))
    assert result.alpha.shape == (16, 16)


def test_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported matting model"):
        create_matting_runner("unknown-model")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_matting_factory.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cv_human_video_matting.matting.factory'`.

- [ ] **Step 3: Implement RVM wrapper**

Create `src/cv_human_video_matting/matting/rvm.py`:

```python
"""Robust Video Matting runner wrapper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from cv_human_video_matting.matting.base import MattingResult


class RVMRunner:
    def __init__(
        self,
        model_name: str = "mobilenetv3",
        checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name, pretrained=True)
        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(str(checkpoint), map_location=self.device)
            self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self._rec = [None, None, None, None]

    def predict(self, frame_bgr: np.ndarray) -> MattingResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor, *self._rec, downsample_ratio=None)

        foreground, alpha, *rec = output
        self._rec = rec[:4]

        alpha_np = alpha[0, 0].detach().cpu().numpy()
        alpha_u8 = np.clip(alpha_np * 255.0, 0, 255).astype(np.uint8)

        foreground_np = foreground[0].detach().cpu().permute(1, 2, 0).numpy()
        foreground_u8 = np.clip(foreground_np * 255.0, 0, 255).astype(np.uint8)
        foreground_bgr = cv2.cvtColor(foreground_u8, cv2.COLOR_RGB2BGR)

        return MattingResult(alpha=alpha_u8, foreground=foreground_bgr)
```

- [ ] **Step 4: Implement runner factory**

Create `src/cv_human_video_matting/matting/factory.py`:

```python
"""Matting runner factory."""

from __future__ import annotations

from pathlib import Path

from cv_human_video_matting.matting.base import MattingRunner
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.matting.rvm import RVMRunner


def create_matting_runner(
    model: str,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> MattingRunner:
    normalized = model.lower()
    if normalized == "dummy":
        return DummyMattingRunner()
    if normalized in {"rvm", "rvm_mobilenetv3", "mobilenetv3"}:
        return RVMRunner(model_name="mobilenetv3", checkpoint=checkpoint, device=device)
    if normalized in {"rvm_resnet50", "resnet50"}:
        return RVMRunner(model_name="resnet50", checkpoint=checkpoint, device=device)
    raise ValueError(f"Unsupported matting model: {model}")
```

- [ ] **Step 5: Update weights README**

Modify `weights/README.md`:

```markdown
# Weights

Place model weights in this directory.

Do not commit model files to Git. Add download links and checksum notes here when a model is introduced.

## RVM

The MVP runner uses `torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", pretrained=True)` by default. This may download model code and weights into the local Torch cache on first use.

If a local checkpoint is used later, place it here:

```text
weights/rvm_mobilenetv3.pth
```

Then pass it through the CLI:

```bash
python app.py matte --model rvm_mobilenetv3 --checkpoint weights/rvm_mobilenetv3.pth --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 24
```
```

- [ ] **Step 6: Run factory tests**

Run:

```bash
pytest tests/test_matting_factory.py -v
```

Expected: PASS, 2 tests. These tests must not instantiate RVM.

- [ ] **Step 7: Commit**

```bash
git add src/cv_human_video_matting/matting/rvm.py src/cv_human_video_matting/matting/factory.py tests/test_matting_factory.py weights/README.md
git commit -m "feat: add RVM runner wrapper"
```

---

### Task 5: CLI Matting Command

**Files:**
- Modify: `src/cv_human_video_matting/cli.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes:
  - `create_matting_runner(model: str, checkpoint: str | Path | None = None, device: str = "cpu") -> MattingRunner`
  - `process_video(input_path, output_path, runner, mode, max_frames) -> PipelineStats`
- Produces:
  - CLI command:
    - `python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_green.mp4 --mode green --max-frames 24`
    - `mode` choices: `alpha`, `green`, `foreground`, `side_by_side`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from cv_human_video_matting.cli import main
from cv_human_video_matting.io.video import probe_video


def _write_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(4):
            frame = np.full((24, 32, 3), 120, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_matte_command_with_dummy_runner_writes_output(tmp_path, capsys):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "out.mp4"
    _write_test_video(input_path)

    main(
        [
            "matte",
            "--model",
            "dummy",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            "green",
            "--max-frames",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert "Matting output:" in captured.out
    assert output_path.exists()
    assert probe_video(output_path).frame_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL because `matte` command does not exist.

- [ ] **Step 3: Add `matte` parser**

Modify `src/cv_human_video_matting/cli.py` imports:

```python
from cv_human_video_matting.io.video import copy_video, probe_video
from cv_human_video_matting.matting.factory import create_matting_runner
from cv_human_video_matting.pipeline import process_video
from cv_human_video_matting.utils.config import load_config
```

Add this parser block inside `build_parser()` after the `copy_parser` block:

```python
    matte_parser = subparsers.add_parser("matte", help="Run video matting on an input video.")
    matte_parser.add_argument("--input", required=True, help="Input video path.")
    matte_parser.add_argument("--output", required=True, help="Output video path.")
    matte_parser.add_argument(
        "--model",
        default="dummy",
        choices=["dummy", "rvm", "rvm_mobilenetv3", "rvm_resnet50"],
        help="Matting model to use.",
    )
    matte_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional model checkpoint path.",
    )
    matte_parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, such as cpu or cuda.",
    )
    matte_parser.add_argument(
        "--mode",
        default="green",
        choices=["alpha", "green", "foreground", "side_by_side"],
        help="Output rendering mode.",
    )
    matte_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
```

- [ ] **Step 4: Wire `matte` command**

Add this branch inside `main()` after the `copy` branch:

```python
    if args.command == "matte":
        runner = create_matting_runner(
            model=args.model,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        stats = process_video(
            input_path=args.input,
            output_path=args.output,
            runner=runner,
            mode=args.mode,
            max_frames=args.max_frames,
        )
        print(f"Matting output: {stats.output_path}")
        print(
            f"Processed {stats.frames} frame(s) in {stats.elapsed_seconds:.3f}s "
            f"({stats.fps:.2f} FPS)."
        )
        return
```

- [ ] **Step 5: Run CLI test**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS, 1 test.

- [ ] **Step 6: Run full test suite**

Run:

```bash
pytest -v
```

Expected: PASS, 11 tests.

- [ ] **Step 7: Run manual smoke command**

Run:

```bash
python scripts/create_smoke_video.py
python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_side_by_side.mp4 --mode side_by_side --max-frames 24
```

Expected output contains:

```text
Matting output: data\output\dummy_side_by_side.mp4
Processed 24 frame(s)
```

- [ ] **Step 8: Update README**

Add this section after the copy command in `README.md`:

```markdown
Run the MVP matting pipeline with the deterministic dummy runner:

```bash
python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_side_by_side.mp4 --mode side_by_side --max-frames 24
```

After RVM dependencies and weights are available, run:

```bash
python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 24 --device cpu
```
```

- [ ] **Step 9: Commit**

```bash
git add src/cv_human_video_matting/cli.py tests/test_cli.py README.md
git commit -m "feat: add matting CLI command"
```

---

### Task 6: Manual RVM Smoke Verification

**Files:**
- Modify: `README.md`
- Modify: `Obsidian-CV-Matting/任务看板.md`

**Interfaces:**
- Consumes:
  - CLI command from Task 5.
  - RVM runner from Task 4.
- Produces:
  - Verified command line for CPU RVM smoke test.
  - Note if local PyTorch remains CPU-only.

- [ ] **Step 1: Run the dummy full path first**

Run:

```bash
python scripts/create_smoke_video.py
python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_green.mp4 --mode green --max-frames 24
```

Expected: output file exists at `data/output/dummy_green.mp4`.

- [ ] **Step 2: Run RVM CPU smoke test**

Run:

```bash
python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 12 --device cpu
```

Expected success condition:

```text
Matting output: data\output\rvm_green.mp4
Processed 12 frame(s)
```

Expected known failure condition if network/model download is blocked:

```text
urllib.error.URLError
```

or:

```text
RuntimeError: It looks like there is no internet connection
```

If this fails due to network restrictions, document the failure exactly and continue with the dummy runner path as the verified local MVP.

- [ ] **Step 3: Update README with verification result**

If RVM succeeds, add:

```markdown
## RVM Smoke Test

Verified on CPU with:

```bash
python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 12 --device cpu
```
```

If RVM fails due to network/model availability, add:

```markdown
## RVM Smoke Test

The pipeline is verified with the deterministic dummy runner. RVM loading is implemented, but first-run model download was blocked in the current environment. Place RVM weights under `weights/` or run once in a network-enabled environment before using:

```bash
python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 12 --device cpu
```
```

- [ ] **Step 4: Update Obsidian task board**

Modify `Obsidian-CV-Matting/任务看板.md`:

```markdown
- [x] 跑通 RVM 单视频推理。
```

If RVM model download is blocked, use:

```markdown
- [ ] 跑通 RVM 单视频推理。
- [x] 跑通可替换 runner 的视频 matting pipeline。
```

- [ ] **Step 5: Commit**

```bash
git add README.md ../Obsidian-CV-Matting/任务看板.md
git commit -m "docs: record RVM smoke verification"
```

---

## Self-Review

**1. Spec coverage**

- Video input and output: covered by Task 3 and Task 5.
- Alpha matte visualization: covered by Task 1, Task 3, and Task 5 `mode="alpha"`.
- Foreground/background output: covered by Task 1 and Task 5 `mode="foreground"` / `mode="green"`.
- RVM as MVP model: covered by Task 4 and Task 6.
- CPU-safe local development: covered by Task 2 dummy runner and Task 5 dummy CLI.
- Git-friendly workflow: every task ends with a focused commit command.
- No report/论文 work: no task writes report content.

**2. Placeholder scan**

No prohibited placeholder phrases remain. Each code-changing step includes concrete code.

**3. Type consistency**

- `MattingResult.alpha` is always `np.ndarray`.
- `MattingRunner.predict(frame_bgr: np.ndarray) -> MattingResult` is used consistently by `DummyMattingRunner`, `RVMRunner`, and `process_video`.
- `process_video(...) -> PipelineStats` is consumed consistently by CLI.
- Render modes are consistently `alpha`, `green`, `foreground`, `side_by_side`.
