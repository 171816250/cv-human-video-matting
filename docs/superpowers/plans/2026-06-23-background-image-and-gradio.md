# Background Image and Gradio UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add image-background replacement and a Gradio demo UI that uses the same tested pipeline as the CLI.

**Architecture:** Keep compositing pure in `render/compositor.py`, keep video processing in `pipeline.py`, expose it through CLI and Gradio without duplicating inference logic. Gradio should be a compact experiment surface, not a landing page.

**Tech Stack:** Python, OpenCV, NumPy, PyTorch, Gradio, pytest.

## Global Constraints

- Project root: `cv-human-video-matting/`.
- GPU execution uses `conda run -n DLF python ...`.
- Keep `DLF` on `numpy==1.23.5` for `torch 1.13.0+cu117`.
- Generated media stays under `data/output/` and is not committed.
- Model weights stay under Torch cache or `weights/` and are not committed.

---

### Task 1: Image Background Compositing

**Files:**
- Modify: `src/cv_human_video_matting/render/compositor.py`
- Modify: `src/cv_human_video_matting/pipeline.py`
- Modify: `tests/test_render_compositor.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Add `compose_on_image(frame_bgr: np.ndarray, alpha: np.ndarray, background_bgr: np.ndarray) -> np.ndarray`.
- Extend `process_video(..., background_path: str | Path | None = None) -> PipelineStats`.
- Add render mode `image`.

**Verification:**
- `pytest tests/test_render_compositor.py tests/test_pipeline.py -v`

### Task 2: CLI Background Image Option

**Files:**
- Modify: `src/cv_human_video_matting/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Add CLI option `--background`.
- Allow `--mode image`.

**Verification:**
- `pytest tests/test_cli.py -v`
- Manual command: `python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_image_bg.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --max-frames 24`

### Task 3: Gradio Demo UI

**Files:**
- Create: `src/cv_human_video_matting/ui/gradio_app.py`
- Modify: `src/cv_human_video_matting/cli.py`
- Modify: `app.py` only if needed.
- Create: `tests/test_gradio_app.py`

**Interfaces:**
- Add `build_demo() -> gradio.Blocks`.
- Add CLI command `ui`.
- Processing callback accepts uploaded video, optional background image, model, mode, max frames, device.

**Verification:**
- `pytest tests/test_gradio_app.py -v`
- `conda run -n DLF python app.py ui --share false`

### Task 4: Docs and Obsidian Update

**Files:**
- Modify: `README.md`
- Modify: `Obsidian-CV-Matting/08-迭代路线图.md`
- Modify: `Obsidian-CV-Matting/任务看板.md`

**Verification:**
- `pytest -v`
- GPU smoke with image background.

