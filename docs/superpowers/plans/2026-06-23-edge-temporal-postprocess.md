# Edge and Temporal Postprocess Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional alpha matte edge refinement and temporal smoothing so the demo can compare baseline output against more stable, presentation-ready output.

**Architecture:** Keep postprocessing model-agnostic under `postprocess/`, run it inside `process_video()` before rendering, and expose the same switches through CLI and Gradio. Defaults must preserve the existing RVM baseline.

**Tech Stack:** Python, NumPy, OpenCV, Gradio, pytest, optional CUDA through the existing `DLF` conda environment.

## Global Constraints

- Project root: `cv-human-video-matting/`.
- Do not change default pipeline behavior unless a postprocess option is explicitly enabled.
- Generated media stays under `data/output/` and is not committed.
- GPU execution uses `conda run -n DLF python ...`.
- Keep `DLF` on `numpy==1.23.5` for `torch 1.13.0+cu117`.
- Temporal smoothing is a demo/ablation option; document that it can introduce motion ghosting.

---

### Task 1: Alpha Edge Refinement Utilities

**Files:**
- Create: `src/cv_human_video_matting/postprocess/alpha_refine.py`
- Modify: `src/cv_human_video_matting/postprocess/__init__.py`
- Create: `tests/test_alpha_refine.py`

**Interfaces:**
- `feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray`
- `clean_alpha(alpha: np.ndarray, open_radius: int = 0, close_radius: int = 0) -> np.ndarray`
- `refine_alpha(alpha: np.ndarray, feather_radius: int = 0, open_radius: int = 0, close_radius: int = 0) -> np.ndarray`

**Verification:**
- First run `pytest tests/test_alpha_refine.py -v` and verify it fails because the module does not exist.
- Implement the utilities.
- Re-run `pytest tests/test_alpha_refine.py -v` and verify it passes.

### Task 2: Temporal Alpha Smoothing

**Files:**
- Create: `src/cv_human_video_matting/postprocess/temporal.py`
- Modify: `src/cv_human_video_matting/postprocess/__init__.py`
- Create: `tests/test_temporal_postprocess.py`

**Interfaces:**
- `TemporalAlphaSmoother(blend: float = 0.75)`
- `TemporalAlphaSmoother.apply(alpha: np.ndarray) -> np.ndarray`
- `TemporalAlphaSmoother.reset() -> None`

**Verification:**
- First run `pytest tests/test_temporal_postprocess.py -v` and verify it fails because the module does not exist.
- Implement the stateful smoother with uint8 output and range clipping.
- Re-run `pytest tests/test_temporal_postprocess.py -v` and verify it passes.

### Task 3: Pipeline and CLI Integration

**Files:**
- Modify: `src/cv_human_video_matting/pipeline.py`
- Modify: `src/cv_human_video_matting/cli.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Extend `process_video(...)` with:
  - `edge_feather: int = 0`
  - `mask_open: int = 0`
  - `mask_close: int = 0`
  - `temporal_smooth: bool = False`
  - `temporal_blend: float = 0.75`
- Extend `matte` CLI with:
  - `--edge-feather`
  - `--mask-open`
  - `--mask-close`
  - `--temporal-smooth`
  - `--temporal-blend`

**Verification:**
- First add tests for enabled flags and run targeted tests to verify failure.
- Implement pipeline and CLI wiring.
- Run `pytest tests/test_pipeline.py tests/test_cli.py -v`.

### Task 4: Gradio Controls

**Files:**
- Modify: `src/cv_human_video_matting/ui/gradio_app.py`
- Modify: `tests/test_gradio_app.py`

**Interfaces:**
- Extend `run_demo(...)` with edge and temporal arguments.
- Add compact controls for edge feather, mask open/close, temporal smoothing, and temporal blend.

**Verification:**
- Run `pytest tests/test_gradio_app.py -v`.

### Task 5: Docs, Obsidian, and Smoke Verification

**Files:**
- Modify: `README.md`
- Modify: `Obsidian-CV-Matting/08-迭代路线图.md`
- Modify: `Obsidian-CV-Matting/任务看板.md`

**Verification:**
- Run `pytest -v`.
- Run `conda run -n DLF python -m pytest -v`.
- Run GPU smoke:

```bash
conda run -n DLF python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_postprocess_gpu.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --max-frames 24 --device cuda --edge-feather 3 --mask-close 3 --temporal-smooth --temporal-blend 0.65
```

Expected: command prints `Processed 24 frame(s)` and output video exists.

## Self-Review

- Spec coverage: edge quality is covered by feather/morphology controls; temporal stability is covered by optional smoothing; demo usability is covered by CLI and Gradio.
- Placeholder scan: no task depends on unspecified code or vague "handle later" steps.
- Type consistency: all postprocess functions consume and return `np.ndarray` alpha masks; pipeline options are simple primitives mirrored by CLI and UI.
