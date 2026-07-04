# Project Superpowers

This project is managed as an iterative engineering project first. Report or thesis writing is paused until the runnable system is stable.

## Rules

- Build the smallest runnable slice first.
- Keep every iteration verifiable.
- Keep code modular enough to replace models later.
- Do not commit model weights, large videos, generated outputs, or local paths.
- Update the README whenever the run command changes.
- Prefer a working baseline over a half-integrated advanced model.

## Iterations

1. Repository skeleton.
2. Video input/output pipeline.
3. RVM minimal inference.
4. Background compositing and alpha visualization.
5. Gradio demo.
6. Edge and temporal postprocessing.
7. Relative distance labeling.
8. Optional MatAnyone2 evaluation.

## MVP Acceptance

- A local video can be processed.
- The project outputs a playable result video.
- The alpha matte is saved or visualized.
- The run command is documented.

