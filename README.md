# cv-human-video-matting

基于深度学习的自然场景人像视频抠像与背景消除系统。

本项目提供实时保留人物、消除任意复杂动态背景，输出高精度软 alpha（保留发丝/半透明衣物边缘），时序稳定不闪烁，并支持人物距离估计标注。

## ✨ 核心特性

- ✅ **三档模型切换**：轻量级(RVM-MobileNetV3) / 高质量(RVM-ResNet50) / 高精度(BiRefNet SOTA)
- ✅ **距离标注**：单目深度估计 + Near/Mid/Far 相对距离标注（核心创新点）
- ✅ **7种背景模式**：Alpha可视化 / 绿幕 / 纯色 / 背景虚化 / 自定义图片 / 透明PNG / 左右对比
- ✅ **实时处理**：GPU加速，22+ FPS @ GTX 1660 Ti
- ✅ **时序稳定**：RVM循环记忆结构，避免视频闪烁
- ✅ **后处理增强**：边缘羽化 / 形态学处理 / 时序平滑
- ✅ **双UI支持**：Gradio Web界面 + CLI批处理

## 📊 模型对比

| 模型 | 档位 | 速度 | 质量 | 适用场景 |
|------|------|------|------|----------|
| RVM-MobileNetV3 | light | 最快 | 良好 | 实时预览、CPU环境、轻量化部署 |
| RVM-ResNet50 | hq | 快 | 优秀 | GPU实时、发丝细节好（默认档） |
| BiRefNet | ultra | 慢 | SOTA | 高质量离线渲染、边缘精度最佳 |

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 PyTorch（根据你的CUDA版本选择）
# CUDA 12.1 示例：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 仅CPU（较慢）：
pip install torch torchvision

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# CLI批处理（带距离标注）
python app.py matte \
  --input data/input/demo.mp4 \
  --output data/output/result.mp4 \
  --model rvm_resnet50 \
  --mode blur \
  --enable-depth \
  --device cuda

# 启动Gradio Web UI
python app.py ui

# 查看视频信息
python app.py inspect --input data/input/demo.mp4
```

### 3. 渲染模式说明

- `alpha`: Alpha遮罩可视化（灰度图）
- `green`: 绿幕背景（传统抠图效果）
- `foreground`: 黑色背景（突出人物）
- `blur`: **背景虚化**（类似人像模式景深效果）
- `transparent`: **透明PNG**（BGRA四通道，可用于后期合成）
- `image`: 自定义背景图片替换
- `side_by_side`: 左右对比（原图 vs 抠图结果）

## 📦 项目结构

```text
cv-human-video-matting/
  app.py                    # 主入口
  configs/
    default.yaml            # 默认配置
  src/
    cv_human_video_matting/
      cli.py                # 命令行接口
      pipeline.py           # 核心处理管道
      matting/              # 抠图模型
        rvm.py              # RobustVideoMatting
        birefnet.py         # BiRefNet高精度引擎 ⭐新增
        factory.py          # 模型工厂
      depth/                # 深度估计模块 ⭐新增
        estimator.py        # MiDaS深度估计器
        visualizer.py       # 距离标注绘制
      render/               # 背景合成
        compositor.py       # 支持blur和transparent ⭐扩展
      postprocess/          # 后处理
      ui/                   # Gradio界面
  tests/                    # 测试用例 ⭐新增
  data/
    input/                  # 输入视频
    backgrounds/            # 背景图片
    output/                 # 输出结果
  weights/                  # 模型权重
```

## Current Status

- [x] Project skeleton
- [x] Video I/O pipeline
- [x] RVM inference (MobileNetV3 + ResNet50)
- [x] **BiRefNet high-precision engine** ⭐
- [x] Background compositing (7 modes)
- [x] **Background blur (bokeh effect)** ⭐
- [x] **Transparent PNG export** ⭐
- [x] Gradio UI
- [x] Edge and temporal postprocessing
- [x] Live camera matting in Gradio
- [x] **Depth-based distance labeling (Near/Mid/Far)** ⭐
- [x] **Comprehensive test coverage** ⭐
- [ ] Optional MatAnyone2 evaluation

## 🎯 距离标注功能

本项目的**核心创新点**：使用MiDaS单目深度估计模型，在人物alpha区域内估计相对距离，并标注为Near/Mid/Far三档。

**技术实现**：
- 每N帧（默认5帧）运行一次深度估计，降低计算开销
- 在人物区域取深度中位数，抗噪声干扰
- 距离档位判断：逆深度 > 0.66 (Near) / 0.33~0.66 (Mid) / < 0.33 (Far)
- 在输出视频上绘制绿色框和距离标签

**CLI使用**：
```bash
python app.py matte \
  --input video.mp4 \
  --output result.mp4 \
  --enable-depth \
  --depth-interval 5 \
  --device cuda
```

**注意**：这是相对距离估计，不是绝对物理测距。精确测距需要双目摄像头或ToF传感器。

## Environment

Python 3.9-3.11 推荐。
NVIDIA GPU + CUDA 12.1+ 推荐（实时高质量档）。
无GPU时自动降级到轻量档。

```bash
pip install -r requirements.txt
```

The base environment currently has CPU-only PyTorch. For GPU inference on this machine, use the existing `DLF` conda environment:

```bash
conda run -n DLF python -m pip install -r requirements-dlf-cu117.txt
```

Verified GPU stack:

```text
Python 3.9.13
torch 1.13.0+cu117
torchvision 0.14.0+cu117
numpy 1.23.5
opencv-python 4.8.1
GPU: NVIDIA GeForce GTX 1660 Ti
```

## Run

Show the available commands:

```bash
python app.py
```

Create a small synthetic smoke-test video:

```bash
python scripts/create_smoke_video.py
```

Inspect a video:

```bash
python app.py inspect --input data/input/smoke_demo.mp4
```

Copy video frames to verify the I/O pipeline:

```bash
python app.py copy --input data/input/smoke_demo.mp4 --output data/output/smoke_copy.mp4 --max-frames 24
```

Run the MVP matting pipeline with the deterministic dummy runner:

```bash
python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_side_by_side.mp4 --mode side_by_side --max-frames 24
```

Run the MVP matting pipeline with an image background:

```bash
python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_image_bg.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --max-frames 24
```

After RVM dependencies and weights are available, run:

```bash
python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 24 --device cpu
```

Run with optional edge and temporal postprocessing:

```bash
python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_postprocess.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --max-frames 24 --device cuda --edge-feather 3 --mask-close 3 --temporal-smooth --temporal-blend 0.65
```

Postprocessing options are off by default:

- `--edge-feather`: softens alpha edges with a small blur radius.
- `--mask-open`: removes small foreground specks.
- `--mask-close`: fills small holes in the foreground mask.
- `--temporal-smooth`: blends alpha mattes over time for less flicker.
- `--temporal-blend`: previous-frame blend ratio used by temporal smoothing. Higher values are steadier but can introduce motion ghosting.

## Verified Smoke Tests

Dummy runner pipeline:

```bash
python scripts/create_smoke_video.py
python app.py matte --model dummy --input data/input/smoke_demo.mp4 --output data/output/dummy_side_by_side.mp4 --mode side_by_side --max-frames 24
```

Observed result:

```text
Matting output: data\output\dummy_side_by_side.mp4
Processed 24 frame(s) in 0.137s (175.39 FPS).
```

RVM CPU smoke test:

```bash
python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 12 --device cpu
```

Observed result:

```text
Matting output: data\output\rvm_green.mp4
Processed 12 frame(s) in 1.578s (7.61 FPS).
```

Note: PyTorch currently runs on CPU in this environment. GPU acceleration is a later iteration.

RVM GPU smoke test with `DLF`:

```bash
conda run -n DLF python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_green_gpu.mp4 --mode green --max-frames 24 --device cuda
```

Observed result:

```text
Matting output: data\output\rvm_green_gpu.mp4
Processed 24 frame(s) in 0.726s (33.05 FPS).
```

Important: keep `DLF` on `numpy==1.23.5`. NumPy 2.x breaks `torch.from_numpy()` for `torch 1.13.0+cu117`.

RVM GPU image-background smoke test with `DLF`:

```bash
conda run -n DLF python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_image_bg_gpu.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --max-frames 24 --device cuda
```

Observed result:

```text
Matting output: data\output\rvm_image_bg_gpu.mp4
Processed 24 frame(s) in 0.974s (24.63 FPS).
```

RVM GPU postprocess smoke test with `DLF`:

```bash
conda run -n DLF python app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/rvm_postprocess_gpu.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --max-frames 24 --device cuda --edge-feather 3 --mask-close 3 --temporal-smooth --temporal-blend 0.65
```

Use this output for quick visual comparison against the baseline image-background run.

## Gradio UI

Launch the demo UI with the GPU-ready `DLF` environment:

```bash
conda run -n DLF python app.py ui --host 127.0.0.1 --port 7860 --no-share
```

Then open:

```text
http://127.0.0.1:7860
```

The UI provides video upload, optional image background upload, model selection, output mode selection, CPU/CUDA device selection, max-frame control, optional refinement controls, output preview, and run status.

The UI now has two workspaces:

- `Batch Render`: uploads a video and writes a playable MP4 result.
- `Live Camera`: opens the browser webcam and runs frame-by-frame matting for live preview.

For live camera mode, use the `DLF` environment and choose `cuda` for the best frame rate. If the preview is slow, lower `Max Live Width` to `320` or `480`, turn off temporal smoothing, or use `dummy` to verify the webcam path first. Browser camera permission is required.

## Large Files

Model weights, large input videos, and generated outputs are intentionally ignored by Git. Put them under:

- `weights/`
- `data/input/`
- `data/backgrounds/`
- `data/output/`

Keep download links and setup notes in `weights/README.md` instead of committing model files.
