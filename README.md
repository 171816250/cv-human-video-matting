# 自然场景人像视频抠像与背景消除

提供本地网页与命令行入口，支持视频抠像、背景替换、透明PNG序列和摄像头预览。默认使用RVM MobileNetV3；RVM ResNet50、BiRefNet为可选模型，MiDaS用于相对远近标注。

当前处于答辩候选版本准备阶段。工程测试、真实模型运行和真人素材效果评估分别记录，不以短片冒烟测试推导持续实时性能。

## 快速开始

以下命令在仓库根目录运行，建议Python 3.10。已有可用环境先运行检查，无须重新安装：

```bash
python -B app.py doctor --report data/output/environment.json
```

新环境先安装匹配设备的PyTorch与torchvision，再安装演示依赖。CPU示例：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-demo.txt
python -B app.py doctor
```

`requirements-demo.txt`固定关键网页和模型依赖版本，限制NumPy版本；新机器仍需环境检查和试运行。已有环境的实际版本以诊断JSON为准。

首次模型准备需联网，请在答辩前完成：

```bash
python scripts/download_models.py --models rvm_mobilenetv3
python -B app.py ui --host 127.0.0.1 --port 7860 --no-share
```

打开本机 `http://127.0.0.1:7860`。Batch Render上传视频、选择背景模式，输出MP4或透明PNG压缩包。Live Camera需允许浏览器使用摄像头；速度取决于硬件、分辨率和模型。

## 功能与配置

| 能力 | 说明 |
| --- | --- |
| 七种渲染模式 | green纯色、image图片、blur虚化、alpha遮罩、foreground黑底前景、side_by_side对比、transparent透明PNG |
| 视频交付 | H.264/yuv420p MP4，保留原视频首条音轨并转AAC；无音轨视频正常处理 |
| 截取与声音 | --max-frames限制帧数，音频随输出时长截断；--no-preserve-audio去掉声音 |
| 失败保护 | CLI先处理临时文件，成功导出后替换最终MP4；禁止覆盖输入 |
| 后处理 | 边缘羽化、形态学开闭、时间平滑；过强平滑可能拖影 |
| 相对距离 | MiDaS，可配置模型、阈值和刷新间隔；Near/Mid/Far不是米数 |
| 环境检查 | doctor检查依赖、设备、FFmpeg编码器和缓存，不下载模型 |
| 运行记录 | CLI使用--report，网页批处理自动保存同名JSON，记录配置及处理/导出耗时 |

默认配置为 `configs/default.yaml`。额外配置使用 `python app.py --config 文件路径 matte ...`。布尔值写YAML的true/false，颜色写带引号的HEX字符串。显式指定不存在的RVM检查点会报错，未指定时使用预训练缓存。

## 可复现操作

合成素材和dummy仅用于验证处理流程：

```bash
python scripts/create_smoke_video.py
python -B app.py matte --model dummy --device cpu --input data/input/smoke_demo.mp4 --output data/output/dummy.mp4 --mode side_by_side --max-frames 24 --report data/output/dummy.json
```

用已准备的RVM处理经授权的真人素材：

```bash
python -B app.py matte --model rvm_mobilenetv3 --device auto --input data/input/demo.mp4 --output data/output/demo-result.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --report data/output/demo-result.json
```

调整输出尺寸和纯色背景：

```bash
python -B app.py matte --input data/input/demo.mp4 --output data/output/small.mp4 --resize-short 360 --background-color "#336699" --no-preserve-audio
```

`--resize-short`限制短边，仅缩小并取偶数尺寸；`matting.downsample_ratio: auto`只限制RVM内部推理长边到512以内，不单独改变输出尺寸。两者对边缘质量和速度的影响需实测。

## 测试与交付资料

```bash
python -m pip install -r requirements-dev.txt
python -B -m pytest -q --cov=cv_human_video_matting --cov-report=term --cov-report=xml --cov-fail-under=70
```

离线测试使用模型替身与本地合成音视频，不下载权重。GitHub Actions仅在对应平台触发，Gitee托管不代表此工作流已运行。

- [答辩操作与交付清单](docs/DEFENSE_GUIDE.md)
- [申报目标对应清单](docs/INCUBATION_CHECKLIST.md)
- [效果和速度评测方法](docs/EVALUATION.md)
- [接口说明](docs/API.md)、[贡献指南](CONTRIBUTING.md)、[变更记录](CHANGELOG.md)

## 使用边界

透明输出是带alpha的PNG序列，没有音轨，不是透明MP4。深度标注当前对整个人物遮罩生成一个相对标签，尚未实现多人独立测距。BiRefNet为可选实验后端，不能直接声称达到某项质量或速度指标。真人发丝、弱光、快速运动、长视频和摄像头跨浏览器效果仍需留存评测证据。

大文件放在 `weights/`、`data/input/`、`data/backgrounds/`，结果在 `data/output/`，均不进入Git。离线交付需另外准备模型缓存，源码包不等于完整离线环境。

项目许可证为MulanPSL-2.0；外部模型与权重遵守各自许可证。正式发布和社区指标以平台真实记录为准。
