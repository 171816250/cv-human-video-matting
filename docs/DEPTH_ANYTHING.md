# Depth Anything V2 Small 本地深度后端

项目增加了实验性的 `DepthAnythingV2_small` 选项，用独立的本地深度模型为人物遮罩生成 Near/Mid/Far 相对标签。它与 RVM 的前景抠像承担不同工作：RVM 产生 alpha 和前景，深度模型提供相对远近。默认仍关闭深度，已有 MiDaS 路径保留。

**当前已实现本地加载接口、参数贯通和资源准备/校验流程；真实权重已在本机 CPU 的合成帧上跑通。目标机器连续速度和真人场景效果尚未完成验收。** 模型替身测试与单帧烟测都不能据此宣称它优于 MiDaS，实测范围见[验证记录](VALIDATION_DEPTH_ANYTHING_2026-10-02.md)。

## 支持范围

| 项目 | 当前约定 |
|---|---|
| 配置模型名 | `DepthAnythingV2_small`，区分大小写 |
| 模型格式 | 官方 `depth-anything/Depth-Anything-V2-Small-hf` 的 Transformers 格式目录 |
| 必需资源 | `config.json`、`preprocessor_config.json`、真实 `model.safetensors` |
| 运行时加载 | `local_files_only=True`、`trust_remote_code=False`、`use_safetensors=True` |
| 适用入口 | CLI 批处理、Gradio 批处理与摄像头回调 |
| 暂未接入 | 独立 FastAPI/WebSocket 实时工作区；它仍只运行 RVM |

不能把模型仓库名、单独的 `.pth` 文件、Git LFS 指针文件或其他模型尺寸当作本地目录。准备工具限定 V2 Small 的相对深度配置，拒绝依赖另行下载骨干权重的配置。文件完整与哈希正确只证明资源符合清单，不能证明推理效果或运行速度。

## 先准备独立试验环境

不要改动已经验收并冻结的 CPU/GPU 演示环境。以下以新的 Windows / Python 3.10 CPU 试验环境 `.venv-depth` 为例，在仓库根目录运行：

```powershell
py -3.10 -m venv .venv-depth
.\.venv-depth\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-depth\Scripts\python.exe -m pip install -r requirements-demo.txt -r requirements-optional.txt
.\.venv-depth\Scripts\python.exe -m pip check
```

需要 CUDA 时，按[安装说明](INSTALLATION.md)选用对应的 Torch/TorchVision 配对构建，在独立环境中重新验证。当前可选依赖候选固定 Transformers 4.41.2、timm 1.0.27、safetensors 0.4.3；该 profile 同时服务其他可选模型，不表示每个后端已经完成真实权重测试。Depth Anything 的加载路径使用 Transformers 和 safetensors。

## 联网准备、离线校验

准备阶段可以访问官方 Hugging Face 模型仓库；实际处理阶段不会自动补下载。首次准备命令：

```powershell
.\.venv-depth\Scripts\python.exe scripts/prepare_depth_anything.py
```

默认生成 `weights/depth-anything-v2-small-hf/`，其中包含三份必需模型文件和 `manifest.json`。也可使用 `--output weights/自定义目录名`；命令行只允许写入仓库 `weights/` 下的直接子目录，不能指向其他目录。该目录受 Git 忽略规则保护，模型权重不应提交到源码仓库。

工具固定以下来源，并在清单中记录来源、许可标识、文件大小与 SHA256：

| 字段 | 固定值 |
|---|---|
| 仓库 | `depth-anything/Depth-Anything-V2-Small-hf` |
| revision | `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| 权重 SHA256 | `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70` |
| 清单中的许可标识 | `apache-2.0`，并保留对应上游来源链接 |

现有目标目录只执行校验，不会覆盖或补写。下载或校验失败不会发布不完整的新目录。准备工具需要 `huggingface_hub`；仅校验使用 Python 标准库，不导入该库、不联网、不加载模型：

```powershell
.\.venv-depth\Scripts\python.exe scripts/prepare_depth_anything.py --verify-only
```

更换机器时复制完整目录，在目标仓库的 `weights/` 下再次运行校验。模型目录不包含 Python 环境、RVM、CUDA 或显卡驱动。分发时同时核对上游许可和署名要求；项目许可证不替代模型许可。

## CLI 与网页启用

先用已准备的 RVM 模型包和经授权的短视频试运行。下面的 `demo.mp4` 是团队自行准备的原片，不随源码提供：

```powershell
.\.venv-depth\Scripts\python.exe -B app.py matte --input data/input/demo.mp4 --output data/output/depth-anything-demo.mp4 --report data/output/depth-anything-demo.json --model rvm_mobilenetv3 --model-bundle weights/offline-rvm-mobilenetv3-rc2 --device cpu --max-frames 60 --enable-depth --depth-model DepthAnythingV2_small --depth-checkpoint weights/depth-anything-v2-small-hf --depth-interval 5
```

也可另存 `configs/depth-anything-local.yaml`。以下是一个独立配置示例，实际模型包路径不同时需替换：

```yaml
device:
  mode: cpu
matting:
  model: rvm_mobilenetv3
  bundle: weights/offline-rvm-mobilenetv3-rc2
  checkpoint: null
  downsample_ratio: auto
depth:
  enabled: true
  model: DepthAnythingV2_small
  checkpoint: weights/depth-anything-v2-small-hf
  interval: 5
  thresholds:
    near: 0.66
    mid: 0.33
```

检查这份配置后启动 Gradio：

```powershell
.\.venv-depth\Scripts\python.exe -B app.py --config configs/depth-anything-local.yaml doctor --report data/output/depth-anything-environment.json
.\.venv-depth\Scripts\python.exe -B app.py --config configs/depth-anything-local.yaml ui --host 127.0.0.1 --port 7860 --no-share
```

`doctor` 是依赖与资源预检，不执行深度推理。网页从配置读取模型和本地目录，不提供自动下载入口；使用 RVM 离线包时，只有另外配置 Depth Anything 本地目录才允许深度选项。`Start-Demo.cmd` 面向已有基础演示环境，启动时默认关闭深度；可选模型试验使用上面的明确配置和解释器。`Start-Realtime.cmd` 不支持深度标注。

## 解释标签并留存验收证据

输出先按当前画面归一化到 0–1，数值越大表示相对越近；人物标签来自整个有效 alpha 区域的中位数。阈值必须满足 `0 <= mid < near <= 1`。标签不表示米数，不能用于跨场景绝对距离比较，也没有给每个人独立测距。刷新间隔大于一帧时会复用前一次深度结果，快速运动可能出现标签滞后；开启模型也会增加计算量。

真实验收应记录模型资源清单、代码版本、环境、原片来源与授权、参数、输出及失败现象，再比较同素材下的 MiDaS 与 Depth Anything。至少检查模型是否实际加载、输出是否有限且与画面同尺寸、人物前后移动时标签是否合理、弱光和遮挡下是否不稳定。不要从模型名称或单次成功运行推导精度提升。

CLI 报告会记录三份深度模型文件在报告生成时的哈希；该身份记录不单独证明处理期间加载的字节未变化。报告与资源清单应一并保留。可按[案例证据流程](CASE_EVIDENCE.md)采集同帧原图和结果截图并进行私下复核；案例收集不会自动证明授权有效或生成质量分。Gitee 的离线测试使用模型替身，不下载这些真实权重；平台运行状态另见[Gitee Go说明](GITEE_CI.md)。
