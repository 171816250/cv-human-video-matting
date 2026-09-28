# 独立环境安装与启动

默认交付范围是 **RVM 抠像、本地网页和视频导出**。BiRefNet、MiDaS 和开发测试单独安装。这里采用源码目录启动；`pip install .` 不替代下面的运行依赖与配置准备。

已有全局 Python、Conda 和 DLF 环境保持不动。为本项目新建 `.venv`，以下命令均在仓库根目录运行。项目声明支持 Python 3.9–3.11；新机器优先选择 Python 3.10，实际验收应记录完整 Python 版本。

## 1. 创建独立环境

Windows PowerShell 示例：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install pip==24.3.1
```

如果没有 `py` 启动器，使用已经安装的 Python 3.10 解释器完整路径创建环境。后续始终调用 `.venv` 内的解释器，无须激活环境或修改系统 PATH。macOS/Linux 对应使用 `python3.10 -m venv .venv`，随后将命令中的解释器替换成 `.venv/bin/python`。

## 2. 选择 CPU 或 CUDA 配对

先装 **Torch 2.5.1 + TorchVision 0.20.1**，并使二者来自同一个官方构建源。CPU 适合先验证安装、网页、导出和基本推理：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

需要 NVIDIA GPU 时，按照目标机器驱动和 [PyTorch 官方历史版本配对](https://pytorch.org/get-started/previous-versions/) 选择构建。以下是 CUDA 12.1 的另一种安装选择，与 CPU 命令二选一：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

不要将不同来源的 CPU/CUDA 构建混装。`doctor` 核对安装元数据中的 Torch/TorchVision 配对、可识别的构建标签、模块导入及 NumPy/Torch 转换。GPU 可用性仍需目标设备实测。

## 3. 安装基础网页演示

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B app.py doctor --report data/output/environment.json
```

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 基础 RVM、网页、视频与诊断所需依赖，保留最低要求和兼容范围 |
| `requirements-demo.txt` | 默认交付 profile，固定直接依赖与关键网页配套版本 |
| `requirements-optional.txt` | BiRefNet / MiDaS 可选依赖，须在基础 profile 之上安装 |
| `requirements-dev.txt` | pytest、覆盖率等开发测试工具，不是运行必需项 |
| `requirements-dlf-cu117.txt` | 历史 DLF 参考，不是当前交付 profile 或完整环境快照 |

默认 profile 固定 NumPy 1.26.4、OpenCV 4.10.0.84、Pillow 10.4.0、PyYAML 6.0.2、Gradio 4.37.2、gradio-client 1.0.2、huggingface-hub 0.23.2、FastAPI 0.111.1、Starlette 0.37.2、Pydantic 2.8.2、packaging 25.0 和 imageio-ffmpeg 0.6.0。基础运行不会因为缺少 Transformers、timm、safetensors 或 pytest 而判定失败。

`requirements-demo.txt` 固定直接依赖和关键兼容组合。另提供 `requirements-lock-windows-py310-cpu.txt`，来自本轮新建的 Windows x64 / Python 3.10.16 / CPU 独立环境，冻结全部 81 项运行包；不含测试工具、可选模型或 Python 本身。它不是 CUDA 或其他系统的锁文件，也不是离线安装包。

要复现已验收的 CPU 组合，完成第 2 步的 CPU 安装后，将第 3 步的安装文件替换为完整清单：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py310-cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B app.py doctor --requirements requirements-lock-windows-py310-cpu.txt
```

安装仍需联网或事先准备匹配的 wheel 文件。本机新环境已完成检查与真实 RVM 网页请求，另一台机器仍需独立验收；不要用全局环境的 `pip freeze` 替代项目环境清单。

`doctor` 默认核对 `requirements-demo.txt`。配置启用 BiRefNet 时，追加核对可选 profile 的 Transformers、timm、safetensors；启用 MiDaS 时核对 timm、safetensors，不额外要求 Transformers。程序化调用 `check_environment(requirements_path=...)` 可覆盖基础 profile；已启用功能的可选约束和 Torch/TorchVision 配对仍会检查。

FAIL 项应先解决。WARN 需按提示完成试运行；PASS 表示本地依赖、配置和编码检查通过，不等于模型已缓存、画质达标或性能满足答辩目标。既有环境中测试通过，也不能证明它符合这里的新安装约束。

## 4. 准备模型并启动

首次使用普通 RVM 缓存需要联网准备模型：

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py --models rvm_mobilenetv3
.\.venv\Scripts\python.exe -B app.py ui --host 127.0.0.1 --port 7860 --no-share
```

在浏览器打开终端显示的本地地址，先运行一个短视频，核对画面、声音和随附报告。运行时保持工作目录为仓库根目录，让默认配置、模型与输出路径一致。演示端口被占用时换用其他 `--port`。

已有可校验的 RVM 模型包时，Windows 可双击仓库根目录的 `Start-Demo.cmd`。它使用项目 `.venv`，在 `weights/` 和交付包的 `../evidence/` 中寻找唯一有效模型包，先检查再启动；多个包时须明确选择：

```powershell
.\Start-Demo.cmd --model-bundle weights/offline-rvm-mobilenetv3-rc2 --port 7860
```

也可使用 `.venv` 解释器运行 `scripts/start_demo.py`，参数一致。启动器从任意目录调用均可；显式参数路径按调用目录解释，YAML 内相对路径按项目根目录解释。它不安装依赖、不下载模型，只监听本机，关闭 Gradio 后台统计；离线界面固定使用包内模型并关闭深度标注。关闭启动窗口或按 Ctrl+C 可停止服务。

模型包包含源码和权重，不包含 `.venv`、Python、CUDA 驱动或可选模型。迁移电脑时分别准备依赖环境和模型包，详见[离线模型说明](OFFLINE_MODEL.md)。

在依赖安装和模型准备完成后，先生成无需外部素材的合成人形短片，再自动验证真实网页上传、RVM推理与下载。下面沿用本机已有的模型包路径；新机器请替换为实际模型包位置：

```powershell
.\.venv\Scripts\python.exe scripts/create_smoke_video.py
.\.venv\Scripts\python.exe -B scripts/verify_local_demo.py --bundle weights/offline-rvm-mobilenetv3-rc2 --input data/input/smoke_demo.mp4 --output-dir data/output/web-rehearsal --device cpu --max-frames 8
```

输出目录必须尚不存在。脚本启动自己的本地服务，在空 Torch 缓存和仅允许本机 Python TCP/DNS 连接的条件下运行，检查下载文件、帧数、参数、代码和模型哈希，最后回收服务，保留 `rehearsal.json` 和日志。这不等于操作系统防火墙测试，也不代替真人画质、浏览器播放或摄像头验收。

合成片仅用于工程验收。真人效果验收需团队另行准备经授权的原始视频、记录来源和使用范围，并将 `--input` 替换为该原片；文档中的 `data/input/demo.mp4` 是待准备素材的示例路径，不随源码提供。

## 5. 可选模型与开发测试

确实需要深度标注或 BiRefNet 时再安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt -r requirements-optional.txt
.\.venv\Scripts\python.exe -m pip check
```

可选 profile 暂固定 Transformers 4.41.2、timm 1.0.27、safetensors 0.4.3；这是待对应模型完整验证的兼容候选，不是画质或全部模型变体已验证声明。切换配置后重新运行 `doctor`，并分别准备模型、真实推理和离线试运行证据。MiDaS_small 还会使用 `gen-efficientnet-pytorch` 的源码及骨干权重，只有 MiDaS 主缓存不足以证明离线完整。BiRefNet 使用的模型远程代码与本地缓存需另行验证。

运行工程测试时另装开发工具：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest -q --cov=cv_human_video_matting --cov-report=term --cov-report=xml --cov-fail-under=70
```

测试中使用的模型替身不需要下载权重，也不能代替实际模型效果验收。历史 DLF 环境可保留作旧结果复查；不要为了让旧环境通过检查而降低当前 profile 的 NumPy、OpenCV 或 Pillow 要求。
