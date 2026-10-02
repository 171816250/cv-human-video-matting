# 自然场景人像视频抠像与背景消除

提供本地网页与命令行入口，支持视频抠像、背景替换、透明PNG序列和摄像头预览。默认使用RVM MobileNetV3；RVM ResNet50、BiRefNet为可选模型。相对远近标注沿用MiDaS，并新增实验性的Depth Anything V2 Small本地后端。

当前处于答辩候选版本准备阶段。工程测试、真实模型运行和真人素材效果评估分别记录，不以短片冒烟测试推导持续实时性能。

当前候选版本为 **0.2.0rc4**：新增独立 GPU 实时 WebSocket 工作区、逐帧在途控制、浏览器摄像头/本地视频来源、实时 FPS 与延迟指标。基础批处理、CPU 独立环境、离线模型校验、自适应平滑、无损中间视频、失败回滚与运行报告继续保留。

后续未发布改进增加可选深度模型的本地准备、私人案例证据收集和Gitee Go配置。Depth Anything真实权重已在本机CPU合成帧上跑通；真人效果、三组授权案例、目标服务器速度及Gitee平台构建仍需分别验收。

## 快速开始

完整步骤见[独立环境安装与启动](docs/INSTALLATION.md)。本机已建立 `.venv` 并准备离线模型包，可双击根目录 **Start-Demo.cmd**，检查通过后在浏览器打开显示的本地地址。新电脑仍需先按说明安装环境和准备模型。

以下命令以 Windows PowerShell 为例，在仓库根目录运行，建议 Python 3.10。始终使用项目 `.venv`，无须激活环境；macOS/Linux 将 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`，创建环境使用 `python3.10 -m venv .venv`。已有项目环境先运行检查，无须重新安装：

```powershell
.\.venv\Scripts\python.exe -B app.py doctor --report data/output/environment.json
```

新环境先安装匹配设备的PyTorch与torchvision，再安装演示依赖。CPU示例：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==24.3.1
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -B app.py doctor
```

`requirements-demo.txt`固定关键网页和模型依赖版本；`requirements-lock-windows-py310-cpu.txt`冻结本机新建 Windows / Python 3.10.16 / CPU 环境全部运行包，可替换上方演示依赖文件。新机器仍需环境检查和试运行。已有环境的实际版本以诊断JSON为准。

首次模型准备需联网，请在答辩前完成：

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py --models rvm_mobilenetv3
.\.venv\Scripts\python.exe -B app.py ui --host 127.0.0.1 --port 7860 --no-share
```

打开本机 `http://127.0.0.1:7860`。“视频处理”上传视频、选择背景模式，默认处理全部帧；可设置输出短边、颜色、是否保留声音。“停止当前处理”会停止本会话正在处理的任务；模型加载或单帧推理需等待当前步骤返回。“摄像头预览”需允许浏览器使用摄像头，速度取决于硬件、分辨率和模型。

需要连续摄像头预览时，按[实时预览说明](docs/REALTIME.md)准备 `.venv-gpu`，运行 `Start-Realtime.cmd`；它与原来的 Gradio 页面分开，默认只监听本机。当前 GTX 1660 Ti 的实测建议是 320 宽约 30 FPS、640 宽约 15–18 FPS；这是合成素材服务基准，不是真人质量或远程网络承诺。

## 功能与配置

| 能力 | 说明 |
| --- | --- |
| 七种渲染模式 | green纯色、image图片、blur虚化、alpha遮罩、foreground黑底前景、side_by_side对比、transparent透明PNG |
| 视频交付 | FFV1无损中间文件，再导出H.264/yuv420p MP4；保留原视频首条音轨并转AAC |
| 截取与声音 | --max-frames限制帧数，音频随输出时长截断；--no-preserve-audio去掉声音 |
| 失败保护 | 建模前预检参数与输入；CLI先生成媒体与报告再发布，常规失败回滚旧结果；网页按完整任务目录发布 |
| 后处理 | 边缘羽化、形态学开闭、EMA/自适应时间平滑；默认关闭，自适应减少变化边缘的历史残留 |
| 相对距离 | MiDaS或实验性的Depth Anything V2 Small；可配置本地深度目录、阈值和刷新间隔；Near/Mid/Far不是米数 |
| 环境检查 | doctor核对依赖版本要求、设备、编码器、FFV1实际写读和离线包/缓存，不下载模型 |
| 离线模型 | 从已有缓存导出RVM源码、权重与LICENSE，加载前逐文件校验；运行报告记录内容身份 |
| 运行记录 | CLI使用--report；网页可下载result.json，记录版本、源码/文件SHA256、实际输出尺寸及处理/导出耗时 |
| 案例证据 | 从已有运行报告采集同帧原图/结果截图、参数和哈希，保存到仓库外的私有目录；不生成质量评分 |

网页任务存放在 `data/output/gradio_时间_标识_模式/`，视频为 `result.mp4`，报告为 `result.json`；透明模式包含帧目录与ZIP。`paths.output_dir`可更改保存位置。网页推理任务依次运行以减少显存争用；停止按钮针对已开始的当前任务，排队任务暂不能撤回。

默认配置为 `configs/default.yaml`。额外配置使用 `.\.venv\Scripts\python.exe app.py --config 文件路径 matte ...`。布尔值写YAML的true/false，颜色写带引号的HEX字符串。显式指定不存在的RVM检查点会报错；没有检查点或模型包时使用预训练缓存。

只有`auto`会自动选择CPU；显式指定`cuda`但GPU不可用时会提示错误。源视频已知帧数却提前解码结束时不发布截短结果；元数据帧数未知时允许读到末尾，但报告明确标记完整性未核验。视频按输入名义帧率写出，变帧率素材应先统一帧率并检查音画同步。

## 离线演示与时间平滑

已有RVM缓存时，可生成独立模型包，并在空缓存、Python网络请求拦截下用合成素材试运行。下文模型包统一以 `weights/offline-rvm-mobilenetv3-rc2` 为例；实际路径不同请替换。已有该模型包则跳过导出命令，勿覆盖已有包：

```powershell
.\.venv\Scripts\python.exe -B scripts/prepare_offline_model.py --output weights/offline-rvm-mobilenetv3-rc2
.\.venv\Scripts\python.exe scripts/create_smoke_video.py
.\.venv\Scripts\python.exe -B scripts/rehearse_offline.py --bundle weights/offline-rvm-mobilenetv3-rc2 --input data/input/smoke_demo.mp4 --output-dir data/output/offline-rehearsal --device cpu --max-frames 8
```

网页可用 `.\.venv\Scripts\python.exe -B app.py ui --model-bundle weights/offline-rvm-mobilenetv3-rc2 --no-share` 指定模型包，也可在配置中设置 `matting.bundle`；CLI同样支持。离线界面固定包内抠像模型；未配置独立的Depth Anything本地目录时，深度标注仍关闭。RVM包不包含运行环境、CUDA驱动或任何深度模型。导出和试运行目录必须是新目录；详细步骤见[离线部署说明](docs/OFFLINE_MODEL.md)及[可选深度模型说明](docs/DEPTH_ANYTHING.md)。

`scripts/verify_local_demo.py` 可自动启动真实本地网页，上传样例并下载、校验视频和报告，最终回收服务。用法及网络限制见[安装与验收说明](docs/INSTALLATION.md)。

网页的时间平滑默认关闭。开启时可选择“自适应”或传统EMA；CLI使用`--temporal-smooth --temporal-mode adaptive --temporal-motion-threshold 0.15`。阈值按0–1归一化alpha变化量计算；较小值更快放弃历史，也可能减少降噪作用。这是局部alpha变化启发式，真人发丝、遮挡和快速动作仍需同素材评估。

无损中间文件通常大于最终MP4，需预留临时磁盘空间。最终H.264仍为有损编码；此次改进消除的是中间一步的额外有损压缩。

## 可复现操作

合成素材和dummy仅用于验证处理流程：

```powershell
.\.venv\Scripts\python.exe scripts/create_smoke_video.py
.\.venv\Scripts\python.exe -B app.py matte --model dummy --device cpu --input data/input/smoke_demo.mp4 --output data/output/dummy.mp4 --mode side_by_side --max-frames 24 --report data/output/dummy.json
```

用已准备的RVM模型包处理经授权的真人素材。`data/input/demo.mp4` 不随源码提供，需团队另行准备原片并记录使用授权；`smoke_bg.jpg` 由上面的合成素材脚本生成：

```powershell
.\.venv\Scripts\python.exe -B app.py matte --model rvm_mobilenetv3 --model-bundle weights/offline-rvm-mobilenetv3-rc2 --device auto --input data/input/demo.mp4 --output data/output/demo-result.mp4 --mode image --background data/backgrounds/smoke_bg.jpg --report data/output/demo-result.json
```

调整输出尺寸和纯色背景：

```powershell
.\.venv\Scripts\python.exe -B app.py matte --model-bundle weights/offline-rvm-mobilenetv3-rc2 --input data/input/demo.mp4 --output data/output/small.mp4 --resize-short 360 --background-color "#336699" --no-preserve-audio
```

`--resize-short`限制短边，仅缩小并取偶数尺寸；`matting.downsample_ratio: auto`只限制RVM内部推理长边到512以内，不单独改变输出尺寸。两者对边缘质量和速度的影响需实测。

可选Depth Anything V2 Small仅从已准备的本地目录加载。先按[准备与验收步骤](docs/DEPTH_ANYTHING.md)安装可选依赖、准备并校验资源，再使用 `--enable-depth --depth-model DepthAnythingV2_small --depth-checkpoint weights/depth-anything-v2-small-hf`。本机真实权重合成帧烟测见[验证记录](docs/VALIDATION_DEPTH_ANYTHING_2026-10-02.md)。这条路径适用于CLI和Gradio；独立WebSocket实时工作区尚未接入深度标注。

## 测试与交付资料

同一素材和参数至少运行三次，分别保存报告后生成可离线打开的中文汇总页：

```powershell
.\.venv\Scripts\python.exe -B app.py summarize --reports "data/output/benchmark/run-*.json" --output data/output/benchmark/summary.html
```

汇总按报告记录的输入哈希、代码版本、设备、参数和处理帧数分组，展示中位数与范围；旧版或信息不足的报告单独列出。默认缓存权重仍需团队固定并另行核验，汇总不产生画质评分。

三组教学案例按[真实试用与案例证据流程](docs/CASE_EVIDENCE.md)准备授权原片，再用 `scripts/collect_case_evidence.py collect` 收集同帧截图与清单、用 `verify` 复核。工具要求已有授权记录和运行报告，拒绝把案例目录写进仓库；截图、原片路径和授权材料应保持私有。它不会验证授权法律效力，也不会替团队作人工画质评价。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest -q --cov=cv_human_video_matting --cov-report=term --cov-report=xml --cov-fail-under=70
```

在代码已提交且无已跟踪文件修改时，生成本地候选包：

```powershell
.\.venv\Scripts\python.exe scripts/package_release.py --output-dir ../deliverables/v0.2.0rc4 --evidence data/output/v0.2.0rc4 --evidence weights/offline-rvm-mobilenetv3-rc2
```

目标目录必须尚不存在。工具从Git提交打包源码，附明确指定的证据、可恢复的Git历史、SHA256清单与使用说明；不会自动收录未跟踪文件、模型权重或整个运行环境。候选包不等于远端正式Release。

单独打包时可将`--evidence weights/offline-rvm-mobilenetv3-rc2`加入命令；模型包文件会逐一纳入外置清单，同时保留包内原始LICENSE和清单。

离线测试使用模型替身与本地合成音视频，不下载权重。GitHub Actions仅在对应平台触发；另已准备 [Gitee Go配置](.workflow/python-ci.yml)及[开通说明](docs/GITEE_CI.md)，配置目标是完整离线测试集和70%覆盖率门槛。Gitee页面开通、配置解析和实际构建尚未验证，仓库托管、本地测试通过或存在YAML都不代表平台流水线已运行。

- [答辩操作与交付清单](docs/DEFENSE_GUIDE.md)
- [本版交付说明](docs/RELEASE_v0.2.0rc4.md)、[本版验证记录](docs/VALIDATION_v0.2.0rc4.md)
- [上一版独立环境记录](docs/RELEASE_v0.2.0rc3.md)、[上一版验证记录](docs/VALIDATION_v0.2.0rc3.md)
- [实时预览说明](docs/REALTIME.md)、[实时验证记录](docs/VALIDATION_v0.2.0rc4.md)
- [申报目标对应清单](docs/INCUBATION_CHECKLIST.md)
- [Depth Anything V2 Small本地准备](docs/DEPTH_ANYTHING.md)、[私人案例证据](docs/CASE_EVIDENCE.md)、[Gitee Go状态与开通](docs/GITEE_CI.md)
- [效果和速度评测方法](docs/EVALUATION.md)
- [接口说明](docs/API.md)、[贡献指南](CONTRIBUTING.md)、[变更记录](CHANGELOG.md)

## 使用边界

透明输出是带alpha的PNG序列，没有音轨，不是透明MP4。深度标注当前对整个人物遮罩生成一个相对标签，尚未实现多人独立测距。BiRefNet与Depth Anything V2 Small属于可选实验后端，不能直接声称达到某项质量或速度指标。真人发丝、弱光、快速运动、长视频和摄像头跨浏览器效果仍需留存评测证据。

大文件放在 `weights/`、`data/input/`、`data/backgrounds/`，结果在 `data/output/`，均不进入Git。离线交付需另外准备模型缓存，源码包不等于完整离线环境。

项目许可证为MulanPSL-2.0；外部模型与权重遵守各自许可证。正式发布和社区指标以平台真实记录为准。
