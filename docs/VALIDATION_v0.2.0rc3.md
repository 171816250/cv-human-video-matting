# 0.2.0rc3 验证记录（2026-09-28）

本轮在新建项目 `.venv` 中验收基础 RVM、本地网页与视频导出。没有升级原全局环境；不包含真人画质、摄像头、CUDA 新安装或可选深度/BiRefNet 的验收。

## 环境与自动化

| 项目 | 实测结果 |
|---|---|
| 系统 / Python | Windows 10 x64 / Python 3.10.16，独立 `.venv`，不继承系统包 |
| 推理组合 | Torch 2.5.1+cpu、TorchVision 0.20.1+cpu |
| 网页 / 视频 | Gradio 4.37.2、gradio-client 1.0.2、OpenCV 4.10.0.84、imageio-ffmpeg 0.6.0 |
| 数值 / 图像 | NumPy 1.26.4、Pillow 10.4.0 |
| 安装校验 | `pip check` 无依赖冲突；完整 81 项运行清单检查通过 |
| Doctor | 98 项检查均 PASS，`ready=true`；在安装测试工具之前已通过 |
| 工程回归 | **579 passed、1 skipped、39 warnings**，48.54 秒 |
| 核心代码覆盖率 | **89.97%**，2,492 条语句中覆盖 2,242 条；不包含 scripts 目录的覆盖率统计 |

运行清单为 `requirements-lock-windows-py310-cpu.txt`；pip 24.3.1、setuptools 65.5.0 及新增开发测试工具版本另记于 `environment.json` 与 `environment-with-tests.txt`。安装来自官方 PyTorch CPU 索引及 PyPI；新环境未安装 Transformers、timm 或 safetensors。

跳过项为 Windows 无创建符号链接权限时的目录链接防护用例；39 条警告来自 Starlette / Gradio / FastAPI 的弃用提示。工程测试包含真实 HTTP 双会话和失败保护测试，但其中模型替身不代表 RVM 画质。

## 实际启动与完整网页验收

1. 从仓库外目录调用 `Start-Demo.cmd --help`，正确使用项目解释器并解析参数。
2. 从仓库外目录用项目解释器运行 `scripts/start_demo.py`，自动发现唯一有效模型包，经 Doctor 后启动本机服务。实际读取网页配置确认可用、队列开启、后台统计关闭，结束后回收服务。记录为 `starter.json` / `starter.log`；没有实测桌面双击或浏览器摄像头。
3. 使用 `scripts/verify_local_demo.py` 启动真实 Gradio 服务，经 HTTP 上传 **320×180、24 FPS、30 秒、720 帧**的合成运动视频，使用真实 RVM MobileNetV3，在 CPU 上完整处理 720 帧。空 Torch 缓存、关闭分享和统计、Python TCP/DNS 限制为本机；这不是操作系统防火墙测试。
4. 自动下载对比视频及 JSON，核对全部提交参数、文件哈希、模型包与实际权重身份、代码版本/源码哈希、Python 与依赖版本、逐帧解码数量。输出为 **640×180、24 FPS、720 帧、30 秒、H.264/yuv420p + AAC**。
5. 另用 FFmpeg 完整解码视频和音轨，返回成功；解码音频长 **30.016 秒**，非静音。检查的是解码、时长和音轨存在，不代表人工试听、真人唇形同步或浏览器播放验收。

本次流水线耗时 **43.399 秒 / 16.59 FPS**；包含模型加载与导出共 **47.260 秒 / 15.24 FPS**，另有元数据采集 0.268 秒。这是单次合成片记录，低于源视频 24 FPS，不能声称 CPU 持续实时，也不能替代真人素材多轮基准。

启动器及验收脚本只回收自己启动的服务。Windows 下主动结束服务记录的退出码为 1，`service_reaped=true` 表示已完成回收；此退出码不是处理失败。

## 版本、证据与追溯

- 实际运行代码提交：`3cdcc2fc131f4e92c24d2ede8878edfd595d3ed8`。收尾提交仅补充本验证文档，核心源码保持一致。
- 核心源码 SHA256：`761e230761b58347b9fad7ef923303ecd867012b1ac8dfc5e4e0437607c46d69`。
- RVM 模型包内容 SHA256：`dcfca84453ff1c932240ab2aa0bcada6bedc85e93f4b9a60a49555e70a0ea375`。
- 完整输出 MP4 SHA256：`b568f331b2c68161685bb4ead4040d364af0ae4980d169aa0e85d3b92573acc2`。

报告保留 `git_dirty=true`，因为仓库原有 `_git_branch.txt`、`_git_count.txt`、`_git_log.txt`、`_git_status.txt` 四个未跟踪记录。没有删除、移动或伪改这些文件以美化报告。额外检查确认运行时所有已跟踪文件均已提交，33 个核心 Python 文件与 Git 提交逐字节一致；记录见 `source-verification.json`。最终候选包另外逐项校验源码及证据哈希。

证据在工作区 `data/output/v0.2.0rc3/`，打包后位于 `evidence/v0.2.0rc3/`：

| 证据 | 内容 |
|---|---|
| `doctor-runtime.json`、`pip-check.txt` | 安装与运行依赖检查 |
| `install-*.log/json`、`environment*.txt/json` | 安装来源、依赖与测试环境 |
| `pytest.xml/log`、`coverage.xml/json` | 完整自动化记录 |
| `starter.json/log` | 外目录启动器验证 |
| `rehearsal-final-30s/rehearsal.json` | **最终完整网页验收主记录** |
| `rehearsal-final-30s/downloads/` | 经 HTTP 下载的视频和报告 |
| `media-validation.json` | 完整解码与音轨检查 |
| `source-verification.json` | 运行源码与提交的一致性 |
| `rehearsal-runtime/` | 早期 8 帧安装冒烟记录；修改前源码，不作为最终版本证据 |

模型包与合成原片分别附在交付包 `evidence/offline-rvm-mobilenetv3-rc2/`、`evidence/synthetic-30s-with-audio.mp4`。模型包名称中的 rc2 表示沿用同一份已校验资产，不改变本次程序版本 rc3。报告中的本机绝对路径是运行当时的记录，移机复现时应使用交付包内文件的实际位置。

## 待完成的项目验收

本机独立 CPU 环境已验证，但目标答辩电脑尚需重新安装试运行；没有打包 `.venv` 或全部离线安装 wheel。已授权真人原片、标准 alpha、复杂背景/快速动作/弱光案例、持续摄像头和长视频音画同步仍需补齐。BiRefNet、MiDaS 和社区发布指标继续单独记录，不由本轮测试推导。
