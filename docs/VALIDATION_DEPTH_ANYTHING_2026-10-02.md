# Depth Anything V2 Small 本地验证记录（2026-10-02）

本记录只证明固定官方权重在本机 CPU 上能完成一次合成画面推理和一帧视频处理。没有使用真人素材，也没有验证连续实时帧率、人物前后移动的标签合理性或租用服务器的 GPU 表现。

| 项目 | 记录 |
| --- | --- |
| 环境 | Windows，Python 3.10.16，PyTorch 2.5.1+cpu；Transformers 4.41.2、safetensors 0.4.3、tokenizers 0.19.1、regex 2024.11.6 临时安装在受 Git 忽略的 `weights/depth-anything-deps/`，基础 `.venv` 未修改 |
| 模型 | 官方 `depth-anything/Depth-Anything-V2-Small-hf`，固定 revision `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| 权重 SHA256 | `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`；准备脚本和离线 `--verify-only` 均通过 |
| 环境检查 | 指定本地深度目录后，`doctor` 的 `ready=true`；Transformers、safetensors 和 `depth-checkpoint` 均通过，仍保留实验性深度告警 |
| 直接推理 | 64×96 合成渐变图；返回 64×96 `float32` 深度图，所有值有限，归一化范围为 0–1 |
| 视频链路 | `data/input/smoke_demo.mp4` 合成素材、dummy 抠像、绿色背景、真实深度权重；处理 1 帧，生成 MP4 与 schema 3 JSON 报告 |
| 单次计时 | 视频流水线约 1.541 秒，包含模型初始化；总计约 1.640 秒，含导出。1 帧不能推算持续 FPS |
| 结果身份 | 输出 MP4 SHA256：`1c7d2b2174df3efc8bdd30a0d8d7e020cc38bc943d1a9feec4622a102754b06c`；报告记录了三份深度文件的哈希，模型资源汇总哈希为 `8ee56d6fa7b54755f54a5fd9ecddc795bd0d3612fbde78e9d627d45c8aef5d29` |
| 评测边界 | 报告 `quality_metrics=null`；汇总工具识别该单份报告为可比较的条件记录，但没有画质评分 |

本地忽略目录保留 `weights/depth-anything-v2-small-hf/`、`data/output/depth-anything-smoke-v2-2026-10-02.mp4`、同名 JSON 运行报告和 `data/output/depth-anything-doctor-2026-10-02.json`。这些文件未进入 Git。模型目录可用 `python scripts/prepare_depth_anything.py --verify-only` 独立复核；复制到其他机器时仍须重建依赖环境和试运行。

下一步需要用已获授权的真人镜头记录发丝、遮挡、弱光和前后移动情况；在同一素材与参数下对比 MiDaS 与 Depth Anything，并在实际交付服务器上测量连续视频速度。正式 Gitee CI 构建和三组教学案例也各自需要平台/素材证据。
