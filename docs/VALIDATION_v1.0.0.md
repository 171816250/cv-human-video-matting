# v1.0.0 发布验证记录

验证日期：2026-10-23。本记录区分工程回归、环境检查及真实模型试运行；测试覆盖率不代表抠像画质。

## 工程回归

在 Windows、Python 3.10.16、pytest 8.4.2 的项目 CPU 测试环境运行完整测试集：

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=cv_human_video_matting --cov-report=term-missing --cov-report=xml --cov-report=json --junitxml=pytest-results.xml
```

- 724 项通过，2 项跳过，46 条依赖弃用警告；耗时 58.01 秒。
- 核心包语句覆盖率 91.03%，覆盖 2599/2855 条语句，超过项目 70% 门槛。
- 两项跳过是当前 Windows 环境不能创建目录符号链接，涉及发行包和服务器备份的链接拒绝测试。
- BiRefNet 新增的 7 项离线预测与渲染回归全部通过；真实权重下载、推理和真人画质不在这 7 项测试的范围内。

现有报告与候选打包相关测试另有 61 项通过、1 项符号链接测试跳过。没有为版本号增加只重复实现的测试；项目声明与实际导入版本均核对为 `1.0.0`。

## GPU 运行环境检查

独立 `.venv-gpu` 环境的 `python app.py doctor` 检查全部通过，`pip check` 无依赖冲突：

- Python 3.10.16、PyTorch 2.5.1+cu121、TorchVision 0.20.1+cu121。
- NVIDIA GeForce GTX 1660 Ti；CUDA 设备可用，NumPy 与张量转换通过。
- 演示依赖的 14 个包均满足 `requirements-demo.txt` 及其引用的声明约束。
- FFmpeg 提供 H.264/AAC 编码；FFV1/Matroska 小文件写入后解码像素一致。
- 默认 RVM 源码及权重缓存存在。缓存存在本身不算真实推理测试。

## 冻结源码与交付包

公开源码从最终提交生成 ZIP，排除本地未跟踪文件、模型、运行环境、私人案例与完整 Git 历史。附件的提交号、逐文件 SHA256 和解压后真实 RVM 复验结果以发行验证附件为准；只有本地提交或打标签不能证明远端发行已经创建。

完整本地 Git 恢复包单独保存，不随公开附件分发。运行模型时仍需按上游许可准备对应权重和源码。

历史 GPU 合成帧速度见 [rc4 验证记录](VALIDATION_v0.2.0rc4.md)，Depth Anything 的历史 CPU 合成帧验证见 [2026-10-02 记录](VALIDATION_DEPTH_ANYTHING_2026-10-02.md)。这些结果不能外推到真人发丝质量、远程网络、多人并发或长期稳定性。
