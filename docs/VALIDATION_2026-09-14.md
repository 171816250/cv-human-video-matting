# 2026-09-14 本地工程验证

本次按开源孵化申报目标完成处理可靠性、距离标注、实时会话隔离、推理比例配置、运行报告和开源协作基础改进。

## 自动化验证

```bash
python -B -m pytest -q --cov=cv_human_video_matting --cov-report=term --cov-report=json:data/output/coverage.json --cov-fail-under=70
```

结果：105项通过，整体语句覆盖率84.08%（999条语句，159条未覆盖）。
覆盖范围为整个 `src/cv_human_video_matting` 包；没有排除UI或模型适配模块来提升覆盖率。
12条警告来自现有Gradio/FastAPI生命周期接口弃用提示。
模型单元测试使用离线替身，BiRefNet真实推理尚未验证，不能将单元测试通过等同于模型质量通过。

## 实际运行验证

输入：仓库已有的 `data/input/smoke_demo.mp4`，320×180、24 FPS、48帧合成示例。
环境：Windows 10，Python 3.9.12，PyTorch 2.5.1+cu121，NVIDIA GeForce GTX 1660 Ti。

```bash
python -B app.py matte --model rvm_mobilenetv3 --input data/input/smoke_demo.mp4 --output data/output/incubation_rvm_auto.mp4 --device cuda --max-frames 48 --report data/output/incubation_rvm_auto.json
```

真实RVM模型使用本机缓存加载，成功输出48帧视频及JSON报告。视频处理耗时约3.652秒，含首帧的处理吞吐约13.14 FPS。
这是合成短视频的单次流程验证，不是人像边缘质量验收，也不足以证明稳定实时。
早前24帧运行约3.24 FPS，两次帧数和启动条件不同，不能据此宣称性能提升比例。

## 尚未完成的验收项

- 三组以上授权真实人像素材、质量和时序对比，以及重复性能实验。
- 实际摄像头及多浏览器端到端操作验证（本次验证了界面构建和会话回调隔离）。
- MiDaS、BiRefNet在当前环境的真实模型效果评估；Depth Anything适配尚未实施。
- 新环境全新安装验证；新增GitHub Actions需推送后才会实际执行，Gitee流水线仍需平台配置。
- 国内平台公开仓库、两次正式版本、社区贡献指标及正式申报材料。

当前变更尚未创建版本标签或推送远端。工程对照表见INCUBATION_CHECKLIST.md。
