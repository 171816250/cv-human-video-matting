# 参与贡献

本项目接受功能修复、测试、文档和教学案例贡献。首次参与可先提交问题，说明使用场景、预期结果和可复现步骤，再通过 Pull Request 提交变更。

## 本地验证

先按 README 安装运行依赖，再执行：

```bash
python -m pip install -r requirements-dev.txt
python -B -m pytest -q
python -B -m pytest -q --cov=cv_human_video_matting --cov-report=term-missing --cov-report=html --cov-fail-under=70
```

默认测试使用确定性替身验证模型适配和数据处理，不下载真实模型。真实模型效果和速度请按 `docs/EVALUATION.md` 单独验证；测试通过不代表抠像质量达标。

命令中的 `-B` 禁止写入Python字节码缓存，避免只读共享环境中的旧版pytest缓存写入重试问题。

## 提交要求

1. 从稳定分支创建功能分支，一个 PR 解决一个明确问题。
2. 说明问题、变更后的行为、复现步骤和实际验证结果。
3. 修改输出、模型接口或运行参数时，同步更新教程和必要的回归测试。
4. 不提交模型权重、个人视频、运行日志、账号信息或申报书中的个人资料。
5. 新模型通过 `MattingRunner.predict` 返回 `MattingResult`；有跨帧状态的模型提供 `reset()`，每段视频开始时清空状态。

维护者评审后合并，持续集成通过后再准备版本发布。代码遵循仓库 MulanPSL-2.0；第三方代码和模型保留各自的授权与来源说明。
