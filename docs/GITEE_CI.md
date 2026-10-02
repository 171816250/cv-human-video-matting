# Gitee Go 自动化测试

仓库内的 [`.workflow/python-ci.yml`](../.workflow/python-ci.yml) 是待启用的 Gitee Go 测试流水线配置。它在 Push 和 Pull Request 事件上运行 Python 3.9 的完整 CPU 测试集，要求项目代码覆盖率至少达到 70%；测试命令设置模型仓库离线环境变量，不下载模型权重，也不需要私有令牌。依赖安装使用项目的 `requirements-demo.txt` 和 `requirements-dev.txt`，其中 PyTorch 与 TorchVision 先从官方 CPU 索引安装。

**当前状态：仅已准备仓库配置和本地测试，尚未在 Gitee 页面开通或执行流水线。** 将文件提交到 Gitee 仓库后，还须由有仓库权限的成员在该仓库的「流水线 / Gitee Go」页面开通服务，并用代码视图创建或关联这份 `.workflow/python-ci.yml`。在页面检查 Push、PR 触发设置，手动执行一次，查看构建历史和日志；之后再用一次提交或 PR 验证自动触发。仅存在 YAML 文件不代表 Gitee 已创建流水线对象或构建成功。

本地复现时，先按 [安装说明](INSTALLATION.md) 建立 Python 3.9–3.11 环境并安装 CPU 依赖及开发工具，然后在仓库根目录运行：

```text
python -B -m pytest -q --cov=cv_human_video_matting --cov-report=term-missing --cov-fail-under=70
```

2026-10-02 在本地 Python 3.9.12 环境运行完整测试集，结果为 **717 passed、2 skipped、覆盖率 90.36%**；在 Python 3.10.16 / CPU PyTorch 2.5.1 / pytest 8.4.2 环境得到 **717 passed、2 skipped、覆盖率 90.37%**。这验证了本机上的完整测试命令；Gitee Go 平台上的解析、依赖下载与触发执行仍待实际开通后确认。CPU 测试不替代 GPU 推理或真实视频验收。

配置依据为 Gitee 官方文档：[开通流程与 `.workflow` 目录](https://help.gitee.com/gitee-go/get-started-in-3-steps)、[YAML 基本字段](https://help.gitee.com/gitee-go/pipeline/basic-config)、[Push 与 PR 触发格式](https://help.gitee.com/gitee-go/pipeline/trigger)、[阶段与任务编排](https://help.gitee.com/gitee-go/pipeline/scheduling)、[Python 构建插件与版本](https://help.gitee.com/gitee-go/plugin/ci-build)。
