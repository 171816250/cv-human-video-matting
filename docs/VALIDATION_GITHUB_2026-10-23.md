# GitHub 平台验证记录（2026-10-23）

本记录固定对应迁移后的 `main` 提交 `bd1d260d18a55b885957e78f77de4c17e222ea0d`，记录该提交的公开访问、完整性和手动 CI 结果。后续提交及其他触发方式应查看各自的运行记录。

## 平台测试结果

| 项目 | 实际记录 |
| --- | --- |
| 仓库 | [171816250/cv-human-video-matting](https://github.com/171816250/cv-human-video-matting) |
| 工作流 | Offline regression tests |
| 运行 | [39953624033](https://github.com/171816250/cv-human-video-matting/actions/runs/39953624033) |
| 触发事件 | `workflow_dispatch`，手动执行 |
| 分支与提交 | `main`，`bd1d260d18a55b885957e78f77de4c17e222ea0d` |
| 环境 | GitHub-hosted Ubuntu 24.04，Python 3.10，CPU 依赖 |
| 结论 | `completed` / `success` |
| 完整测试 | **726 passed、0 failed、0 skipped、46 warnings** |
| pytest 耗时 | **23.44 秒**，不含依赖安装等工作流步骤 |
| 核心语句覆盖率 | **91.03%（2599/2855）**，达到 70% 门槛 |
| 覆盖率附件 | `coverage-report` 已成功上传并下载核对，包含 `coverage.xml` |

该结果证明上述提交在 GitHub Ubuntu 环境中完成了一次真实的手动工作流。Push 自动触发、Pull Request 触发及其他分支不在这次运行的验收范围内。Windows 本地记录为 724 passed、2 skipped；本次 Ubuntu 运行的两项目录符号链接相关测试也实际通过，因此为 726 passed、0 skipped。

测试覆盖工程处理与模型适配路径，仍使用模型替身和合成音视频。平台测试通过不代表真实模型画质、GPU 实时速度、真人摄像头或远程网络已经完成验收，也不改变 Gitee Go 的独立验证状态。

## 公开访问与源码完整性

匿名访问确认仓库为公开仓库，默认分支为 `main`。本次快照中有以下三个分支：

- `main`：`bd1d260d18a55b885957e78f77de4c17e222ea0d`，可达历史为 19 次真实提交。
- `feature/incubation-evidence-depth-2026-10-02`：保留当前正式版基线。
- `deploy/remote-gpu-2026-10-02`：保留部署分支。

通过匿名克隆下载源码，对该基线的 **159 个受 Git 跟踪文件**逐字节与提交内容核对，全部一致。该检查验证公开下载的源码完整性，不统计本机模型、虚拟环境或私人素材。

## 正式版本与附件

软件版本仍为 **1.0.0**。正式标签 `v1.0.0` 对应源码提交 `35f6ed9de7b047c912481307468a33a2f7a5dd85`；迁移后的 `main` 包含后续托管文档，不移动正式标签或改写已冻结附件。

[GitHub v1.0.0 发行页](https://github.com/171816250/cv-human-video-matting/releases/tag/v1.0.0)已确认公开，非草稿、非预发行。以下四个附件均实际下载，SHA256 与原 Gitee 冻结文件一致：

- `cv-human-video-matting-1.0.0-source.zip`
- `cv-human-video-matting-1.0.0-validation.zip`
- `RELEASE_NOTES.md`
- `SHA256SUMS.txt`

平台测试日志、运行元数据、覆盖率和匿名克隆校验记录已另行备份。双平台的后续同步、贡献分支及指标核验见[交接指引](GITHUB_MIGRATION.md)。
