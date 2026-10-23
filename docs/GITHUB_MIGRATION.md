# GitHub 托管与双平台同步交接

记录日期：2026-10-23。本文说明本次迁移的源码基线和后续同步方法，实际远端分支、Actions 与发行状态需在平台核验后记录。

## 仓库与基线

| 项目 | 迁移约定 |
| --- | --- |
| GitHub | [171816250/cv-human-video-matting](https://github.com/171816250/cv-human-video-matting)，默认分支 `main` |
| Gitee | [ciyechenchen/cv-human-video-matting](https://gitee.com/ciyechenchen/cv-human-video-matting)，保留原有国内访问与协作入口 |
| 当前正式标签 | `v1.0.0` |
| 正式版源码提交 | `35f6ed9de7b047c912481307468a33a2f7a5dd85` |
| 正式版可达历史 | 本地核对为 **17 次真实提交**，后续提交数随实际改进增加，平台统计口径需另行核对 |
| Gitee 当前正式分支 | `feature/incubation-evidence-depth-2026-10-02` |
| 旧 Gitee main | 迁移准备时为 `2039b50`，尚未包含上述正式版；不能直接将它当作新 GitHub 默认分支的源码基线 |

GitHub 的 `main` 应从上述正式版及后续已核实的改进提交建立。README 托管入口与贡献/交接说明属于发行后的文档改进，不改变已冻结 `v1.0.0` 的源码、标签、发行附件或 SHA256。后续新版在完成实际变更、检查和说明后另行发行。

## 两个远端的用途

本地约定已有 `origin` 指向 Gitee，新增 `github` 指向 GitHub。实际远端别名或地址不同时，先核对再操作，避免把不同平台的 `main` 当成相同内容。

首次同步只选择明确需要的引用：

- GitHub `main`：正式版源码及后续已核实的改进，是 GitHub 的贡献基线。
- `feature/incubation-evidence-depth-2026-10-02`：仅在需要保留开发分支时明确选择推送。
- `deploy/remote-gpu-2026-10-02`：仅在需要保留部署分支时明确选择推送；其中的历史运行记录不代表当前租赁服务器仍在线。
- 正式 `v1.0.0` 标签：单独选择同步，推送后同时核对标签对象及其对应源码提交。

本地 `backup/*`、`defense/*` 等备份和候选引用不在本次推送清单中。使用明确的源分支、目标分支与标签，不使用批量镜像或推送全部本地引用的方式。每项提交应对应已完成的修复、测试或文档改进，提交说明写明实际变化。

## 后续同步顺序

1. 先查看两个平台的目标分支，确认是否有新的贡献；记录各自的提交号。
2. GitHub 贡献从 `main` 创建功能分支并经 Pull Request 评审。Gitee 当前贡献基线见[贡献说明](../CONTRIBUTING.md)。
3. 合并前执行对应检查，处理另一平台新增变更；发生分歧时通过正常合并或 Pull Request 解决，再同步可追溯的结果。
4. 对每个平台显式指定目标分支。GitHub 的 `main` 与 Gitee 的当前正式分支名字不同，不能依赖同名推送来完成同步。
5. 推送后重新读取远端提交号，核对预期内容。对正式标签另核对其对应提交，已冻结标签不移动。
6. 记录同步日期、来源与目标分支、提交号、实际检查结果及未解决事项。记录不包含令牌、密码、私钥或私人素材。

## GitHub Actions 验证

已有工作流为 [`.github/workflows/tests.yml`](../.github/workflows/tests.yml)，配置 Push、Pull Request 与手动触发，使用 Ubuntu、Python 3.10 和 CPU 依赖运行完整 pytest，要求核心语句覆盖率至少 70%，并上传 `coverage.xml`。

首次推送后，在 [GitHub Actions 页面](https://github.com/171816250/cv-human-video-matting/actions/workflows/tests.yml)核对：

- 触发事件、分支与提交号是否属于本次同步。
- 依赖安装、完整测试和覆盖率门槛的实际结论。
- 失败、跳过和警告的原因，以及覆盖率附件是否可下载。
- 后续 Pull Request 与手动触发是否按预期工作。

本地 Windows 的历史结果不等于 GitHub Ubuntu 的实际结果；例如目录符号链接权限与 Node 是否可用会影响跳过项。平台执行结论尚未核验时应记录“待核验”，仅存在工作流文件不能写成 CI 已通过。Gitee Go 状态继续按[原说明](GITEE_CI.md)独立核实。

## 发行与指标记录

若在 GitHub 同步 `v1.0.0` Release，沿用对应的正式标签，公开附件使用已经冻结并核对 SHA256 的文件。推送标签本身不代表已经创建 GitHub Release；发行页面、附件下载及哈希需要逐项确认。原 Gitee 发行与历史记录继续保留。

GitHub 的公开访问、默认分支、实际提交与分支数量、成员、Stars、Forks、访问/下载量，以及 CI 和发行结果均需在平台逐项核验。两平台数据分别记录，不直接将 Gitee 的数字填入 GitHub。普通仓库体积、Git LFS 额度与已用空间也是不同指标，需另行取证。

交接时至少保存两个仓库地址、正式标签与提交号、实际同步引用、Actions 链接和结果、公开附件及哈希，以及后续待完成的指标。备份 Git bundle 与服务器私人数据继续在本地妥善保存。
