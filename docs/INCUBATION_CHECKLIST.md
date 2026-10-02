# 申报目标与工程交付对应

| 申报目标 | 工程实现/验收入口 | 状态与后续工作 |
| --- | --- | --- |
| 视频抠像、背景合成 | pipeline、matting、render；CLI和Gradio | 已有实现，增加模型前景合成与异常资源释放回归测试 |
| 相对距离标注 | MiDaS；实验性Depth Anything V2 Small本地后端；[准备说明](DEPTH_ANYTHING.md)与[验证记录](VALIDATION_DEPTH_ANYTHING_2026-10-02.md) | 已接入CLI和Gradio，本机CPU真实权重合成帧烟测通过；真人效果与连续速度尚待验收。Near/Mid/Far非米数，独立WebSocket工作区未接入深度 |
| 时序稳定与模块化 | RVM循环状态、独立后处理 | 新视频及分辨率变化重置RVM状态；效果需真实素材评测 |
| 可复现性能验证 | CLI `--report`、网页同名JSON、EVALUATION.md | 记录处理和交付导出耗时；尚不代表达到某个GPU帧率目标 |
| 演示与视频交付 | doctor、H.264/AAC导出、DEFENSE_GUIDE.md | 环境预检、声音保留、异常保护已实现；仍需目标机器真人素材彩排 |
| 自动化测试、覆盖率≥70% | pytest、pytest-cov、GitHub Actions；Gitee Go完整离线测试配置 | 流水线配置完整测试集和70%覆盖率门槛；本地结果以对应版本记录为准，Gitee平台仍待开通运行，不宣称已通过 |
| 中文教程与贡献指南 | README、EVALUATION、CONTRIBUTING、API.md | 随参数和接口变更维护 |
| 国内开源托管 | Gitee仓库、LICENSE、贡献与反馈模板；[Gitee Go说明](GITEE_CI.md) | 已有托管和`.workflow/python-ci.yml`；尚无本轮Gitee平台构建成功证据，需开通后核对触发和日志 |
| 正式版本发布 | CHANGELOG、发布检查 | 按申报阶段计划准备版本，正式Release以平台记录为准；本地候选包不计作远端发布 |
| 三组教学案例 | [案例规范与收集工具](CASE_EVIDENCE.md)、`collect_case_evidence.py collect/verify` | 已有私有目录收集、同帧截图及哈希复核工具；授权真人素材、三组案例和人工效果结论仍需实际补齐 |
| 社区指标与市级申报 | 真实issue、PR、贡献者记录 | 属于后续运营与申报工作，不可由代码代替 |

已有的PROJECT_COMPLETE及历史报告记录过去阶段，不能替代本清单和当前测试结果。

Depth Anything的本地文件校验、单帧真实权重烟测与真人模型效果是不同验收项。案例收集工具保留`quality_metrics: null`和未评价状态，不把反馈意见转换成评分。Gitee Go配置与本地测试结果也不构成平台运行记录。
