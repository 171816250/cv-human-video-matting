# 申报目标与工程交付对应

| 申报目标 | 工程实现/验收入口 | 状态与后续工作 |
| --- | --- | --- |
| 视频抠像、背景合成 | pipeline、matting、render；CLI和Gradio | 已有实现，增加模型前景合成与异常资源释放回归测试 |
| 相对距离标注 | MiDaS、批处理和实时摄像头 | 使用现有MiDaS，统一alpha比例；Depth Anything仍为可选后续评估，不宣称已接入 |
| 时序稳定与模块化 | RVM循环状态、独立后处理 | 新视频及分辨率变化重置RVM状态；效果需真实素材评测 |
| 可复现性能验证 | CLI `--report`、网页同名JSON、EVALUATION.md | 记录处理和交付导出耗时；尚不代表达到某个GPU帧率目标 |
| 演示与视频交付 | doctor、H.264/AAC导出、DEFENSE_GUIDE.md | 环境预检、声音保留、异常保护已实现；仍需目标机器真人素材彩排 |
| 自动化测试、覆盖率≥70% | pytest、pytest-cov、GitHub Actions | 默认离线测试；提交远端后才会运行托管CI |
| 中文教程与贡献指南 | README、EVALUATION、CONTRIBUTING、API.md | 随参数和接口变更维护 |
| 国内开源托管 | Gitee仓库、LICENSE、贡献与反馈模板 | 已有托管；GitHub Actions配置不等于Gitee流水线已运行 |
| 正式版本发布 | CHANGELOG、发布检查 | 按申报阶段计划准备版本，正式Release以平台记录为准；本地候选包不计作远端发布 |
| 三组教学案例 | EVALUATION中的案例规范 | 需实际授权素材和人工效果评估 |
| 社区指标与市级申报 | 真实issue、PR、贡献者记录 | 属于后续运营与申报工作，不可由代码代替 |

已有的PROJECT_COMPLETE及历史报告记录过去阶段，不能替代本清单和当前测试结果。
