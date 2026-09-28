# 性能与效果验收

## 可复现的运行报告

以下命令均在项目根目录执行，真实模型需提前下载权重。使用相同输入和帧数比较参数，分别保存输出与报告：

```bash
python app.py matte --input data/input/demo.mp4 --output data/output/baseline.mp4 --model rvm_mobilenetv3 --device cuda --max-frames 300 --report data/output/baseline.json
python app.py matte --input data/input/demo.mp4 --output data/output/refined.mp4 --model rvm_mobilenetv3 --device cuda --max-frames 300 --edge-feather 1 --mask-close 1 --temporal-smooth --report data/output/refined.json
python app.py matte --input data/input/demo.mp4 --output data/output/depth.mp4 --model rvm_mobilenetv3 --device cuda --max-frames 300 --enable-depth --depth-interval 5 --report data/output/depth.json
```

报告包含输入尺寸/帧率、模型、实际设备、依赖版本、后处理参数、处理帧数和两种耗时：

RVM的 `--downsample-ratio auto` 使用 `min(1, 512 / 输入长边)` 作为内部推理比例，
`--downsample-ratio 1` 为全尺度推理。模型保持输出尺寸，但不同推理比例可能改变发丝等细节，
比较模型时应保持该参数一致。小于等于512像素长边的输入不会自动缩小。

- `elapsed_seconds` / `fps`：视频处理流程，包括首帧、视频读写、深度模型初始化和资源释放；不包含抠像模型加载。
- `total_seconds` / `total_fps`：模型加载加完整处理及交付导出，不包含报告生成、最终发布复制及旧文件备份准备。
- `export_seconds`：H.264转码/音频合并，或网页透明序列打包的耗时。
- schema3的 `metadata_seconds`：额外版本识别、文件哈希与媒体信息采集时间，不包含在total_seconds中，也不包含JSON序列化与落盘。

这两个速度均不是纯模型推理FPS。CLI与网页的总耗时均包含交付转换。首次下载会影响总耗时，正式评测需提前准备权重，每组至少重复3次并报告中位数与波动。历史schema_version=1未计入导出，不能直接比较。

## 报告追溯与汇总

schema3记录软件版本、Git提交与工作区状态、源码SHA256、输入/输出文件校验值、输出实际尺寸/帧率/帧数。透明PNG记录逐帧校验清单；帧序列时长按输入帧率解释。自定义背景与显式检查点另记录哈希。默认模型缓存/上游权重未自动核验，正式评测仍须固定并核验权重及机器负载。

0.2.0rc2使用`--model-bundle`时，报告会保存实际加载的包内容SHA256与权重SHA256，并在报告生成前复验。缺少加载身份确认的离线包报告不进入严格分组。源帧数未知时，输入帧数与时长记为null，明确标记完整性未验证；处理输出仍记录实际帧数，该类报告单列为信息不足。

EMA与自适应平滑的合成遮罩实验只验证算法对受控噪声、移动边界的行为，不能替代真人发丝质量。FFV1与mp4v中间编码的实验衡量编码失真，不等于抠像准确率；必须与抠像质量指标分开报告。

```bash
python -B app.py summarize --reports "data/output/benchmark/run-*.json" --output data/output/benchmark/summary.html
python -B app.py summarize --reports "data/output/benchmark/run-*.json" --output data/output/benchmark/summary.json
```

汇总按已记录条件分组，输出运行次数、处理和总FPS的中位数、最小值与最大值。不同分辨率、帧数、设备、代码或参数不混合。schema1/2及缺少追溯字段的报告只展示待核信息；坏JSON和自相矛盾的FPS会明确报错。报告路径支持通配符，但不要把环境诊断JSON或已生成的汇总JSON混入运行报告输入。

## 质量和教学案例

准备经授权的至少3组案例，覆盖清晰单人、复杂背景和运动人物；另外记录发丝、弱光、半透明边缘等困难场景。每组保留相同帧位置的原图、alpha、合成结果，记录失败现象。

有真实alpha标注时再计算像素误差指标；无标注素材仅用于视觉比较，不编造质量分数。性能JSON中的 `quality_metrics` 为null，表示未测量。

MiDaS提供每帧归一化的相对逆深度，标签受场景内容变化影响，不能解释为米或跨场景标定结果。当前整个人物mask共用一个标签，不是多人独立测距。比较后处理时同时检查边缘污染、闪烁与拖影，不能只选择效果最好的一帧。

## 验收证据

保留输入素材说明、硬件信息、运行命令、JSON报告、对比截图、测试覆盖率报告和版本标识。离线单元测试、真实模型冒烟运行与人工质量评估分别记录，不能互相替代。
