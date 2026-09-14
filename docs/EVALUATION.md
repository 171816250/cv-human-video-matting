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
- `total_seconds` / `total_fps`：模型加载加完整处理流程，不包含报告生成。

这两个速度均不是纯模型推理FPS。CLI导出目前为无音轨MP4；网页端另有H.264转换，其耗时不包含在CLI报告中。首次模型下载会影响总耗时，正式评测需先下载权重，每组至少重复3次并报告中位数与波动。

## 质量和教学案例

准备经授权的至少3组案例，覆盖清晰单人、复杂背景和运动人物；另外记录发丝、弱光、半透明边缘等困难场景。每组保留相同帧位置的原图、alpha、合成结果，记录失败现象。

有真实alpha标注时再计算像素误差指标；无标注素材仅用于视觉比较，不编造质量分数。性能JSON中的 `quality_metrics` 为null，表示未测量。

MiDaS提供每帧归一化的相对逆深度，标签受场景内容变化影响，不能解释为米或跨场景标定结果。当前整个人物mask共用一个标签，不是多人独立测距。比较后处理时同时检查边缘污染、闪烁与拖影，不能只选择效果最好的一帧。

## 验收证据

保留输入素材说明、硬件信息、运行命令、JSON报告、对比截图、测试覆盖率报告和版本标识。离线单元测试、真实模型冒烟运行与人工质量评估分别记录，不能互相替代。
