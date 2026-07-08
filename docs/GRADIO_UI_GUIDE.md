# 🎨 Gradio UI 使用指南

**更新日期**: 2026-07-02  
**版本**: v2.0 - 完整功能版

---

## ✨ 新功能亮点

Gradio UI 现已集成所有新功能：

- ✅ **三档模型**: RVM-MobileNetV3 / RVM-ResNet50 / BiRefNet
- ✅ **7种渲染模式**: green / blur / transparent / image / alpha / foreground / side_by_side
- ✅ **⭐ 距离标注**: Near/Mid/Far 相对距离标注（核心创新）
- ✅ **后处理增强**: 边缘羽化 / 形态学处理 / 时序平滑

---

## 🚀 快速启动

### 1. 启动UI

```bash
# 进入项目目录
cd /d/claude_projects/开源项目/cv-human-video-matting

# 启动Gradio Web UI
python app.py ui

# 或指定端口
python app.py ui --port 7860 --host 0.0.0.0
```

### 2. 浏览器访问

启动后会显示：
```
Running on local URL:  http://127.0.0.1:7860
```

在浏览器中打开此URL！

---

## 📋 功能详解

### Tab 1: Batch Render（批量渲染）

#### 输入区域
- **Input Video File**: 上传视频文件（mp4, avi, mov等）
- **Background Image**: 自定义背景图片（仅在image模式下需要）

#### 模型选择
1. **rvm_mobilenetv3** - 轻量级，最快，适合CPU
2. **rvm_resnet50** - 高质量，平衡速度和质量（推荐）
3. **birefnet** - 高精度，SOTA边缘质量，较慢

#### 渲染模式
- **green** - 绿幕背景（传统抠图）
- **blur** ⭐ - 背景虚化（人像景深效果）
- **transparent** ⭐ - 透明PNG帧序列（BGRA四通道，UI提供zip下载）
- **image** - 自定义背景图片替换
- **alpha** - Alpha遮罩可视化
- **foreground** - 黑色背景
- **side_by_side** - 左右对比

#### ⭐ 距离标注（核心创新）
- **Enable Distance Labeling**: 开启/关闭距离标注
  - 勾选后会在人物上显示 Near/Mid/Far 标签
  - 显示相对距离值（如 d~0.45）
- **Depth Interval**: 深度估计间隔
  - 每N帧估计一次深度
  - 默认5帧，平衡性能和准确性
  - 值越小越精确，但更慢

#### Alpha Controls（Alpha控制）
- **Edge Feather**: 边缘羽化（0-9）
  - 柔化Alpha边缘
  - 值越大边缘越柔和
- **Mask Open**: 开运算（0-9）
  - 移除小的前景噪点
- **Mask Close**: 闭运算（0-9）
  - 填充小的背景孔洞

#### Temporal Controls（时序控制）
- **Temporal Smooth**: 时序平滑开关
  - 减少视频闪烁
- **Temporal Blend**: 混合比率（0.0-0.95）
  - 当前帧与前一帧的混合权重
  - 值越大越平滑，但可能产生拖影

#### 其他参数
- **Device**: 选择cpu或cuda
- **Max Frames**: 最大处理帧数（0=全视频）

---

## 🎯 使用示例

### 示例1: 基础绿幕抠图
```
1. 上传视频
2. Model: rvm_resnet50
3. Mode: green
4. Device: cuda
5. Max Frames: 0 (全视频)
6. 点击"Run Batch Matting"
```

### 示例2: 背景虚化 + 距离标注 ⭐
```
1. 上传视频
2. Model: rvm_resnet50
3. Mode: blur
4. Enable Distance Labeling: ✓
5. Depth Interval: 5
6. Device: cuda
7. 点击"Run Batch Matting"
```

### 示例3: 高精度透明PNG导出
```
1. 上传视频
2. Model: birefnet
3. Mode: transparent
4. Device: cuda
5. Max Frames: 1 (仅第一帧)
6. 点击"Run Batch Matting"
```

### 示例4: 自定义背景 + 边缘优化
```
1. 上传视频
2. 上传背景图片
3. Model: rvm_resnet50
4. Mode: image
5. Alpha Controls:
   - Edge Feather: 3
   - Mask Close: 2
6. 点击"Run Batch Matting"
```

---

## 📊 输出说明

处理完成后会显示：
```
Output: data/output/gradio_20260702_153000_blur.mp4
Model: rvm_resnet50
Mode: Blur
Frames: 120
Elapsed: 6.500s
FPS: 18.46
Depth Labeling: Enabled (interval=5)
```

输出视频会自动转换为浏览器可播放格式。

---

## 💡 性能优化建议

### CPU环境
- 使用 **rvm_mobilenetv3** 模型
- **不启用**距离标注
- 降低分辨率或减少帧数

### GPU环境（推荐）
- 使用 **rvm_resnet50** 模型（平衡）
- 或 **birefnet** 用于最佳质量
- 可启用距离标注
- 全分辨率处理

### 距离标注性能影响
- 每5帧估计：FPS下降约10%
- 每10帧估计：FPS下降约5%
- 每1帧估计：FPS下降约20%

---

## ⚠️ 注意事项

### 1. BiRefNet首次使用
- 首次使用会从HuggingFace下载模型（~1.5GB）
- 国内已配置镜像加速（hf-mirror.com）
- 下载后会缓存到 `weights/BiRefNet/`

### 2. 距离标注说明
- 这是**相对距离**估计，不是绝对物理距离
- Near/Mid/Far是基于单目深度的分类
- 适用于视觉效果，不适用于精确测量

### 3. 透明模式输出
- `transparent` 模式输出BGRA四通道PNG帧序列
- UI会提供zip下载，不使用视频播放器预览
- CLI传入 `--output result.mp4` 时，实际输出目录为 `result_frames/`
- 可导入后期软件或再用ffmpeg编码成支持alpha的格式

### 4. 显存占用
- **rvm_mobilenetv3**: ~2GB
- **rvm_resnet50**: ~3GB
- **birefnet**: ~5GB
- **+距离标注**: 额外 ~1GB

如果遇到OOM错误，尝试：
- 切换到更轻量的模型
- 降低Max Frames
- 关闭距离标注

---

## 🐛 常见问题

### Q: UI启动失败
A: 确保安装了gradio：`pip install gradio`

### Q: 视频上传后无法预览
A: 浏览器可能不支持该格式，处理后输出会自动转换

### Q: BiRefNet加载很慢
A: 首次下载模型需要时间，请耐心等待

### Q: 距离标注不准确
A: 单目深度估计固有限制，结果仅供参考

### Q: 处理速度很慢
A: 检查Device是否选择cuda，或降低Max Frames

---

## 🎉 享受使用！

现在你可以通过直观的Web界面体验所有新功能了！

如有问题，请查看：
- README.md
- docs/COMPLETION_REPORT.md
- 或运行CLI查看更多选项：`python app.py matte --help`
