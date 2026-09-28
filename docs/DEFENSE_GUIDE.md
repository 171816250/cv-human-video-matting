# 答辩演示与交付操作

此文面向本地答辩演示。最终交付需要源码、模型及依赖准备说明、经授权的演示素材、输出样例和验证记录。版本发布、社区指标及校级/市级申报按真实材料填写。

## 上场前准备

0.2.0rc4 在 rc3 基础上增加独立 GPU 实时入口。批处理演示仍使用项目 `.venv` 与 `Start-Demo.cmd`；连续预览按[实时说明](REALTIME.md)准备 `.venv-gpu` 并使用 `Start-Realtime.cmd`。新机器先按安装说明准备环境和离线包；两种入口都不下载模型。实时页面默认只监听本机，摄像头仍需浏览器授权和现场实测。本机 GPU 基准见[实时验证记录](VALIDATION_v0.2.0rc4.md)。

下文以 Windows PowerShell 为例，在仓库根目录运行，始终使用项目 `.venv`，无须激活。macOS/Linux 将 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。模型包路径以 `weights/offline-rvm-mobilenetv3-rc2` 为例，交付位置不同须替换；启动器使用的临时配置不会自动影响之后单独执行的 CLI 命令。

1. 确认所用代码版本，记录 `git rev-parse HEAD`。保留可恢复的源码备份，演示期间固定环境与参数。
2. 在仓库根目录运行 `.\.venv\Scripts\python.exe -B app.py doctor --report data/output/environment.json`。FAIL需修复，WARN需结合提示完成试运行；PASS不是画质验收。`Start-Demo.cmd` 还会检查实际选定的离线模型包。
3. 提前准备并校验RVM离线模型包，运行一次实际素材；换电脑同时携带模型包和依赖准备说明。首次下载与模型包导出在联网准备阶段完成，具体见[安装说明](INSTALLATION.md)和[离线模型说明](OFFLINE_MODEL.md)。深度或BiRefNet首次运行可能需要额外资源，不要在现场首次启用。
4. 至少准备三组经授权的真人短片：清晰单人、复杂背景、运动人物。每组保留原片、参数、输出、JSON、典型失败截图。不要用合成人形冒烟片替代真实效果证据。
5. 在实际答辩机器、浏览器和投影环境上测试视频播放、声音、摄像头权限、导出和下载。准备一份已生成结果作为演示备用。

## 五分钟演示顺序

| 时间 | 操作 | 解释重点 |
| --- | --- | --- |
| 0:00–0:40 | 展示原片与任务背景 | 无绿幕自然场景的人像与背景分离 |
| 0:40–2:00 | “视频处理”选择RVM MobileNetV3，展示左右对比和图片背景 | 软alpha与模型前景合成；原始画面保持可比较 |
| 2:00–2:40 | 播放导出MP4并检查声音 | 文件可播放，保留原音轨；截取模式不带出多余音频 |
| 2:40–3:30 | 展示alpha和透明PNG | 透明通道用于后续合成，普通MP4不存透明alpha |
| 3:30–4:15 | 条件允许时开启相对距离与摄像头 | Near/Mid/Far是场景内相对远近，摄像头速度现场实测 |
| 4:15–5:00 | 展示JSON与测试记录、失败案例 | 区分处理速度、含加载/导出的总耗时与效果评估 |

## 可复制命令

```powershell
# 工程验收：提前准备模型包后，生成合成片并自动验证网页链路
.\.venv\Scripts\python.exe scripts/create_smoke_video.py
.\.venv\Scripts\python.exe -B scripts/verify_local_demo.py --bundle weights/offline-rvm-mobilenetv3-rc2 --input data/input/smoke_demo.mp4 --output-dir data/output/defense-web-rehearsal --device cpu --max-frames 8

# 离线网页启动；保持终端开启，后续命令在另一终端运行
.\.venv\Scripts\python.exe -B app.py ui --model-bundle weights/offline-rvm-mobilenetv3-rc2 --host 127.0.0.1 --port 7860 --no-share

# 正式素材运行；demo.mp4须团队另行准备经授权的真人原片
.\.venv\Scripts\python.exe -B app.py matte --model rvm_mobilenetv3 --model-bundle weights/offline-rvm-mobilenetv3-rc2 --device auto --input data/input/demo.mp4 --output data/output/demo-compare.mp4 --mode side_by_side --max-frames 300 --report data/output/demo-compare.json

# 透明输出到demo-alpha_frames；再次运行换输出名
.\.venv\Scripts\python.exe -B app.py matte --model-bundle weights/offline-rvm-mobilenetv3-rc2 --input data/input/demo.mp4 --output data/output/demo-alpha.mp4 --mode transparent --max-frames 60 --report data/output/demo-alpha.json

# 自动化回归；提前按安装说明安装开发依赖，不下载模型
.\.venv\Scripts\python.exe -B -m pytest -q --cov=cv_human_video_matting --cov-report=term --cov-report=xml --cov-fail-under=70

# GPU 实时服务基准；需要独立 .venv-gpu，输出目录必须不存在
.\.venv-gpu\Scripts\python.exe -B scripts/verify_realtime.py --bundle weights/offline-rvm-mobilenetv3-rc2 --output-dir data/output/realtime-gpu-640 --device cuda --frames 120 --warmup 10 --width 640
```

工程验收目录必须是新目录，重复运行须换名。`smoke_demo.mp4` 仅验证流程；`demo.mp4` 不随源码提供，真人质量验收仍需原片来源与使用授权、同帧对比和失败记录。

网页批处理保存result.json并提供下载，状态栏显示路径。CLI报告路径必须以.json结尾，不能与输入或输出相同。总耗时含模型加载、处理和交付导出；短片单次FPS不能代表长视频或所有设备性能。

### 0.2.0rc1操作变化

网页“视频处理”默认处理完整视频，帧数上限0表示全部；可直接设置输出短边、背景颜色和音频。新增进度显示及停止按钮。停止仅影响当前浏览器会话的运行中任务，模型加载/单帧处理需等待当前步骤返回；排队中的任务暂不支持撤回。摄像头与批处理共享推理队列，批处理期间摄像头请求会等待。

每次成功的网页任务保存为一个独立目录，包含媒体结果和result.json，可在界面直接下载报告；失败或停止时不发布部分结果。报告还保存源码与输入/输出SHA256以及实际输出尺寸。哈希采集时间单独记录，不能当作模型推理时间。

答辩时可打开 `summarize` 生成的中文汇总页，展示相同已记录条件下多次运行的中位数和波动，并用 `scripts/package_release.py` 生成可校验的源码与证据包。详见README中的命令。

## 现场故障处理

0.2.0rc2可提前准备带内容校验的RVM离线模型包，并运行`scripts/rehearse_offline.py`完成空缓存试运行。时间平滑新增自适应选项，默认仍关闭；对比相同动作的结果后再决定是否用于演示。Doctor现在核对声明的依赖版本，本机运行成功与全新环境安装验收需要分别记录，失败项不能改填为通过。操作见[离线模型与部署](OFFLINE_MODEL.md)。

| 现象 | 处理 |
| --- | --- |
| 模型加载等待或网络失败 | 检查权重缓存，先用准备好的输出讲解；回到已试运行的RVM模型 |
| 处理速度低 | 使用MobileNetV3，关闭深度，降低实时预览宽度或批处理短边；保存新参数以便比较 |
| 显式CUDA不可用 | 运行doctor核对环境；选择CPU完成基本功能演示并记录实际设备 |
| 视频不能导出 | 查看明确的FFmpeg错误，运行doctor；CLI失败保留原有结果文件 |
| 画面边缘拖影 | 关闭或降低时间平滑，用相同帧比较原始与后处理结果 |
| 摄像头无图像 | 核对浏览器权限、其他软件占用及访问地址；使用批处理素材继续演示 |
| 透明输出已存在 | 换一个输出名，避免覆盖已有帧 |

## 最终提交前仍需确认

- [ ] 三组真人案例与边缘/运动/弱光失败案例已整理，并确认素材授权。
- [ ] 真实素材每组至少三次运行，保留硬件、帧数、耗时中位数与波动。
- [ ] MiDaS与可选BiRefNet是否纳入正式演示，分别具有实际运行证据。
- [ ] 目标机器的全新环境安装、离线启动、摄像头、长视频和浏览器播放完成验收。
- [ ] 源码、运行指南、模型来源及许可说明、样例、报告和版本标识齐全。
- [ ] 根据申报书核对正式Release、提交/分支/贡献者及社区指标，附平台证据。

以上未勾选项不能仅凭自动化测试填为已完成。经费与管理材料单独核对，不从代码或测试推算。
