# 实时在线预览

本版新增独立的实时工作区。它使用浏览器 WebSocket 持续发送 JPEG 单帧，服务端为每个连接维护一份 RVM 循环状态；浏览器只有在收到并绘制上一帧后才采集下一帧，所以不会把慢帧堆在内存里。原来的 Gradio 视频处理页面继续保留，两个入口互不覆盖。

实时工作区默认只绑定 `127.0.0.1`，只接受同源页面的 WebSocket。它不接收客户端文件路径，也不接收整段视频；摄像头或本地视频只在浏览器中逐帧编码后发送。远程访问必须自行配置 HTTPS/WSS 反向代理，并明确承担访问控制、带宽和隐私责任。

## GPU 环境

Windows PowerShell 示例使用 Python 3.10。GPU 环境与原来的 CPU `.venv` 分开，避免影响已经验收的 CPU 交付环境：

```powershell
py -3.10 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install pip==24.3.1
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-realtime.txt
.\.venv-gpu\Scripts\python.exe -m pip check
.\.venv-gpu\Scripts\python.exe -B app.py doctor --requirements requirements-realtime.txt
```

本机已实际验证这套组合：Python 3.10.16、Torch 2.5.1+cu121、TorchVision 0.20.1+cu121、GTX 1660 Ti 6GB。完整运行包快照见 `requirements-lock-windows-py310-cu121.txt`；它绑定本机 Windows / Python / CUDA 组合，不适用于 CPU、Linux 或其他驱动。GPU 驱动和 CUDA 运行库仍需在目标电脑上单独检查。

## 启动

模型包必须已准备并通过逐文件校验。推荐双击 `Start-Realtime.cmd`；它只使用 `.venv-gpu`，不会安装依赖或下载模型：

```powershell
.\Start-Realtime.cmd --model-bundle weights/offline-rvm-mobilenetv3-rc2 --port 7861 --max-width 640
```

也可以直接运行：

```powershell
.\.venv-gpu\Scripts\python.exe -B app.py realtime --model-bundle weights/offline-rvm-mobilenetv3-rc2 --device cuda --host 127.0.0.1 --port 7861 --max-width 640
```

浏览器访问 `http://127.0.0.1:7861/`。点击“开始预览”后才会请求摄像头权限。选择“本地视频”可先用合成素材检查浏览器到服务端的完整链路；这不会上传整个文件。切换来源、效果、宽度或帧率会停止当前会话，重新点击开始才会应用新设置。

实时页面支持 15、24、30 帧/秒上限，实际显示帧率由服务耗时和浏览器绘制共同决定。宽度选择是发送前缩放：320 适合低延迟，640 是默认平衡值，960/1280 提供更多细节但需要更强 GPU。页面分别显示实际显示 FPS、从采集到结果显示的往返时间、服务端处理时间和已显示帧数。

## 服务基准

不使用浏览器摄像头时，可用脚本测试服务协议和单帧在途性能。它会启动自己的服务、限制连接只到本机、阻止 Torch Hub 下载并在结束时回收自己的进程；输出目录必须不存在：

```powershell
.\.venv-gpu\Scripts\python.exe -B scripts/verify_realtime.py --bundle weights/offline-rvm-mobilenetv3-rc2 --output-dir data/output/realtime-gpu-640 --device cuda --frames 120 --warmup 10 --width 640
```

本机 GTX 1660 Ti 6GB 的实测结果如下。数值是合成移动图形、RVM MobileNetV3、单连接、单帧在途；不是真人质量、摄像头权限或远程网络的结论。

| 发送宽度 | 预热后服务处理 p50 | 预热后往返 p50 | 连续交付 FPS | 峰值已分配显存 |
|---:|---:|---:|---:|---:|
| 320 | 20.4 ms | 22.6 ms | 33.7 | 37.3 MiB |
| 640 | 44.7 ms | 48.4 ms | 17.8 | 92.1 MiB |
| 960 | 87.2 ms | 93.0 ms | 9.5 | 99.1 MiB |

因此，当前实现若希望接近 30 FPS，应先把发送宽度设为 320；640 宽适合答辩展示和较平稳的约 15–18 FPS；960 宽不适合作为实时档。更换 GPU 可以降低推理时间，但不能绕过浏览器、JPEG 编解码和网络往返。多用户目前明确限制为一个实时会话，第二个连接会收到忙碌提示；需要并发时应做多 GPU/多进程压测，不能只按显存相加估算。

## 交付边界

本轮已验证真实 CUDA 服务、模型包身份、帧顺序、参数限制、同源校验、损坏 JPEG 拒绝、断开清理和单会话占用。浏览器控制器有 Node 仿真测试，覆盖摄像头权限失败、本地视频逐帧发送、停止、后台标签、迟到帧和对象 URL 回收；实际浏览器摄像头、不同浏览器兼容性、远程 HTTPS/WSS 和真人画质仍需在目标设备上补测。合成素材的帧率只能说明工程链路速度，不能替代真人案例。
