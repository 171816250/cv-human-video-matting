# 租赁 GPU 服务器迁移工具

本工具按 `/data/coding/cv-human-video-matting` 作为本次服务器默认根目录。`current` 指向当前版本的 `releases/<版本>/source`；`.venv-server` 是独立 Python 3.11 环境；`shared/weights`、`shared/data`、`shared/logs` 保存模型、输入、结果与运行日志。新供应商可用 `MATTE_SERVER_ROOT` 指向其持久盘上的其他绝对目录，`MATTE_SOURCE_DIR`、`MATTE_MODEL_DIR`、`MATTE_VENV` 可覆盖其余路径。所有需跨服务器保存的内容应下载到自己的电脑或持久存储；租赁服务器本身不是长期备份。

## 本地制作交付包

在项目根目录执行，指定已校验的离线 RVM 模型包和**新**的输出文件名：

```powershell
python -B scripts/server/portable.py pack --project . --model weights/offline-rvm-mobilenetv3-rc2 --output data/output/server-deployment.zip
python -B scripts/server/portable.py verify --archive data/output/server-deployment.zip
```

生成 `server-deployment.zip` 和 `server-deployment.zip.sha256`。ZIP 包含当前项目源码、`requirements-server-realtime.txt`、服务器脚本、`Connect-Remote-Realtime.ps1` 及整个模型包；不含虚拟环境、CUDA 驱动、密码、用户素材和输出结果。完整包通过模型包自身的逐文件 SHA256 校验，迁移 ZIP 也有逐文件清单与外置 SHA256。包中的源码取当前工作树，因此打包前应确认改动已保存；这与 Git 历史包用途不同。当前租赁机最初上传的旧 rc4 `candidate.zip` 不含后来新增的服务器部署脚本，不能用它代替这里的新 ZIP。新 ZIP 制作完成后需保存到自己的电脑并记录实际 SHA256。

## 新服务器安装

先将 ZIP、SHA256 文件、`portable.py`、`server.sh` 复制到新服务器的持久目录下。Ubuntu 24.04 通常自带可运行验证脚本的 `python3`；还需可用的 Python 3.11 来创建 `.venv-server`。如已有独立 Python 3.11 环境，可通过 `MATTE_PYTHON` 指向它，或提前将其放在目标 `.venv-server` 路径。脚本不会修改平台自带的 Torch 环境。以下 `/data` 路径是本次服务器示例；换供应商应替换为实际持久存储路径。

```bash
export MATTE_SERVER_ROOT=/data/coding/cv-human-video-matting
bash server.sh install-archive /data/incoming/server-deployment.zip v1.0.0
bash /data/coding/cv-human-video-matting/current/scripts/server/server.sh start
bash /data/coding/cv-human-video-matting/current/scripts/server/server.sh status
```

`install-archive` 验证 ZIP 与哈希、建立共享目录、安装独立依赖，然后完成 CUDA 张量与 120 帧 RVM WebSocket 真实推理。默认安装 Torch 2.10.0、TorchVision 0.25.0、CUDA cu128；不同显卡可设置 `MATTE_TORCH_VERSION`、`MATTE_VISION_VERSION`、`MATTE_CUDA_CHANNEL`。实时服务只监听 `127.0.0.1:7861`，其他电脑通过 SSH 转发到本机访问。该后台进程不依赖 systemd；容器重启后须再次执行 `start`。若旧进程已不存在，脚本会清理残留 PID 文件；若进程号已被其他程序复用，需先人工核实。

已有同名版本目录时使用新的版本名；安装失败若遗留未启用目录，同名重试前先核实残留并清理，或改用另一个版本名。`current` 只会在安装与真实推理验证成功后切换。

Windows 客户端可在项目根目录运行 `.\Connect-Remote-Realtime.ps1 -ServerHost SSH_HOST -SshPort SSH_PORT -SshUser SSH_USER` 建立本地转发，再打开 `http://127.0.0.1:7861/`。若本机端口已占用，可传入 `-LocalPort`。每台访问电脑都需自己的 SSH 登录权限；脚本不会保存密码。当前实时服务只允许一个活动会话。

已有源码与模型按当前目录布局准备好的服务器，可从项目源码运行 `bash scripts/server/server.sh setup`，它使用项目内独立环境，不重新解包，且包含 120 帧真实 RVM WebSocket 冒烟测试。`smoke` 可单独复验，报告保存在 `shared/data/output/server-smoke-*`。

本次 Ubuntu 24.04 实例实测 GPU 为 **RTX 5060 Ti 16GB**（租赁页面写 4060 Ti），使用 Conda-forge Python 3.11、Torch 2.10.0+cu128、TorchVision 0.25.0+cu128。120 帧合成画面、320×180、排除前 10 帧后，交付吞吐为 **31.51 FPS**，服务器处理 p50 **20.18 ms**、p95 **23.22 ms**。服务曾以 PID 1031 监听 `127.0.0.1:7861`；Windows 经 SSH 本地转发取得 HTTP 200，10 帧 WebSocket 请求在 CUDA 0 上通过。PID 是当次快照，重启后须重新检查。实际浏览器摄像头、真人画质和长时间稳定性尚未验收。

## 保存与迁移数据

停用服务器或换机器之前，在旧机器执行：

```bash
ROOT=/data/coding/cv-human-video-matting
bash "$ROOT/current/scripts/server/server.sh" stop
BACKUP="$ROOT/exports/data-backup-$(date -u +%Y%m%dT%H%M%SZ).zip"
bash "$ROOT/current/scripts/server/server.sh" backup "$BACKUP"
```

备份包括 `shared/weights` 下的模型实体、`shared/data/input`、`backgrounds`、`output` 与 `shared/logs`；`current-rvm` 绝对路径符号链接由新机器自行重建。即使这些目录都没有普通文件，脚本也允许生成只有清单的空备份；务必检查清单，不能把空备份当成数据已保存的证据。将 ZIP 和同名 `.sha256` **一起下载到自己的电脑**。部署包保存源码和模型，备份包保存模型副本及运行数据，二者都应留存。新机器完成安装后、启动服务前执行：

```bash
ROOT=/data/coding/cv-human-video-matting
BACKUP=/data/incoming/data-backup-实际时间.zip
bash "$ROOT/current/scripts/server/server.sh" restore "$BACKUP"
bash "$ROOT/current/scripts/server/server.sh" smoke
bash "$ROOT/current/scripts/server/server.sh" start
```

恢复命令允许相同文件重复恢复，遇到内容不同的同名文件会停止，不覆盖新机器上的数据。备份与部署包应只从可信来源接收；SHA256 检查用于发现传输损坏，不是数字签名。

如下一台服务器无法联网，可在一台**同架构 Linux/Python 3.11** 服务器上执行 `bash scripts/server/server.sh wheels /data/coding/cv-human-video-matting/exports/wheelhouse`，下载整个目录及 `SHA256SUMS`。迁移到新机后设置 `MATTE_WHEELHOUSE=/data/incoming/wheelhouse` 再执行安装。
