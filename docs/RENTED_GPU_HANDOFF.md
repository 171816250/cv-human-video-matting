# 临时 GPU 服务器交接与迁移

本文用于每次更换租赁服务器时复现本项目的实时预览，并把必须留存的数据带回自己的电脑。先填写本次机器记录，再按安装、远程访问、备份、恢复和验收的顺序操作。命令中的路径和版本以本次机器的实际检查结果为准。完整浏览器摄像头和真人效果仍需逐台验收。

## 2026-10-02 当前实例已核实的信息

服务器实际运行 **Ubuntu 24.04，NVIDIA RTX 5060 Ti 16GB**；租赁页面所称的“4060 Ti”与 `nvidia-smi` 实际型号不一致，后续以服务器实测为准。`/data` 容量约 90GB，检查时约 81GB 可用。平台登录提示：释放实例时只有 `/data` 可以选择“保存数据”，默认保留 30 天，并可在同集群换机时挂载。这个平台保存选项仍需在释放前主动确认；本地备份是独立的必要步骤。

本次根目录为 `/data/coding/cv-human-video-matting`。独立 `.venv-server` 使用 Conda-forge Python 3.11、Torch 2.10.0+cu128、TorchVision 0.25.0+cu128。旧 rc4 `candidate.zip` 已上传并核对 SHA256 `fcd92333b79ae63355f531cd9cd1c26e040fd024c5f915d4f18d8aeb7d25e0e9`；它不含后来新增的服务器部署脚本，**不能直接作为下一台机器的一键部署包**。初始验证源码位于 `releases/v0.2.0rc4/source`，模型位于 `shared/weights/offline-rvm-mobilenetv3-rc2`；最终启用的版本以本地 `machine-record.md` 和服务器 `current` 链接为准。

`server.sh smoke` 的真实 CUDA/RVM WebSocket 基准已处理 120 帧合成画面，尺寸 320×180。去掉前 10 帧后，交付吞吐 **31.51 FPS**，服务器处理耗时 p50 **20.18 ms**、p95 **23.22 ms**。实测时服务以 PID 1031 监听 `127.0.0.1:7861`；Windows 通过 SSH 本地转发取得 HTTP 200，10 帧 WebSocket 请求在 CUDA 0 上通过。以上只证明当前实例的服务链路和合成帧速度；实际浏览器摄像头、真人画质及长时间稳定性尚未验证。PID 只用于这次记录，重启后须重新检查。

## 每台机器都要新建的记录

将下表复制到本地备份目录中的 `machine-record.md`，每租一台机器填写一次。可以保存 SSH 地址和端口，**不要保存 SSH 密码、私钥、访问令牌或真实人物素材的公开下载链接**。项目仓库可能公开，不要把机器记录直接提交到 Git。

| 项目 | 本次实测值 |
| --- | --- |
| 租用日期、到期时间、服务商和实例编号 | 待填写 |
| SSH 用户、主机名、端口 | 待填写；下次租机全部重新核对 |
| 服务器系统、CPU、内存、磁盘空间 | 待填写 |
| `nvidia-smi` 显示的 GPU、显存、驱动版本 | 待填写 |
| 服务器根目录、源码目录、`current` 链接目标 | 待填写 |
| 源码提交 SHA 或交付包 SHA256；远端未提交修改 | 待填写 |
| Python、PyTorch、TorchVision、CUDA 可用性 | 待填写 |
| 模型包目录及 `content_sha256` | 待填写 |
| 服务启动命令、监听地址、应用端口、进程管理方式 | 待填写 |
| 从另一台电脑的访问方式和实测地址 | 待填写 |
| 本次验证日期、浏览器、输入来源、显示 FPS、往返延迟 | 待填写；区分脚本基准与实际浏览器 |
| 备份文件名、保存位置、文件大小、SHA256、核对日期 | 待填写 |
| `/data` 保存选项、平台保留截止日期、可挂载集群 | 待填写；每台机器重新确认 |
| 未解决问题及下次迁移要避开的配置 | 待填写 |

每次换机器都要重填 `SSH_HOST`、`SSH_PORT`、`SSH_USER`、`MATTE_SERVER_ROOT`、`MATTE_PORT`、模型包位置、备份文件路径和远程访问地址。不要沿用上一台机器的 GPU、驱动、域名、端口映射、磁盘持久性或测速结论。

## 新机器准备

1. 用租赁平台本次给出的地址和端口连接；首次连接时如平台提供主机指纹，应与 SSH 提示核对。记录 `uname -a`、`nvidia-smi`、`df -h`、`python3 --version`。若驱动未使 GPU 可用，先解决驱动或更换实例，再安装项目。
2. 从当前源码和离线模型制作 `server-deployment.zip` 及同名 `.sha256`；制作方法见 [服务器脚本说明](../scripts/server/README.md)。安装前核对新包及逐文件清单。已上传的旧 rc4 `candidate.zip` 不含新增的 `server.sh`、`portable.py` 和服务器依赖清单，不适合作为下一台机器的安装输入。把新包和数据备份分别保存在自己的电脑，再传到新机的持久存储目录；核对双方 SHA256。
3. 首次建机时，把与新部署包同版的 `scripts/server/server.sh`、`portable.py` 放到引导目录，执行 `bash 引导目录/server.sh install-archive server-deployment.zip 版本名`。它会校验归档，把源码安装到 `releases/版本名/source`，模型放到 `shared/weights/`，创建 `current`，安装独立环境并执行真实 RVM WebSocket 冒烟测试；目标版本目录和 `current` 必须未被占用。不要只记录一个版本名，还要保存包的 SHA256。
4. Linux 实时部署使用独立 `.venv-server` 和 `requirements-server-realtime.txt`。`server.sh` 目前默认 Python 3.11、Torch 2.10.0、TorchVision 0.25.0、`cu128`；版本可通过脚本列出的 `MATTE_*` 环境变量调整，必须按新 GPU 和驱动重新验证。此前 Windows `.venv-gpu`、`requirements-realtime.txt` 和 Torch 2.5 / CUDA 12.1 的验收结果不能套用到本次 RTX 5060 Ti。`setup` 会检查依赖、模型包、CUDA 张量并运行 120 帧 RVM WebSocket 推理；`smoke` 可单独复验。模型包自身的内容身份按 [离线模型说明](OFFLINE_MODEL.md) 记录。
5. 对于**已手动建好** `current` 和 `.venv-server`、但没有通过 `install-archive` 安装的服务器，可用以下命令安装、测试并启动；`install-archive` 成功时已做 `setup` 和 `smoke`，不必重复。将服务状态和日志路径写进机器记录。脚本强制服务只监听 `127.0.0.1`：

   ```bash
   ROOT=/data/coding/cv-human-video-matting
   bash "$ROOT/current/scripts/server/server.sh" setup
   bash "$ROOT/current/scripts/server/server.sh" start
   bash "$ROOT/current/scripts/server/server.sh" status
   ```

   换机器时把 `MATTE_SERVER_ROOT` 和 `ROOT` 设为实际持久存储路径，可以不是 `/data`；若应用端口变化，设置 `MATTE_PORT` 并同步修改各客户端 SSH 转发。服务器脚本将日志写在 `shared/logs/`，运行数据写在 `shared/data/`，模型位于 `shared/weights/`；这些路径比 `releases/.../source` 更适合跨版本保留。

## 从其他电脑访问

先使用 SSH 本地端口转发。它沿用已经开放的 SSH 连接，服务器仍可只监听 `127.0.0.1`。在**要使用网页的那台电脑**上执行，替换本次的主机、端口和用户：

```bash
ssh -p SSH_PORT -N -L 127.0.0.1:7861:127.0.0.1:7861 SSH_USER@SSH_HOST
```

Windows 用户也可在项目根目录运行 `.\Connect-Remote-Realtime.ps1 -ServerHost SSH_HOST -SshPort SSH_PORT -SshUser SSH_USER`，脚本会检查转发建立情况，默认连接到服务器 7861 端口，且不会保存密码。保持隧道窗口开启，在同一台电脑浏览器打开 `http://127.0.0.1:7861/`，先用本地视频验证，再按浏览器提示授权摄像头。若本机 7861 已占用，手工命令只改 `-L` 左侧端口，脚本则使用 `-LocalPort`，并访问对应的新本机端口。每台需要访问的电脑都要各自建立 SSH 转发并有 SSH 登录权限；当前实时工作区只允许一个活动会话。浏览器上传的是逐帧 JPEG，服务器会处理这些帧，因此只使用有权处理的素材。本次已从 Windows 经 SSH 转发验证 HTTP 与 WebSocket/CUDA 10 帧，浏览器视频和摄像头仍需单独测试。

如需要让没有 SSH 账号的电脑直接通过公网网址访问，应先配置 HTTPS/WSS 反向代理、身份验证、访问限制与服务器防火墙，再实测浏览器摄像头和 WebSocket；记录最终网址和配置位置。不能仅把应用监听地址改为 `0.0.0.0` 就视为完成远程交付。公网方案与具体域名、证书和端口映射依赖每台服务器，须逐台验收。

## 数据保存到自己的电脑

本次部署把源码和模型装在版本目录及 `shared/weights/`；运行数据、日志分别在 `shared/data/`、`shared/logs/`。源码目录中的 `weights/portable-rvm`、`data/` 等可能是符号链接，不能只打包 `current/` 或 `releases/.../source`，否则会漏掉链接指向的真实文件。未来迁移应保存两类归档：**`server-deployment.zip` 与 `.sha256` 保存工程源码和模型**；**`server.sh backup` 的 ZIP 与 `.sha256` 保存 `shared/weights` 模型实体、`shared/data/input`、`backgrounds`、`output` 和 `shared/logs`**。旧 `candidate.zip` 仅作为本次 rc4 来源记录，不能代替新部署包。项目外的素材、配置、证书或其他运行目录若实际使用，还要单独列出并备份。不要把密码和私钥收入这些归档。

当前租赁平台只提示 `/data` 可在释放实例时选择保存 30 天并在同集群挂载。若本次安装在其他路径，应在退租前将所需文件搬到 `/data` 并核对内容；即便安装在 `/data`，仍应按下文拉回自己的电脑。下次换服务商、换集群或超过保留期时，不能依赖该平台副本。

每次重要测试结束后保存一次，退租前停服务并做最终备份。以下命令在服务器上创建共享内容归档；输出路径必须是新的文件名，且位于 `shared/` 之外。脚本会创建同名 `.sha256` 校验文件。空备份也允许，清单会明确没有文件；不能把空备份当成模型和数据已保存的证据：

```bash
ROOT=/data/coding/cv-human-video-matting
bash "$ROOT/current/scripts/server/server.sh" stop
BACKUP="$ROOT/backups/data-$(date -u +%Y%m%dT%H%M%SZ).zip"
bash "$ROOT/current/scripts/server/server.sh" backup "$BACKUP"
python3 "$ROOT/current/scripts/server/portable.py" verify --archive "$BACKUP"
cat "$BACKUP.sha256"
```

在自己的 Windows PowerShell 上，把**共享内容 ZIP 和同名 `.sha256`** 都拉回本地；新 `server-deployment.zip` 制作完成后，也把它及校验文件保存在本地。示例变量都要替换成本次机器和归档的实值；密码由 SSH 交互提示输入，不写进命令或文件：

```powershell
$SshTarget = 'SSH_USER@SSH_HOST'
$SshPort = 'SSH_PORT'
$RemoteArchive = '/data/coding/cv-human-video-matting/backups/data-实际时间.zip'
$BackupDir = 'D:\cv-matting-backups\本次实例编号'
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
Push-Location $BackupDir
scp -P $SshPort "${SshTarget}:${RemoteArchive}" .
scp -P $SshPort "${SshTarget}:${RemoteArchive}.sha256" .
Get-FileHash (Split-Path $RemoteArchive -Leaf) -Algorithm SHA256
Pop-Location
```

本地 SHA256 必须与 `.sha256` 文件和服务器输出一致；再用同版 `scripts/server/portable.py verify --archive 数据ZIP路径` 检查 ZIP 内逐文件清单，抽查模型包、原片、背景和报告。把两类归档、各自校验文件、`machine-record.md`、诊断报告和本次性能记录放在同一备份目录。对真人素材按其授权范围保存和分享。仅有传输成功或文件大小相近，不代表备份已经可恢复。

## 换到下一台服务器时恢复

1. 在本地核对新 `server-deployment.zip` 和 `.sha256`；把部署包、共享内容 ZIP 及各自 `.sha256` 传到新服务器。用 `portable.py verify --archive` 校验归档。新机器上重新安装运行环境，不复制旧机器的 `.venv-server`。
2. 在新服务器的持久存储上建立新的根目录，将 `MATTE_SERVER_ROOT` 设为该绝对路径，按“新机器准备”使用 `server.sh install-archive` 安装源码与模型，并确认 `current`、`shared/weights` 和模型 `content_sha256`。安装脚本会执行真实 RVM 冒烟测试；若目标 GPU 或驱动不同，先调整并验证 Torch/CUDA 组合。随后停服务，通过 `server.sh restore 数据ZIP路径` 恢复数据；它会校验归档并避免覆盖内容不同的已有文件。以下 `/data` 上传命令仅适用于本次平台，换供应商要改成新机的持久目录：

   ```powershell
   $SshTarget = 'SSH_USER@SSH_HOST'
   $SshPort = 'SSH_PORT'
   $LocalArchive = 'D:\cv-matting-backups\本次实例编号\data-实际时间.zip'
   Get-FileHash $LocalArchive -Algorithm SHA256
   ssh -p $SshPort $SshTarget 'mkdir -p /data/coding/cv-human-video-matting/incoming'
   scp -P $SshPort $LocalArchive "${SshTarget}:/data/coding/cv-human-video-matting/incoming/"
   scp -P $SshPort "${LocalArchive}.sha256" "${SshTarget}:/data/coding/cv-human-video-matting/incoming/"
   ```

   ```bash
   ROOT=/data/coding/cv-human-video-matting
   bash "$ROOT/current/scripts/server/server.sh" stop
   bash "$ROOT/current/scripts/server/server.sh" restore "$ROOT/incoming/data-实际时间.zip"
   ```
3. 核对模型包 `content_sha256`、`shared/data/input/` 原片、`shared/data/backgrounds/` 背景、`shared/data/output/` 报告与视频及 `shared/logs/`。若自定义配置或其他文件单独备份，也恢复到本次记录的位置。
4. 重跑环境检查、实际 CUDA 推理、浏览器远程访问和短视频输出。新 GPU 上重新测 FPS 与延迟，不沿用旧机器数据。确认新备份保存路径后，再开始正式处理。

## 本次机器验收清单

- [ ] 记录 SSH 主机指纹、端口、系统、磁盘、`nvidia-smi` 和实际代码版本。
- [ ] `server.sh setup`（包含 `smoke`）通过；PyTorch 在 CUDA 上完成张量计算，模型包哈希与实际 RVM WebSocket 推理通过。
- [ ] 服务启动后在服务器本机可访问；从另一台电脑经本次 SSH 转发或已配置的 HTTPS/WSS 地址打开页面。
- [ ] 从另一台电脑用本地视频完成短时抠像，再测试浏览器摄像头授权、开始、停止、重连和页面刷新。
- [ ] 记录输入宽度、浏览器显示 FPS、往返延迟、GPU 占用、首帧等待及任何错误；当前实现只允许一个实时会话。
- [ ] 如使用批处理，核对输出视频、音轨和 `result.json`；有真人素材时核对其使用授权与实际画质。
- [ ] 新 `server-deployment.zip`、共享内容 ZIP 及各自 `.sha256` 已保存在自己的电脑；哈希一致、逐文件校验通过，并抽查模型包、素材和报告。

## 退租或关机前

1. 停止新任务，等待正在处理的视频和报告写完，再运行 `server.sh stop`；记下最后日志位置。
2. 用 `server.sh backup` 做最后一次共享内容备份并拉回本地，核对 SHA256、逐文件清单及关键文件；同时确认本地有同版新 `server-deployment.zip` 和校验文件。备份不能只存放在即将释放的服务器上。
3. 更新本次 `machine-record.md`：源码及部署包身份、模型身份、验证结果、备份时间与路径、故障及恢复办法。若远端有未提交的源码修改，另存补丁或重新制作部署包；共享内容备份包含模型实体，但不会保存源码修改。
4. 检查浏览器下载、录制文件和授权素材是否也需要纳入备份。确定下次可从备份恢复后，再按租赁平台流程释放实例。
5. 如要使用平台的临时保留功能，确认本次所有需保留文件确实在 `/data`，在释放实例时明确选择“保存数据”，记录平台显示的保留截止时间和可挂载集群；此操作不替代本地备份。

曾在聊天、工单或共享文档中发送过的服务器密码应在租赁平台更换；机器记录只写凭据的保管位置，不写凭据本身。
