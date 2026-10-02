param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9.-]+$')]
    [string]$ServerHost,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$SshPort,

    [ValidatePattern('^[A-Za-z_][A-Za-z0-9_-]*$')]
    [string]$SshUser = 'root',

    [ValidateRange(1, 65535)]
    [int]$LocalPort = 7861,

    [ValidateRange(1, 65535)]
    [int]$RemotePort = 7861
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw 'OpenSSH 客户端未安装或不在 PATH 中。'
}

$forward = '127.0.0.1:{0}:127.0.0.1:{1}' -f $LocalPort, $RemotePort
$destination = '{0}@{1}' -f $SshUser, $ServerHost
$url = 'http://127.0.0.1:{0}/' -f $LocalPort

Write-Host "请先核对租赁平台显示的 SSH 主机指纹，再接受首次连接提示。"
Write-Host "隧道建立后，在本机浏览器打开 $url"
Write-Host '保持此窗口开启；关闭窗口即断开远程预览。密码仅在 SSH 提示时输入，不会保存在脚本中。'

& ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=3 -o StrictHostKeyChecking=ask `
    -L $forward -p $SshPort $destination

if ($LASTEXITCODE -ne 0) {
    throw "SSH 隧道退出，状态码 $LASTEXITCODE。请检查地址、端口、主机指纹和服务器服务状态。"
}
