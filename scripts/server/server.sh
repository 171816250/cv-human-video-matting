#!/usr/bin/env bash
# Linux GPU realtime deployment control. Run with bash; no systemd is required.
set -Eeuo pipefail
umask 077

APP_HOME="${MATTE_SERVER_ROOT:-/data/coding/cv-human-video-matting}"
VENV="${MATTE_VENV:-$APP_HOME/.venv-server}"
PYTHON_BOOTSTRAP="${MATTE_PYTHON:-python3.11}"
TORCH_VERSION="${MATTE_TORCH_VERSION:-2.10.0}"
VISION_VERSION="${MATTE_VISION_VERSION:-0.25.0}"
CUDA_CHANNEL="${MATTE_CUDA_CHANNEL:-cu128}"
PORT="${MATTE_PORT:-7861}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PORTABLE="$SCRIPT_DIR/portable.py"
MODEL_DEST="weights/portable-rvm"
PID_FILE="$APP_HOME/run/realtime.pid"

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
note() { printf '%s\n' "$*"; }

source_dir() {
  if [[ -n "${MATTE_SOURCE_DIR:-}" ]]; then printf '%s\n' "$MATTE_SOURCE_DIR";
  else printf '%s\n' "$APP_HOME/current"; fi
}

model_dir() {
  if [[ -n "${MATTE_MODEL_DIR:-}" ]]; then printf '%s\n' "$MATTE_MODEL_DIR";
  elif [[ -d "$APP_HOME/shared/weights/current-rvm" ]]; then
    printf '%s\n' "$APP_HOME/shared/weights/current-rvm"
  else printf '%s\n' "$APP_HOME/shared/weights/offline-rvm-mobilenetv3-rc2"; fi
}

validate_paths() {
  [[ "$APP_HOME" == /* && "$APP_HOME" != / ]] || die "Server root must be an absolute directory other than /: $APP_HOME"
  [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || die "Invalid port: $PORT"
  [[ "$TORCH_VERSION" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]] || die "Invalid Torch version"
  [[ "$VISION_VERSION" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]] || die "Invalid TorchVision version"
  [[ "$CUDA_CHANNEL" =~ ^cu[0-9]+$ ]] || die "Invalid CUDA channel"
}

python_from_env() { [[ -x "$VENV/bin/python" ]] || die "Python environment missing: $VENV"; }

make_env() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    command -v "$PYTHON_BOOTSTRAP" >/dev/null || die "Python 3.11 not found: $PYTHON_BOOTSTRAP"
    "$PYTHON_BOOTSTRAP" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' || die "Use Python 3.11"
    mkdir -p "$APP_HOME"
    "$PYTHON_BOOTSTRAP" -m venv "$VENV"
  fi
  python_from_env
  "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' || die "Environment must use Python 3.11"
}

install_packages() {
  local src="$1" wheelhouse="${MATTE_WHEELHOUSE:-}"
  local -a pip_flags=()
  if [[ -n "$wheelhouse" ]]; then
    [[ -d "$wheelhouse" ]] || die "Wheelhouse missing: $wheelhouse"
    if [[ -f "$wheelhouse/SHA256SUMS" ]]; then (cd "$wheelhouse" && sha256sum -c SHA256SUMS); fi
    pip_flags=(--no-index --find-links "$wheelhouse")
  fi
  if ! "$VENV/bin/python" - "$TORCH_VERSION" "$VISION_VERSION" "$CUDA_CHANNEL" <<'PY'
import sys
try:
    import torch, torchvision
    assert torch.__version__.startswith(sys.argv[1] + "+" + sys.argv[3]), torch.__version__
    assert torchvision.__version__.startswith(sys.argv[2] + "+" + sys.argv[3]), torchvision.__version__
except Exception as exc:
    print(f"Installing matching CUDA Torch pair: {exc}")
    raise SystemExit(1)
PY
  then
    if [[ -n "$wheelhouse" ]]; then
      "$VENV/bin/python" -m pip install --no-cache-dir "${pip_flags[@]}" \
        "torch==${TORCH_VERSION}+${CUDA_CHANNEL}" "torchvision==${VISION_VERSION}+${CUDA_CHANNEL}"
    else
      "$VENV/bin/python" -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${VISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"
    fi
  fi
  "$VENV/bin/python" -m pip install --no-cache-dir "${pip_flags[@]}" \
    -r "$src/requirements-server-realtime.txt"
  "$VENV/bin/python" -m pip check
  "$VENV/bin/python" - "$TORCH_VERSION" "$VISION_VERSION" "$CUDA_CHANNEL" <<'PY'
import sys
import torch, torchvision
assert torch.__version__.startswith(sys.argv[1] + "+" + sys.argv[3]), torch.__version__
assert torchvision.__version__.startswith(sys.argv[2] + "+" + sys.argv[3]), torchvision.__version__
print("Torch", torch.__version__, "TorchVision", torchvision.__version__)
PY
}

check_model() {
  local src="$1" model="$2"
  [[ -f "$src/app.py" ]] || die "Source missing: $src/app.py"
  [[ -f "$src/requirements-server-realtime.txt" ]] || die "Server requirements missing; use the current source"
  [[ -f "$model/manifest.json" ]] || die "Offline model missing: $model"
  PYTHONPATH="$src/src" "$VENV/bin/python" - "$model" <<'PY'
import sys
from cv_human_video_matting.utils.model_bundle import verify_model_bundle
bundle = verify_model_bundle(sys.argv[1])
print("RVM model SHA256:", bundle["content_sha256"])
PY
}

cuda_tensor_check() {
  "$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
device = torch.device("cuda:0")
tensor = torch.ones((128, 128), device=device)
result = tensor @ tensor
torch.cuda.synchronize()
assert torch.isfinite(result).all().item() and result[0, 0].item() == 128.0
print("CUDA tensor check:", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
PY
}

smoke() {
  python_from_env
  local src model output
  src="$(source_dir)"
  model="$(model_dir)"
  check_model "$src" "$model"
  cuda_tensor_check
  mkdir -p "$APP_HOME/shared/data/output" "$APP_HOME/shared/logs"
  output="$APP_HOME/shared/data/output/server-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  (cd "$src" && "$VENV/bin/python" -B scripts/verify_realtime.py \
    --bundle "$model" --output-dir "$output" --device cuda \
    --frames 120 --warmup 10 --width 320)
  note "RVM WebSocket inference report: $output/rehearsal.json"
}

is_ours() {
  local pid="$1" src="$2" command_line
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command_line" == *"$src/app.py realtime"* ]] || return 1
}

is_listening_loopback() {
  "$VENV/bin/python" - "$PORT" <<'PY'
import sys
port = int(sys.argv[1])
seen = []
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(table, encoding="ascii").read().splitlines()[1:]
    except FileNotFoundError:
        continue
    for line in lines:
        fields = line.split()
        address, value = fields[1].rsplit(":", 1)
        if int(value, 16) == port and fields[3] == "0A":
            seen.append((table, address))
assert seen, f"No listener on port {port}"
assert all(table.endswith("tcp") and address == "0100007F" for table, address in seen), seen
print(f"Port {port} listens only on 127.0.0.1")
PY
}

start_server() {
  python_from_env
  local src model pid logfile
  src="$(source_dir)"
  model="$(model_dir)"
  check_model "$src" "$model"
  cuda_tensor_check
  mkdir -p "$APP_HOME/run" "$APP_HOME/shared/logs"
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if is_ours "$pid" "$src"; then die "Realtime server already running as PID $pid"; fi
    if [[ "$pid" =~ ^[0-9]+$ && ! -e "/proc/$pid" ]]; then
      rm -- "$PID_FILE"
      note "Removed stale PID file for exited process $pid"
    else
      die "Stale/unrecognized PID file: $PID_FILE; inspect it before removal"
    fi
  fi
  logfile="$APP_HOME/shared/logs/realtime-$(date -u +%Y%m%dT%H%M%SZ).log"
  (
    cd "$src"
    nohup "$VENV/bin/python" -u -B "$src/app.py" realtime \
      --model-bundle "$model" --device cuda --host 127.0.0.1 --port "$PORT" \
      >"$logfile" 2>&1 </dev/null &
    echo $! > "$PID_FILE"
  )
  pid="$(cat "$PID_FILE")"
  for _ in {1..30}; do
    if ! is_ours "$pid" "$src"; then die "Realtime server exited; see $logfile"; fi
    if "$VENV/bin/python" - "$PORT" <<'PY' >/dev/null 2>&1
import sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{int(sys.argv[1])}/", timeout=1) as response:
    assert response.status == 200
PY
    then
      is_listening_loopback
      note "Realtime server PID $pid; http://127.0.0.1:$PORT; log $logfile"
      return
    fi
    sleep 1
  done
  die "Realtime server did not become ready; see $logfile"
}

stop_server() {
  [[ -f "$PID_FILE" ]] || { note "Realtime server is not running (no PID file)"; return; }
  local pid src
  pid="$(cat "$PID_FILE")"
  src="$(source_dir)"
  is_ours "$pid" "$src" || die "PID $pid is not this realtime server; will not kill it"
  kill -TERM "$pid"
  for _ in {1..15}; do
    if ! kill -0 "$pid" 2>/dev/null; then rm -- "$PID_FILE"; note "Stopped PID $pid"; return; fi
    sleep 1
  done
  die "PID $pid did not stop after SIGTERM; inspect it manually"
}

status_server() {
  [[ -f "$PID_FILE" ]] || { note "Stopped"; return 1; }
  local pid src
  pid="$(cat "$PID_FILE")"; src="$(source_dir)"
  is_ours "$pid" "$src" || die "PID file is stale or belongs to another process: $PID_FILE"
  is_listening_loopback
  note "Running PID $pid on 127.0.0.1:$PORT"
}

install_archive() {
  [[ $# -eq 2 ]] || die "Usage: server.sh install-archive deployment.zip release-name"
  local archive="$1" release="$2" src model temporary_link
  [[ "$release" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "Invalid release name"
  [[ ! -f "$PID_FILE" ]] || die "Stop the realtime server before installing a release"
  if [[ -e "$APP_HOME/current" && ! -L "$APP_HOME/current" ]]; then
    die "Current release path is not a symlink; refusing to replace it"
  fi
  if [[ -e "$APP_HOME/shared/weights/current-rvm" && ! -L "$APP_HOME/shared/weights/current-rvm" ]]; then
    die "Current model path is not a symlink; refusing to replace it"
  fi
  src="$APP_HOME/releases/$release/source"
  [[ ! -e "$src" && ! -L "$src" ]] || die "Release already exists: $src"
  command -v python3 >/dev/null || die "Python 3 is needed to verify/extract the archive"
  python3 "$PORTABLE" extract --archive "$archive" --target "$src"
  model="$APP_HOME/shared/weights/$release-rvm"
  [[ -d "$src/$MODEL_DEST" ]] || die "Archive model directory missing"
  mkdir -p "$APP_HOME/shared/weights" "$APP_HOME/shared/data/input" \
    "$APP_HOME/shared/data/backgrounds" "$APP_HOME/shared/data/output" "$APP_HOME/shared/logs"
  [[ ! -e "$model" && ! -L "$model" ]] || die "Model destination already exists: $model"
  mv -- "$src/$MODEL_DEST" "$model"
  ln -s -- "$model" "$src/$MODEL_DEST"
  [[ ! -e "$src/data" ]] && ln -s -- "$APP_HOME/shared/data" "$src/data"
  [[ ! -e "$src/logs" ]] && ln -s -- "$APP_HOME/shared/logs" "$src/logs"
  MATTE_SOURCE_DIR="$src" MATTE_MODEL_DIR="$model" setup_server
  temporary_link="$APP_HOME/.current-$$"
  ln -s -- "$src" "$temporary_link"
  mv -Tf -- "$temporary_link" "$APP_HOME/current"
  temporary_link="$APP_HOME/shared/weights/.current-rvm-$$"
  ln -s -- "$model" "$temporary_link"
  mv -Tf -- "$temporary_link" "$APP_HOME/shared/weights/current-rvm"
  note "Installed and verified release $release"
}

setup_server() {
  local src
  src="$(source_dir)"
  [[ -f "$src/requirements-server-realtime.txt" ]] || die "Server requirements missing: $src"
  make_env
  install_packages "$src"
  check_model "$src" "$(model_dir)"
  cuda_tensor_check
  smoke
  note "Dependencies, CUDA tensor and actual RVM WebSocket inference verified."
}

download_wheels() {
  [[ $# -eq 1 ]] || die "Usage: server.sh wheels NEW_DIRECTORY"
  python_from_env
  local target="$1" src
  [[ ! -e "$target" ]] || die "Wheelhouse target already exists: $target"
  mkdir -p "$target"
  src="$(source_dir)"
  "$VENV/bin/python" -m pip download --only-binary :all: --dest "$target" \
    "torch==${TORCH_VERSION}" "torchvision==${VISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"
  "$VENV/bin/python" -m pip download --only-binary :all: --dest "$target" \
    -r "$src/requirements-server-realtime.txt"
  (cd "$target" && sha256sum ./*.whl > SHA256SUMS)
  note "Linux wheelhouse ready: $target (copy the entire directory and SHA256SUMS)"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/server/server.sh COMMAND [ARGUMENTS]
  setup                 Install only into .venv-server; verify CUDA tensor and RVM WebSocket inference
  smoke                 Run 120-frame real RVM WebSocket GPU rehearsal
  start | stop | status Start, stop or inspect the loopback-only service
  backup FILE.zip       Back up shared/weights, shared/data and shared/logs, with SHA256 sidecar
  restore FILE.zip      Restore backup without overwriting different files
  wheels NEW_DIRECTORY  Download matching Linux wheelhouse for a future server
  install-archive FILE.zip RELEASE_NAME  Extract, set up and validate a fresh release

Defaults: root=/data/coding/cv-human-video-matting, Python=3.11,
Torch=2.10.0, TorchVision=0.25.0, CUDA=cu128, port=7861.
Override with MATTE_SERVER_ROOT, MATTE_SOURCE_DIR, MATTE_MODEL_DIR,
MATTE_VENV, MATTE_PYTHON, MATTE_TORCH_VERSION, MATTE_VISION_VERSION,
MATTE_CUDA_CHANNEL, MATTE_PORT, MATTE_WHEELHOUSE.
EOF
}

main() {
  validate_paths
  local command="${1:-help}"
  shift || true
  case "$command" in
    setup) [[ $# -eq 0 ]] || die "Unexpected argument"; setup_server ;;
    smoke) [[ $# -eq 0 ]] || die "Unexpected argument"; smoke ;;
    start) [[ $# -eq 0 ]] || die "Unexpected argument"; start_server ;;
    stop) [[ $# -eq 0 ]] || die "Unexpected argument"; stop_server ;;
    status) [[ $# -eq 0 ]] || die "Unexpected argument"; status_server ;;
    backup)
      [[ $# -eq 1 ]] || die "Usage: server.sh backup NEW_FILE.zip"
      python3 "$PORTABLE" backup --root "$APP_HOME/shared" --output "$1" ;;
    restore)
      [[ $# -eq 1 ]] || die "Usage: server.sh restore FILE.zip"
      [[ ! -f "$PID_FILE" ]] || die "Stop the realtime server before restoring data"
      mkdir -p "$APP_HOME/shared"
      python3 "$PORTABLE" restore --archive "$1" --root "$APP_HOME/shared" ;;
    wheels) download_wheels "$@" ;;
    install-archive) install_archive "$@" ;;
    help|-h|--help) usage ;;
    *) usage; die "Unknown command: $command" ;;
  esac
}

main "$@"
