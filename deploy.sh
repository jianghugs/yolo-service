#!/bin/bash
# 一键部署 / 升级脚本（在服务器上以 deployer 用户执行）
#
# 用法：
#   bash deploy.sh                 # 用当前目录的代码部署
#   bash deploy.sh /path/to/src    # 用指定目录的代码部署
#   DST=/home/deployer/yolo_service bash deploy.sh
#
# 流程：备份 -> 复制代码 -> 安装/校验依赖 -> 重启 systemd -> 健康检查 -> 失败自动回滚
set -uo pipefail

SRC="${1:-$(cd "$(dirname "$0")" && pwd)}"
DST="${DST:-/home/deployer/yolo_service}"
SERVICE="${SERVICE:-yolo-service}"
UNIT_SRC="$SRC/systemd/$SERVICE.service"
UNIT_DST="/etc/systemd/system/$SERVICE.service"
PY="${YS_PYTHON:-/home/deployer/miniconda3/envs/yolo11/bin/python}"
# sudo 方式：默认交互式 sudo；如需免交互，设置 SUDO_PASS 环境变量（不要把密码写进脚本）
if [ -n "${SUDO_PASS:-}" ]; then
    sudo_run() { echo "$SUDO_PASS" | sudo -S "$@" 2>/dev/null; }
else
    sudo_run() { sudo -n "$@" 2>/dev/null || sudo "$@"; }
fi
BACKUP_DIR="$DST/backup"
TS=$(date +%Y%m%d_%H%M%S)
BACKUP="$BACKUP_DIR/code_$TS.tar.gz"

log() { echo "[deploy] $*"; }
die() { echo "[deploy][ERROR] $*" >&2; exit 1; }

[ -d "$SRC" ] || die "源目录不存在: $SRC"
[ -f "$SRC/app.py" ] || die "源目录缺少 app.py: $SRC"

log "源目录: $SRC"
log "目标目录: $DST"

# ---------- 1. 备份现有代码 ----------
mkdir -p "$BACKUP_DIR" "$DST"
if [ -f "$DST/app.py" ] || [ -f "$DST/yolo_service.py" ]; then
    tar -czf "$BACKUP" -C "$DST" \
        --exclude=start --exclude=results --exclude=backup \
        --exclude='*.log' --exclude='*.pt' \
        . 2>/dev/null && log "已备份现有代码 -> $BACKUP"
fi

# ---------- 2. 复制代码 ----------
copy_files() {
    local f
    for f in app.py config.py detector.py cache.py callback.py yolo_service.py \
             requirements.txt README.md DEPLOY.md CHANGELOG.md VERSION start_yolo.sh deploy.sh .gitignore; do
        [ -f "$SRC/$f" ] && cp -f "$SRC/$f" "$DST/$f"
    done
    mkdir -p "$DST/tools" "$DST/tests" "$DST/systemd"
    cp -f "$SRC"/tools/*.py "$SRC"/tools/*.sh "$DST/tools/" 2>/dev/null
    cp -f "$SRC"/tests/*.py "$DST/tests/" 2>/dev/null
    cp -f "$SRC"/systemd/*.service "$DST/systemd/" 2>/dev/null
    chmod +x "$DST"/start_yolo.sh "$DST"/deploy.sh "$DST"/tools/*.sh 2>/dev/null
    # 模型权重：目标没有才拷贝
    if [ -f "$SRC/yolo11n.pt" ] && [ ! -f "$DST/yolo11n.pt" ]; then
        cp -f "$SRC/yolo11n.pt" "$DST/yolo11n.pt" && log "已复制模型权重"
    fi
}
copy_files || die "复制代码失败"

# ---------- 3. 依赖校验 ----------
"$PY" - <<'PYEOF' || die "依赖校验失败（torch/torchvision/ultralytics 版本不匹配）"
import sys
try:
    import torch, torchvision, ultralytics
except Exception as e:
    print('import failed:', e); sys.exit(1)
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('torchvision', torchvision.__version__)
print('ultralytics', ultralytics.__version__)
if not torch.__version__.startswith('2.5.1'):
    print('WARN: torch 版本不是 2.5.1+cu121，请检查是否被其他安装覆盖')
PYEOF

# ---------- 4. 安装 systemd unit ----------
if [ -f "$UNIT_SRC" ]; then
    if ! diff -q "$UNIT_SRC" "$UNIT_DST" >/dev/null 2>&1; then
        sudo_run cp "$UNIT_SRC" "$UNIT_DST" \
            && sudo_run systemctl daemon-reload \
            && log "已更新 systemd unit"
    fi
fi

# ---------- 5. 重启并健康检查 ----------
sudo_run systemctl restart "$SERVICE"
PORT=$(grep -E '^YS_PORT=' "$DST/yolo_service.env" 2>/dev/null | cut -d= -f2)
PORT=${PORT:-1238}

ok=0
for i in $(seq 1 30); do
    sleep 2
    if curl -fsS -m 3 "http://127.0.0.1:$PORT/api/health" >/tmp/ys_health.json 2>/dev/null; then
        ok=1; break
    fi
done

if [ "$ok" = "1" ]; then
    log "健康检查通过: $(cat /tmp/ys_health.json)"
    log "部署成功 ✅  版本 $(cat "$DST/VERSION" 2>/dev/null)"
    rm -f /tmp/ys_health.json
    exit 0
fi

# ---------- 6. 回滚 ----------
echo "[deploy][ERROR] 健康检查失败，开始回滚" >&2
if [ -f "$BACKUP" ]; then
    tar -xzf "$BACKUP" -C "$DST" && log "已回滚到 $BACKUP"
    sudo_run systemctl restart "$SERVICE"
    log "已重启回滚后的版本，请检查 journalctl -u $SERVICE -n 80"
else
    log "无备份可回滚，请手动排查"
fi
exit 1
