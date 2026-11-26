#!/bin/bash
# 手动启动脚本（应急用）
# 生产环境由 systemd 管理：sudo systemctl {start|stop|restart|status} yolo-service
# 注意：systemd 正在运行时不要执行本脚本，会因端口占用启动失败。
set -e
cd "$(dirname "$0")"

PY=${YS_PYTHON:-/home/deployer/miniconda3/envs/yolo11/bin/python}
if [ ! -x "$PY" ]; then
    PY=$(command -v python3)
fi

echo "[start_yolo] python=$PY dir=$(pwd)"
exec "$PY" -u app.py
