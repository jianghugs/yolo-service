#!/bin/bash
# 打包脚本：生成 dist/yolo_service-<version>.tar.gz（含 sha256 校验与文件清单）
#
# 用法：
#   bash tools/package.sh
#   YS_PYTHON=/usr/bin/python3 bash tools/package.sh
set -euo pipefail

SRC=$(cd "$(dirname "$0")/.." && pwd)
cd "$SRC"

VER=$(cat VERSION 2>/dev/null | tr -d '[:space:]')
VER=${VER:-0.0.0}
NAME="yolo_service-${VER}"
OUT="$SRC/dist"
STAGE=$(mktemp -d)
PKG="$STAGE/$NAME"
PY=${YS_PYTHON:-python3}

cleanup() {
    [ -d "$STAGE" ] && "$PY" -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$STAGE"
}
trap cleanup EXIT

echo "[package] 版本 $VER"
mkdir -p "$OUT" "$PKG"

# ---------- 代码与文档 ----------
for f in app.py config.py detector.py cache.py callback.py yolo_service.py \
         requirements.txt README.md DEPLOY.md CHANGELOG.md VERSION \
         start_yolo.sh deploy.sh .gitignore; do
    [ -f "$f" ] && cp -f "$f" "$PKG/"
done
mkdir -p "$PKG/tools" "$PKG/tests" "$PKG/systemd"
cp -f tools/*.py tools/*.sh "$PKG/tools/" 2>/dev/null || true
cp -f tests/*.py "$PKG/tests/" 2>/dev/null || true
cp -f systemd/*.service "$PKG/systemd/" 2>/dev/null || true
chmod +x "$PKG"/start_yolo.sh "$PKG"/deploy.sh "$PKG"/tools/*.sh 2>/dev/null || true

# ---------- 模型权重 ----------
if [ -f yolo11n.pt ]; then
    cp -f yolo11n.pt "$PKG/"
    echo "[package] 已包含模型权重 yolo11n.pt"
else
    echo "[package] 警告：未找到 yolo11n.pt，包内不含权重（部署后需单独下载）"
fi

# ---------- 文件清单 ----------
"$PY" tools/make_manifest.py "$PKG" -o "$PKG/MANIFEST.sha256"

# ---------- 打 tar.gz ----------
TARBALL="$OUT/$NAME.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"

# ---------- zip（便于 Windows 解压） ----------
if command -v zip >/dev/null 2>&1; then
    (cd "$STAGE" && zip -qr "$OUT/$NAME.zip" "$NAME")
fi

# ---------- 外层校验文件 ----------
(
    cd "$OUT"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256"
    else
        "$PY" - <<PYEOF
import hashlib, io
p = "$NAME.tar.gz"
h = hashlib.sha256()
with open(p, 'rb') as f:
    for c in iter(lambda: f.read(1 << 20), b''):
        h.update(c)
io.open(p + '.sha256', 'w', newline='\n').write('%s  %s\n' % (h.hexdigest(), p))
PYEOF
    fi
)

echo
echo "[package] 完成："
ls -lh "$OUT" | sed 's/^/    /'
echo
echo "分发后校验：sha256sum -c $NAME.tar.gz.sha256"
echo "解包部署：tar -xzf $NAME.tar.gz && cd $NAME && bash deploy.sh"
