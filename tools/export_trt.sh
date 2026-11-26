#!/usr/bin/env bash
# 导出 TensorRT engine 加速推理（可选优化，需 GPU + tensorrt 环境）。
# 用法：bash tools/export_trt.sh   → 生成 yolo11n.engine
# 之后设置 YS_MODEL=yolo11n.engine（或 WC_DETECT_VIA 保持默认即可）重启服务。
#
# 注意：engine 与显卡强绑定，换卡需要重新导出。
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'EOF'
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='engine', half=True, device=0, imgsz=640)
print('export done -> yolo11n.engine')
EOF
