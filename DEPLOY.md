# yolo_service — Deployment Guide

> 🌐 Language: **English** | [简体中文](DEPLOY.zh-CN.md)

## 1. Environment

| Item | Requirement |
|---|---|
| OS | Ubuntu 20.04+ (tested on 22.04) |
| GPU | NVIDIA with driver ≥ 535 (CUDA 12.x); CPU-only works too |
| Python | 3.10 (conda env recommended) |
| Packages | see `requirements.txt` — install torch cu121 wheels first |

## 2. Install from release package

```bash
tar -xzf yolo_service-1.2.0.tar.gz && cd yolo_service-1.2.0
sha256sum -c MANIFEST.sha256          # verify every file
bash deploy.sh                        # installs unit + logrotate, starts service
```

Manual install: copy the files anywhere writable, adjust
`systemd/yolo-service.service` paths, then:

```bash
sudo cp systemd/yolo-service.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now yolo-service
```

## 3. systemd unit essentials

```ini
[Service]
WorkingDirectory=/home/deployer/yolo_service
ExecStart=/home/deployer/miniconda3/envs/yolo11/bin/python app.py
Restart=always
RestartSec=5
StandardOutput=append:/home/deployer/yolo_service/service.log
```

The app also rotates its own log (`YS_LOG_MAX_BYTES`, default 10MB x 5);
logrotate with `copytruncate` is installed as a second safety net.

## 4. Docker

```bash
docker build -t yolo-service .
# CPU
docker run -d -p 1238:1238 --name yolo-service yolo-service
# GPU (needs nvidia-container-toolkit; cu121 wheels bundle the CUDA runtime)
docker run -d --gpus all -p 1238:1238 --name yolo-service yolo-service
```

## 5. Verification checklist

```bash
# 1) service
systemctl is-active yolo-service                       # active
# 2) health
curl -s http://127.0.0.1:1238/api/health | python3 -m json.tool
# 3) unit tests (offline, no GPU needed for cache tests)
python -m pytest tests/ -q
# 4) live smoke test (noise images, no real alerts triggered)
python tests/test_api.py
# 5) callback reachability (only if alerts expected)
curl -s -o /dev/null -w '%{http_code}' http://<ALERT_HOST>:8600/
```

## 6. Upgrade & rollback

```bash
cd /home/deployer/yolo_service
tar -xzf yolo_service-1.2.0.tar.gz -C /tmp && bash /tmp/yolo_service-1.2.0/deploy.sh
# deploy.sh backs up to backup/<ts>.tar.gz; roll back with:
bash deploy.sh --rollback backup/<ts>.tar.gz
```

Runtime dirs (`start/`, `results/`, `service.log`) are preserved on upgrade.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed to load model` | `YS_MODEL` path wrong or weights missing |
| First request slow (~1s) | warmup disabled? set `YS_WARMUP=1` |
| Frames arrive but never detected | dHash dedup hit — scene unchanged; verify with `YS_DEDUP_HASH_DIST=0` temporarily |
| No callback received | check `YS_CALLBACK_URL` reachability, `alarms` in `/api/stats`, log "callback failed" |
| `CUDA error` on startup | driver/runtime mismatch; `nvidia-smi` and reinstall torch cu121 wheels |
| Port already in use | `ss -lntp | grep 1238`; another instance or stale process |

## 8. Port map (reference deployment)

| Port | Service |
|---|---|
| 1237 | legacy Pedestron (retired) |
| 1238 | **yolo_service** (this project) |
| 1239 | yolo_web_compare detection console |

## 9. Security notes

- Bind behind a firewall / reverse proxy; the API has no auth by design
  (LAN-only deployment).
- `devices` is validated against `^[A-Za-z0-9_.\-]{1,64}$` (no path
  traversal); upload size capped by `YS_MAX_UPLOAD_MB` (20MB default).
