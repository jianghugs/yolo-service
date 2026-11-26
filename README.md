# yolo_service — Pedestrian Detection & Alert Service

> 🌐 Language: **English** | [简体中文](README.zh-CN.md)

A pedestrian detection service built on **Ultralytics YOLO11**, replacing the legacy
Pedestron (mmdet cascade_hrnet) implementation. The API is **fully compatible** with the
original `b.py` service — callers only need to change the port from `1237` to `1238`.

```
Camera / client --POST snapshot-->  yolo_service (1238)
                                      ├─ save image + per-device cache
                                      ├─ frame similarity dedup (0.99)
                                      ├─ YOLO11n pedestrian detection (person, conf>=0.5)
                                      └─ count > 0 -> base64 annotated image -> alert callback
```

## 1. Layout

```
yolo_service/
├── app.py                  # HTTP layer (Flask + waitress): routes, validation, stats
├── config.py               # config center, everything overridable via YS_* env vars
├── detector.py             # model singleton, serialized GPU inference, result plotting
├── cache.py                # snapshot cache, similarity dedup, TTL cleanup
├── callback.py             # alert callback (async + retry)
├── yolo_service.py         # compatibility entry point (same as app.py)
├── yolo11n.pt              # model weights (5.6MB)
├── requirements.txt        # dependencies (torch must be +cu121)
├── README.md               # this file
├── DEPLOY.md               # deployment & operations manual
├── CHANGELOG.md            # change log
├── VERSION                 # version string
├── start_yolo.sh           # manual start (emergency only)
├── deploy.sh               # one-shot deploy/upgrade (backup + rollback)
├── systemd/
│   └── yolo-service.service
├── tools/
│   ├── package.sh          # build dist/yolo_service-<ver>.tar.gz
│   ├── make_manifest.py    # generate MANIFEST.sha256
│   └── audit_files.py      # file integrity / suspicious-deletion audit
├── tests/
│   └── test_api.py         # API smoke test
├── start/                  # runtime: snapshot cache (auto-created)
└── results/                # runtime: annotated images (deleted right after callback)
```

## 2. API

### 2.1 `POST /api/portraitRecognition`

| Field | In | Description |
|---|---|---|
| `devices` | form | device ID, `^[A-Za-z0-9_.-]{1,64}$` |
| `file` | form | image (jpg/png/bmp, <= 20MB) |

Response (HTTP is always 200, business code in body):

| code | msg | Scenario |
|---|---|---|
| `400` | `图片不足两张` (fewer than two images) | first/only image for this device, waiting for the next one |
| `200` | `ok` | compared (changed -> detected; unchanged -> skipped by dedup) |
| `400` | invalid `devices` / missing `file` / undecodable image | bad request |
| `500` | `internal error` | server-side failure (see logs) |

### 2.2 `GET /api/clearPicture?devices=xxx`

Clears the snapshot cache of the given device,
returns `{"code":"200","msg":"ok","removed":N}`.

### 2.3 `GET /api/health`

```json
{"code":"200","msg":"ok","device":"cuda:0","model":"yolo11n",
 "version":"1.1.0","uptime_s":3600,"requests":1024,"alarms":3,
 "cache_files":1,"gpu":{"used_mb":812,"total_mb":6144}}
```

### 2.4 `GET /api/stats`

Request count, detections, dedup skips, alarms, errors, average inference time,
callback config, effective config.

> The alert callback payload is identical to the legacy service:
> `{"deviceId": "...", "count": N, "imageBase64": "..."}`
> -> `POST http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo`

## 3. Configuration (environment variables)

All defaults match the legacy behaviour — it runs out of the box with no configuration.

| Variable | Default | Description |
|---|---|---|
| `YS_PORT` | `1238` | listen port |
| `YS_HOST` | `0.0.0.0` | bind address |
| `YS_SERVER` | `waitress` | `waitress` (production) / `flask` (debug) |
| `YS_THREADS` | `8` | waitress thread count |
| `YS_MODEL` | `./yolo11n.pt` | weights path |
| `YS_DEVICE` | auto | `cuda:0` / `cpu` |
| `YS_SCORE_THR` | `0.5` | detection confidence threshold |
| `YS_DEDUP_THRESHOLD` | `0.99` | similarity threshold, >= means "frame unchanged" |
| `YS_CALLBACK_URL` | `http://<ALERT_HOST>:8600/...` | alert callback URL |
| `YS_CALLBACK_ASYNC` | `1` | async callback (does not block the request) |
| `YS_CALLBACK_TIMEOUT` | `10` | per-attempt callback timeout (seconds) |
| `YS_CALLBACK_RETRY` | `2` | retries on failure |
| `YS_START_DIR` / `YS_RESULTS_DIR` | `./start` / `./results` | cache / result dirs (use separate ones for canary instances) |
| `YS_IMAGE_TTL_HOURS` | `24` | snapshot retention |
| `YS_MAX_IMAGES_PER_DEVICE` | `20` | per-device cache file limit |
| `YS_CLEANUP_INTERVAL` | `3600` | periodic cleanup interval (seconds), 0 disables |
| `YS_LOG_FILE` | `./service.log` | log file (10MB x 5 rotated) |
| `YS_LOG_LEVEL` | `INFO` | log level |

When deploying with systemd, put config into `/home/deployer/yolo_service/yolo_service.env`,
one `KEY=VALUE` per line.

## 4. Quick start

```bash
# 1) dependencies (torch first, then ultralytics)
/home/deployer/miniconda3/envs/yolo11/bin/pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
/home/deployer/miniconda3/envs/yolo11/bin/pip install -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# 2) start (production)
sudo systemctl start yolo-service

# 3) verify
curl -s http://127.0.0.1:1238/api/health
/home/deployer/miniconda3/envs/yolo11/bin/python tests/test_api.py --url http://127.0.0.1:1238
```

Canary a new version without touching the live 1238 service:

```bash
YS_PORT=1240 YS_START_DIR=./start_1240 YS_RESULTS_DIR=./results_1240 \
YS_LOG_FILE=./service_1240.log python app.py
```

## 5. Packaging & distribution

```bash
bash tools/package.sh
# dist/yolo_service-1.1.0.tar.gz
# dist/yolo_service-1.1.0.tar.gz.sha256
```

The package contains `MANIFEST.sha256`. On the target machine:

```bash
sha256sum -c yolo_service-1.1.0.tar.gz.sha256   # 1) verify the package
tar -xzf yolo_service-1.1.0.tar.gz && cd yolo_service-1.1.0
bash deploy.sh                                   # 2) deploy (auto backup, health check, rollback)
```

## 6. File integrity / suspicious-deletion audit

```bash
# verify project files against MANIFEST.sha256
python3 tools/audit_files.py

# also scan the legacy business dir for "deleted but still held open" files
python3 tools/audit_files.py --extra /home/deployer/Pedestron

# only show deleted-but-open handles
python3 tools/audit_files.py --deleted-only
```

The audit reports three classes: `MISSING` (in manifest, not on disk), `MODIFIED`
(size/hash mismatch), `EXTRA` (not in manifest), plus `(deleted)` handles under
`/proc/*/fd` (these still consume disk space although invisible in the directory).

## 7. Operations

```bash
sudo systemctl status  yolo-service
sudo systemctl restart yolo-service
sudo journalctl -u yolo-service -n 100 --no-pager -f
tail -f /home/deployer/yolo_service/service.log
```

- start on boot: `systemctl is-enabled yolo-service` -> `enabled`
- crash recovery: `Restart=always`, `RestartSec=5`
- do not run `start_yolo.sh` manually alongside systemd (port conflict; emergency only)

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| exits immediately on start | `journalctl -u yolo-service -n 50`; usually missing weights or torch/cuDNN mismatch |
| `cuda: False` | `pip show torch` must be `2.5.1+cu121`; if upgraded to `+cu13*`, reinstall from the local whl |
| `CUDNN_STATUS_NOT_INITIALIZED` | cu13 libs mixed into `site-packages/nvidia`; clear that dir and reinstall cu12 torch |
| always "fewer than two images" | device has fewer than 2 images yet; or `clearPicture` is called too often |
| frames arrive but never detected | similarity to the previous frame >= 0.99, normal dedup; temporarily set `YS_DEDUP_THRESHOLD=0.995` to verify |
| people detected but no callback | check `YS_CALLBACK_URL` reachability, `alarms` counter in `/api/stats`, log lines "callback failed" |
| memory / VRAM keeps growing | `curl /api/health` and watch `gpu.used_mb`; inference is serialized behind a lock, it should stay flat |
