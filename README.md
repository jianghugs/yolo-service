# yolo_service — Pedestrian Detection & Alert Service

> 🌐 Language: **English** | [简体中文](README.zh-CN.md)

Drop-in replacement for a legacy Pedestron (mmdet cascade_hrnet) pedestrian
detection service, built on **Ultralytics YOLO11**. The HTTP contract is
byte-compatible with the old `b.py` endpoint — callers only need to switch
the port from `1237` to `1238`.

```
camera / client --POST snapshot--> yolo_service (1238)
                                     ├─ save image + per-device cache
                                     ├─ dedup via dHash perceptual hash
                                     ├─ YOLO11n person detection (conf >= 0.5)
                                     └─ count > 0 → base64 result → alert callback
```

## 1. Highlights (v1.2.0)

- **Single model instance** loaded once at startup (the legacy code re-loaded
  the model on every request: ~4s + repeated VRAM churn)
- **dHash dedup** (microseconds, mtime-cached) replaces 1920x1080
  `matchTemplate` (~tens of ms)
- **Exact per-device ownership**: cache files are matched by the
  `<timestamp>_<devices>.<ext>` name pattern — device `1` no longer matches
  device `21` or `camera1`
- **Per-device locks** serialize cache read → compare → prune, removing
  request races
- **Zero-detection fast path**: when no person is found, no annotated image
  is encoded or written
- **Optional TensorRT**: run `tools/export_trt.sh` to export
  `yolo11n.engine` and point `YS_MODEL` at it (2-3x faster inference)

## 2. Layout

```
yolo_service/
├── app.py            # HTTP layer (Flask + waitress)
├── config.py         # all knobs, overridable via YS_* env vars
├── detector.py       # model singleton + GPU lock + result image
├── cache.py          # snapshot cache + dHash scene-change detection
├── callback.py       # async alert callback with retry
├── yolo11n.pt        # model weights (also supports .engine)
├── tests/            # pytest unit tests + live smoke test
├── tools/            # packaging / audit / TensorRT export
├── systemd/          # unit file + logrotate config
└── deploy.sh         # one-shot deploy: backup → overwrite → verify
```

## 3. API

### 3.1 `POST /api/portraitRecognition`

Form fields: `devices` (device ID, `[A-Za-z0-9_.\-]{1,64}`), `file` (image).

```
{"code": "200", "msg": "ok"}                       # processed (or dedup-skipped)
{"code": "400", "msg": "devices 参数缺失或非法"}     # invalid device
{"code": "400", "msg": "图片不足两张"}               # first frame of a device
```

When 2+ frames exist for a device, the new frame is compared with the
previous one via dHash; identical scenes skip inference. When the scene
changed and at least one person is detected, an alert is POSTed to the
callback URL.

### 3.2 `POST /api/detectPreview`  *(new in v1.2.0)*

Side-effect-free preview: no disk writes, no dedup, no callback. Returns
`{code, device, count, elapsed_ms, image(base64)}`. Used by the detection
web console (`yolo_web_compare`) to share this service's GPU via
`WC_DETECT_VIA`.

### 3.3 `GET /api/clearPicture?devices=xxx`

Removes all cached frames of a device. Returns `removed` count.

### 3.4 `GET /api/health`

```json
{"code":"200","device":"cuda:0","model":"yolo11n","version":"1.2.0",
 "uptime_s":86400,"requests":123,"alarms":45,"cache_files":2,
 "gpu":{"used_mb":2495,"total_mb":5931}}
```

### 3.5 `GET /api/stats`

Counters (`requests / detections / skipped / alarms / errors`), average
inference time, callback stats and effective config.

### 3.6 Alert callback

```
POST http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo
{"deviceId": "...", "count": 2, "imageBase64": "..."}
```

Async by default (thread pool of 4), 2 retries with backoff on failure.

## 4. Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `YS_PORT` | `1238` | listen port |
| `YS_THREADS` | `8` | waitress threads |
| `YS_MODEL` | `./yolo11n.pt` | weights path (`.pt` or `.engine`) |
| `YS_DEVICE` | auto | `cuda:0` / `cpu` |
| `YS_SCORE_THR` | `0.5` | person confidence threshold |
| `YS_DEDUP_HASH_SIZE` | `16` | dHash edge size (bits = size²) |
| `YS_DEDUP_HASH_DIST` | `10` | hamming distance ≤ ⇒ "scene unchanged" |
| `YS_CALLBACK_URL` | `http://<ALERT_HOST>:8600/...` | alert endpoint |
| `YS_CALLBACK_ASYNC` | `1` | async callback (0 = blocking) |
| `YS_CALLBACK_RETRY` | `2` | retries after the first attempt |
| `YS_START_DIR` | `./start` | snapshot cache dir |
| `YS_MAX_IMAGES_PER_DEVICE` | `20` | per-device cache cap |
| `YS_IMAGE_TTL_HOURS` | `24` | snapshot TTL |
| `YS_CLEANUP_INTERVAL` | `3600` | cleanup loop seconds (0 = off) |

## 5. Quick start

```bash
pip install -r requirements.txt          # torch cu121 first, see file header
python app.py                            # waitress on 0.0.0.0:1238

# Docker
docker build -t yolo-service .
docker run -d --gpus all -p 1238:1238 yolo-service

# verify
curl -s http://127.0.0.1:1238/api/health
python tests/test_api.py                 # live smoke test (noise images)
python -m pytest tests/ -q               # unit tests
```

## 6. systemd (production)

```
/etc/systemd/system/yolo-service.service   # see systemd/ in the repo
/etc/logrotate.d/pedestron-legacy          # weekly, compress, copytruncate
```

## 7. Packaging

```bash
bash tools/package.sh    # → dist/yolo_service-<ver>.tar.gz + .sha256 + MANIFEST
```

## 8. License

MIT — see [LICENSE](LICENSE).
