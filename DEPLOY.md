# YOLO Pedestrian Detection Service — Deployment & Operations Manual

> 🌐 Language: **English** | [简体中文](DEPLOY.zh-CN.md)

> Service: `yolo_service` (Ultralytics YOLO11 pedestrian detection, replacing the legacy
> Pedestron/mmdet service)
> Version: **v1.1.0** | Updated: 2026-10-22
> Code location (server): `/home/deployer/yolo_service/` | Port: **1238**

---

## 1. Overview

### 1.1 Background

The legacy service was built on [Pedestron](https://github.com/mightyCelu/Pedestron)
(mmdetection 2.x, unmaintained since 2022) and had problems:

- every request re-ran `init_detector` and reloaded the HRNet model (seconds of latency +
  repeated VRAM alloc/free); long-running memory grew past 8GB;
- depends on mmdet/mmcv 2.x, poorly compatible with newer CUDA/PyTorch;
- cascade HRNet inference is slow (measured ~4.9s per call).

The new service uses **Ultralytics YOLO11n**: the model is loaded once at startup and kept
resident in VRAM, single inference takes 30~80ms, and the API is fully compatible with the
legacy service — callers only need to switch the port.

### 1.2 Deployment layout

| Item | Legacy service (kept untouched) | New service (this document) |
|---|---|---|
| Location | `/home/deployer/Pedestron/demo/` | `/home/deployer/yolo_service/` |
| Port | 1237 | **1238** |
| Python env | conda `open-mmlab` (mmdet 2.x) | conda `yolo11` (torch 2.5.1+cu121 + ultralytics 8.4.183) |
| Model | cascade_hrnet + epoch_19.pth.stu | yolo11n.pt (COCO person, conf>=0.5) |
| Process mgmt | `demo/start.sh` (nohup) | **systemd** `yolo-service.service` (boot start + crash restart) |
| WSGI | Flask built-in | waitress (8 threads), falls back to Flask if unavailable |

The two services coexist without interfering — you can canary-switch between them.

---

## 2. Server environment

| Item | Value |
|---|---|
| Server | <SERVER_IP> (Ubuntu 22.04, kernel 6.8.0-52) |
| GPU | GTX 1660 6GB, driver 535.230.02 (supports up to CUDA 12.x) |
| Miniconda | `/home/deployer/miniconda3` (owned by deployer, writable) |
| Disk | `/` 414G (14% used); `/home` 28G (10% used after the 2026-10-22 cleanup, 24G free) |
| Internet | slow (~1MB/s); upload large wheels manually |

> On 2026-10-22 `/home` was cleaned: ~3.5G of installers/old projects removed, 11G pip cache
> and 1.6G old logs emptied; large files (`Pedestron.rar`, `torch-*.whl`) moved to
> `/backup/deployer/`. Log rotation: see §9.2.

---

## 3. Environment setup (follow this when rebuilding a machine)

### 3.1 Create the environment

```bash
/home/deployer/miniconda3/bin/conda create -n yolo11 python=3.10 -y
PY=/home/deployer/miniconda3/envs/yolo11/bin
```

### 3.2 Install torch first (large file — upload the wheel manually if needed)

```bash
# Option A: via the Aliyun PyTorch mirror (when the network is good)
$PY/pip install --timeout 60 torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu121/

# Option B: download the wheel locally and upload (recommended)
#   local: https://mirrors.aliyun.com/pytorch-wheels/cu121/
#   scp torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl deployer@<SERVER_IP>:/home/deployer/
$PY/pip install /home/deployer/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
$PY/pip install --no-deps torchvision==0.20.1+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

### 3.3 Install the remaining dependencies

```bash
$PY/pip install --timeout 60 -i https://mirrors.aliyun.com/pypi/simple/ \
    -r /home/deployer/yolo_service/requirements.txt
```

> ⚠️ **Order matters**: torch before ultralytics. If reversed, pip pulls a CUDA 13 build of
> torch (2.14.1+cu130) from PyPI, causing `cuda: False` or even cuDNN init failure.

### 3.4 Verify

```bash
$PY/python - <<'EOF'
import torch, torchvision, ultralytics
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('torchvision', torchvision.__version__, 'ultralytics', ultralytics.__version__)
torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 64, 64).cuda())
print('cudnn OK', torch.backends.cudnn.version())
EOF
# expected: 2.5.1+cu121 True / 0.20.1+cu121 8.4.183 / cudnn OK 91000
```

> If you see `torchvision==0.29 is incompatible with torch==2.5`, stale dist-info remains in
> site-packages: `rm -rf .../torchvision-0.29.1.dist-info` (the actual 0.20.1 module is fine).

---

## 4. Directory layout (v1.1)

```
/home/deployer/yolo_service/
├── app.py                     # HTTP layer: routes, validation, stats, periodic cleanup
├── config.py                  # config center (YS_* env vars)
├── detector.py                # model singleton + serialized GPU inference
├── cache.py                   # snapshot cache, similarity dedup, TTL cleanup
├── callback.py                # alert callback (async + retry)
├── yolo_service.py            # compatibility entry (systemd/legacy scripts may point here)
├── yolo11n.pt                 # model weights (5.4MB)
├── requirements.txt / VERSION / README.md / CHANGELOG.md
├── DEPLOY.md                  # this file
├── start_yolo.sh              # emergency manual start (do not use together with systemd)
├── deploy.sh                  # one-shot deploy/upgrade (backup + health check + rollback)
├── systemd/yolo-service.service
├── tools/{package.sh, make_manifest.py, audit_files.py}
├── tests/test_api.py
├── start/                     # runtime: snapshot cache
├── results/                   # runtime: annotated images (deleted after use)
├── service.log                # application log (10MB x 5 rotated)
├── backup/                    # old code auto-backed-up by deploy.sh
└── yolo_service.env           # optional: centralized config (KEY=VALUE)
```

### 4.1 Weights

- Source: `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`
- For better accuracy swap in `yolo11n s/s/m` (a 1660 can still run `yolo11s.pt`) or a
  CrowdHuman-finetuned model; just point `YS_MODEL=/path/to/xx.pt` — no code change needed.

### 4.2 Key configuration

Everything is overridden via environment variables; **defaults match the legacy service**
(full table in `README.md` §3). For production, write them into
`/home/deployer/yolo_service/yolo_service.env`:

```ini
YS_PORT=1238
YS_CALLBACK_URL=http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo
YS_SCORE_THR=0.5
YS_DEDUP_THRESHOLD=0.99
YS_CALLBACK_ASYNC=1
YS_IMAGE_TTL_HOURS=24
YS_LOG_LEVEL=INFO
```

The systemd unit already includes `EnvironmentFile=-.../yolo_service.env`
(no error if the file is absent).

---

## 5. API

### 5.1 `POST /api/portraitRecognition`

multipart/form-data:

| Field | Description |
|---|---|
| `devices` | device ID, whitelist `^[A-Za-z0-9_.-]{1,64}$` (invalid values are rejected) |
| `file` | snapshot (jpg/jpeg/png/bmp, <= 20MB) |

Internal flow (v1.1):

1. save as `start/<millisecond-timestamp>_<device>.<ext>` (**avoids same-second overwrite**)
   and verify it decodes;
2. compare the **latest two** images of the device (grayscale template matching scaled to
   1920x1080):
   - score >= 0.99: frame unchanged, skip detection;
   - score < 0.99: run YOLO detection (person class only, conf>=0.5);
3. count > 0: annotated image base64 -> callback to `YS_CALLBACK_URL` (async + 2 retries);
4. the result image is deleted right away; the cache keeps only the latest image;
   each processed image logs one INFO line (with timing).

Response (HTTP always 200, business code in body):

```json
{"code": "200", "msg": "ok"}
{"code": "400", "msg": "图片不足两张"}
{"code": "400", "msg": "devices 参数缺失或非法"}
```

### 5.2 `GET /api/clearPicture?devices=xxx`

Deletes all cached images of the device,
returns `{"code":"200","msg":"ok","removed":N}`.

### 5.3 `GET /api/health`

```json
{"code":"200","msg":"ok","device":"cuda:0","model":"yolo11n","version":"1.1.0",
 "uptime_s":3600,"requests":1024,"alarms":3,"cache_files":1,
 "gpu":{"used_mb":812,"total_mb":6144}}
```

### 5.4 `GET /api/stats`

Requests / detections / dedup skips / alarms / errors / average inference time /
callback config / effective config.

### 5.5 Switching callers over

Change `http://<SERVER_IP>:1237/api/portraitRecognition` to `:1238`, nothing else.
Recommended: canary one camera first (its `devices` goes to the new service), watch
`/api/stats` and the alert platform data, then switch the rest in batch.

---

## 6. Start/stop and verification

The service is a systemd unit: starts on boot, auto-restarts within 5s after a crash.

```bash
sudo systemctl start   yolo-service
sudo systemctl stop    yolo-service
sudo systemctl restart yolo-service
sudo systemctl status  yolo-service
systemctl is-enabled yolo-service     # enabled
systemctl is-active  yolo-service     # active
```

Three-step verification:

```bash
ss -tlnp | grep 1238
curl -s http://127.0.0.1:1238/api/health
tail -30 /home/deployer/yolo_service/service.log
```

Smoke test (does not trigger real alerts — a noise image contains no people):

```bash
/home/deployer/miniconda3/envs/yolo11/bin/python /home/deployer/yolo_service/tests/test_api.py \
    --url http://127.0.0.1:1238
```

### 6.1 Unit file

`/etc/systemd/system/yolo-service.service` (a copy lives in `systemd/`):

```ini
[Unit]
Description=YOLO Pedestrian Detection Service (port 1238)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=deployer
Group=deployer
WorkingDirectory=/home/deployer/yolo_service
Environment=PYTHONUNBUFFERED=1
Environment=YS_PORT=1238
EnvironmentFile=-/home/deployer/yolo_service/yolo_service.env
ExecStart=/home/deployer/miniconda3/envs/yolo11/bin/python -u /home/deployer/yolo_service/app.py
Restart=always
RestartSec=5
TimeoutStopSec=20
KillSignal=SIGTERM
StandardOutput=journal
StandardError=journal
OOMScoreAdjust=-100

[Install]
WantedBy=multi-user.target
```

After editing: `sudo systemctl daemon-reload && sudo systemctl restart yolo-service`.

> Logs live in two places: `journalctl -u yolo-service` (stdout) and `service.log`
> (rotating app log); their formats match.

---

## 7. Upgrade / rollback / packaging

### 7.1 In-place upgrade on the server

```bash
cd /home/deployer/yolo_service
bash deploy.sh            # auto: backup -> copy code -> deps check -> restart -> health check -> rollback on failure
```

Backups are kept at `backup/code_<timestamp>.tar.gz`; rollback:

```bash
tar -xzf backup/code_20261022_103000.tar.gz -C /home/deployer/yolo_service
sudo systemctl restart yolo-service
```

### 7.2 Packaging & distribution

```bash
bash tools/package.sh
# dist/yolo_service-1.1.0.tar.gz        code + docs + weights + manifest
# dist/yolo_service-1.1.0.tar.gz.sha256 checksum file
```

On the target machine:

```bash
sha256sum -c yolo_service-1.1.0.tar.gz.sha256
tar -xzf yolo_service-1.1.0.tar.gz && cd yolo_service-1.1.0
bash deploy.sh
```

### 7.3 Canary instance (does not touch 1238)

```bash
YS_PORT=1240 YS_START_DIR=./start_1240 YS_RESULTS_DIR=./results_1240 \
YS_LOG_FILE=./service_1240.log python app.py
python tests/test_api.py --url http://127.0.0.1:1240
```

---

## 8. File integrity & suspicious-deletion audit

The package's `MANIFEST.sha256` records sha256 and size of every file; verify anytime:

```bash
cd /home/deployer/yolo_service
python3 tools/audit_files.py                         # verify this service
python3 tools/audit_files.py --extra /home/deployer/Pedestron   # include the legacy dir
python3 tools/audit_files.py --deleted-only          # only deleted-but-open handles
```

Output meaning:

| Flag | Meaning | Action |
|---|---|---|
| `MISSING` | in manifest, not on disk | re-extract the file from the release package |
| `MODIFIED` | size or sha256 mismatch | confirm whether it was edited intentionally, otherwise overwrite from the package |
| `EXTRA` | file not in manifest | usually runtime artifacts (logs/cache); confirm and ignore |
| deleted handle | `(deleted)` under `/proc/*/fd` | file removed but a process still holds it; restart that process to free space |

---

## 9. Operations notes & FAQ

### 9.1 Lessons learned

1. **pip stuck / very slow**: the Tsinghua mirror often times out on this box; use Aliyun
   `-i https://mirrors.aliyun.com/pypi/simple/`; upload big torch wheels manually (§3.2).
2. **Installing ultralytics reinstalls torch**: ordering issue (§3.3).
3. **torch overwritten to 2.14.1+cu130**: `cuda: False`. Fix:

   ```bash
   $PY/pip install --force-reinstall --no-deps /home/deployer/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
   $PY/pip install --force-reinstall --no-deps torchvision==0.20.1+cu121 \
       -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
   ```
   Note: the running process keeps using the old libs in memory and looks fine —
   **it breaks on the next restart**. Fix before restarting.
4. **`CUDNN_STATUS_NOT_INITIALIZED`**: cu13 libs mixed into `site-packages/nvidia`.
   `rm -rf .../site-packages/nvidia`, reinstall torch from the whl, restart the service.
5. **snap curl cannot read /tmp files**: test the API with `python -c "import requests..."`.
6. **pkill kills itself**: `pkill -f 'yolo_service.py'` matches the ssh command line itself;
   use `pkill -f '[y]olo_service.py'`.
7. **port occupied on start**: usually a leftover nohup process;
   `pkill -f '[a]pp.py'`, then restart systemd.

### 9.2 Disk maintenance

```bash
# disk usage per service
du -sh /home/deployer/yolo_service/{start,results} /home/deployer/Pedestron/demo/{start,results,log}

# force-rotate the legacy service log (normally done daily by logrotate.timer)
sudo logrotate -f /etc/logrotate.d/pedestron-legacy

# inspect rotation config
cat /etc/logrotate.d/pedestron-legacy
```

**Current disk baseline (after the 2026-10-22 cleanup)**: `/home` 2.5G / 28G (10%), 24G free.

Per-service log policy:

| Service | Log file | Policy |
|---|---|---|
| 1238 yolo_service | `service.log` | in-app RotatingFileHandler, 10MB x 5 |
| 1239 yolo-compare | `service.log` | same |
| 1237 pedestron-legacy | `log/b_service.log` | logrotate: weekly or >100MB, keep 4, compressed |

> ⚠️ 1237 used to run as `nohup ... > b.log` (**no O_APPEND**); a direct `truncate` left the
> file offset at the old position and created a 1.6G sparse hole (apparent size did not
> shrink). It has been migrated to systemd (`StandardOutput=append:`, O_APPEND), so both
> `truncate` and `copytruncate` now work cleanly. Historical log archived at
> `log/b.log.tail_20261022.gz`.

### 9.3 Troubleshooting order

1. `systemctl status yolo-service` / `journalctl -u yolo-service -n 80`
2. search `service.log` for `ERROR|异常|回调失败` (error / exception / callback failed)
3. `curl /api/health` — check `device` is `cuda:0` and `gpu.used_mb` looks sane
4. `nvidia-smi` — processes and VRAM
5. if callbacks fail, verify `<ALERT_HOST>:8600` reachability

---

## 10. Other services on the same machine

| Port | Service | systemd unit | Notes |
|---|---|---|---|
| 1237 | legacy Pedestron | `pedestron-legacy.service` | enabled + active, managed by systemd since 2026-10-22 (old root/nohup process retired); logs via `log/b_service.log` + logrotate |
| 1238 | **this service** yolo_service | `yolo-service.service` | enabled + active |
| 1239 | web console | `yolo-compare.service` | `http://<SERVER_IP>:1239` |
| 3306 / 1194 | MySQL / OpenVPN (Docker) | — | unrelated |

```bash
sudo systemctl status yolo-service yolo-compare pedestron-legacy
```

---

## 11. Change log

| Date | Version | Change |
|---|---|---|
| 2026-10-21 | 1.0.0 | initial deploy: yolo11 env + yolo_service(1238) + compare console(1239) |
| 2026-10-21 | 1.0.0 | fixed a lagged install script overwriting torch to 2.14.1+cu130; reinstalled 2.5.1+cu121 |
| 2026-10-22 | 1.0.0 | fixed cu13 nvidia packages conflicting with cu12 causing cuDNN init failure |
| 2026-10-22 | 1.0.0 | three web services moved under systemd (boot start + crash restart) |
| 2026-10-22 | 1.1.0 | 1237 migrated from legacy root/nohup to `pedestron-legacy.service`; logs switched to `b_service.log` + logrotate (weekly/100MB, keep 4); the 1.6G `b.log` emptied & archived as `.tail_20261022.gz`; /home cleanup freed ~17G |
| 2026-10-22 | **1.1.0** | modular refactor; fixed same-second overwrite, path traversal, concurrent inference, log/disk growth; added stats/warmup/async callback/waitress; added tests, deploy script, packaging and file audit |
