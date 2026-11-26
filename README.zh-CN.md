# yolo_service — 行人检测告警服务

> 🌐 语言 / Language: [English](README.md) | 简体中文]

用 **Ultralytics YOLO11** 替代原 Pedestron（mmdet cascade_hrnet）实现的行人检测服务。
接口协议与原 `b.py` **完全一致**，调用方只需把端口从 `1237` 改成 `1238`。

```
摄像头 / 客户端  --POST 抓拍图-->  yolo_service (1238)
                                    ├─ 图片落盘 + 设备级缓存
                                    ├─ 画面相似度去重（0.99）
                                    ├─ YOLO11n 行人检测（person, conf≥0.5）
                                    └─ 人数 > 0 → base64 结果图 → 回调告警平台
```

## 1. 目录结构

```
yolo_service/
├── app.py                  # HTTP 层（Flask + waitress），路由、校验、统计
├── config.py               # 配置中心，全部支持 YS_* 环境变量覆盖
├── detector.py             # 模型单例、GPU 串行推理、结果图绘制
├── cache.py                # 抓拍图缓存、相似度去重、过期清理
├── callback.py             # 告警回调（异步 + 重试）
├── yolo_service.py         # 兼容入口（等价于 app.py）
├── yolo11n.pt              # 模型权重（5.6MB）
├── requirements.txt        # 依赖（torch 必须 +cu121）
├── README.md               # 本文件
├── DEPLOY.md               # 部署运维手册
├── CHANGELOG.md            # 变更记录
├── VERSION                 # 版本号
├── start_yolo.sh           # 手动启动（应急用）
├── deploy.sh               # 一键部署/升级（备份 + 回滚）
├── systemd/
│   └── yolo-service.service
├── tools/
│   ├── package.sh          # 打包 dist/yolo_service-<ver>.tar.gz
│   ├── make_manifest.py    # 生成 MANIFEST.sha256
│   └── audit_files.py      # 文件完整性 / 可疑删除审计
├── tests/
│   └── test_api.py         # 接口冒烟测试
├── start/                  # 运行期：抓拍图缓存（自动创建）
└── results/                # 运行期：标注结果图（检测后即删）
```

## 2. 接口

### 2.1 `POST /api/portraitRecognition`

| 参数 | 位置 | 说明 |
|---|---|---|
| `devices` | form | 设备号，`^[A-Za-z0-9_.-]{1,64}$` |
| `file` | form | 图片文件（jpg/png/bmp，≤20MB） |

响应（HTTP 恒为 200，业务码在 body）：

| code | msg | 场景 |
|---|---|---|
| `400` | `图片不足两张` | 该设备首次/仅一张图，等待下一张 |
| `200` | `ok` | 已比对（画面变化则已检测；未变化则跳过） |
| `400` | `devices 参数缺失或非法` / `缺少 file 文件` / `图片无法解码` | 参数错误 |
| `500` | `internal error` | 服务内部异常（详见日志） |

### 2.2 `GET /api/clearPicture?devices=xxx`

清空该设备的抓拍缓存，返回 `{"code":"200","msg":"ok","removed":N}`。

### 2.3 `GET /api/health`

```json
{"code":"200","msg":"ok","device":"cuda:0","model":"yolo11n",
 "version":"1.1.0","uptime_s":3600,"requests":1024,"alarms":3,
 "cache_files":1,"gpu":{"used_mb":812,"total_mb":6144}}
```

### 2.4 `GET /api/stats`

请求数、检测次数、跳过去重次数、告警次数、错误数、平均推理耗时、回调配置、生效配置。

> 告警回调 payload 与旧服务一致：
> `{"deviceId": "...", "count": N, "imageBase64": "..."}`
> → `POST http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo`

## 3. 配置（环境变量）

所有默认值都与旧版行为一致，不配置即可直接跑。

| 变量 | 默认 | 说明 |
|---|---|---|
| `YS_PORT` | `1238` | 监听端口 |
| `YS_HOST` | `0.0.0.0` | 监听地址 |
| `YS_SERVER` | `waitress` | `waitress`（生产）/ `flask`（调试） |
| `YS_THREADS` | `8` | waitress 线程数 |
| `YS_MODEL` | `./yolo11n.pt` | 权重路径 |
| `YS_DEVICE` | 自动 | `cuda:0` / `cpu` |
| `YS_SCORE_THR` | `0.5` | 检测置信度阈值 |
| `YS_DEDUP_THRESHOLD` | `0.99` | 相似度阈值，≥ 视为画面未变化 |
| `YS_CALLBACK_URL` | `http://<ALERT_HOST>:8600/...` | 告警回调地址 |
| `YS_CALLBACK_ASYNC` | `1` | 异步回调（不阻塞请求） |
| `YS_CALLBACK_TIMEOUT` | `10` | 单次回调超时（秒） |
| `YS_CALLBACK_RETRY` | `2` | 失败重试次数 |
| `YS_START_DIR` / `YS_RESULTS_DIR` | `./start` / `./results` | 缓存/结果目录（灰度实例建议分开） |
| `YS_IMAGE_TTL_HOURS` | `24` | 抓拍图保留时长 |
| `YS_MAX_IMAGES_PER_DEVICE` | `20` | 单设备缓存文件上限 |
| `YS_CLEANUP_INTERVAL` | `3600` | 定时清理间隔（秒），0 关闭 |
| `YS_LOG_FILE` | `./service.log` | 日志文件（10MB × 5 轮转） |
| `YS_LOG_LEVEL` | `INFO` | 日志级别 |

systemd 部署时把配置写到 `/home/deployer/yolo_service/yolo_service.env`，一行一个 `KEY=VALUE` 即可。

## 4. 快速开始

```bash
# 1) 依赖（必须先 torch 再 ultralytics）
/home/deployer/miniconda3/envs/yolo11/bin/pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
/home/deployer/miniconda3/envs/yolo11/bin/pip install -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# 2) 启动（生产）
sudo systemctl start yolo-service

# 3) 验证
curl -s http://127.0.0.1:1238/api/health
/home/deployer/miniconda3/envs/yolo11/bin/python tests/test_api.py --url http://127.0.0.1:1238
```

灰度新版本（不影响线上 1238）：

```bash
YS_PORT=1240 YS_START_DIR=./start_1240 YS_RESULTS_DIR=./results_1240 \
YS_LOG_FILE=./service_1240.log python app.py
```

## 5. 打包与分发

```bash
bash tools/package.sh
# dist/yolo_service-1.1.0.tar.gz
# dist/yolo_service-1.1.0.tar.gz.sha256
```

包内已含 `MANIFEST.sha256`。目标机器：

```bash
sha256sum -c yolo_service-1.1.0.tar.gz.sha256   # 1) 校验包
tar -xzf yolo_service-1.1.0.tar.gz && cd yolo_service-1.1.0
bash deploy.sh                                   # 2) 部署（自动备份、健康检查、失败回滚）
```

## 6. 文件完整性 / 可疑删除审计

```bash
# 校验项目文件是否缺失或被篡改（依据 MANIFEST.sha256）
python3 tools/audit_files.py

# 同时扫描旧业务目录里「已删除但仍被进程占用」的文件
python3 tools/audit_files.py --extra /home/deployer/Pedestron

# 只看删除句柄
python3 tools/audit_files.py --deleted-only
```

审计会输出三类结果：`MISSING`（清单中有、磁盘上没有）、`MODIFIED`（大小/哈希不符）、
`EXTRA`（清单外新增），以及 `/proc/*/fd` 中 `(deleted)` 的句柄（这类文件占用磁盘但目录里看不到）。

## 7. 运维

```bash
sudo systemctl status  yolo-service
sudo systemctl restart yolo-service
sudo journalctl -u yolo-service -n 100 --no-pager -f
tail -f /home/deployer/yolo_service/service.log
```

- 开机自启：`systemctl is-enabled yolo-service` → `enabled`
- 崩溃自愈：`Restart=always`，`RestartSec=5`
- 别再手动跑 `start_yolo.sh`（会与 systemd 抢端口，仅应急用）

## 8. 排错速查

| 现象 | 排查 |
|---|---|
| 启动即退出 | `journalctl -u yolo-service -n 50`；多为模型文件缺失或 torch/cuDNN 版本不匹配 |
| `cuda: False` | `pip show torch` 应为 `2.5.1+cu121`；若被升级成 `+cu13*` 需用本地 whl 重装 |
| `CUDNN_STATUS_NOT_INITIALIZED` | `site-packages/nvidia` 里混入 cu13 库，清空该目录后重装 cu12 版 torch |
| 一直返回「图片不足两张」 | 该设备还没积攒到 2 张；或 `clearPicture` 被高频调用 |
| 有画面但一直不检测 | 画面与上一张相似度 ≥ 0.99，属正常去重；可临时调 `YS_DEDUP_THRESHOLD=0.995` 验证 |
| 检测到人不回调 | 检查 `YS_CALLBACK_URL` 可达性、`/api/stats` 的 alarms 计数、日志 `回调失败` |
| 内存/显存持续上涨 | `curl /api/health` 看 `gpu.used_mb`；推理已加锁串行，正常应稳定 |
