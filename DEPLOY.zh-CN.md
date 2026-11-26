# YOLO 行人检测服务部署与运维手册

> 服务：`yolo_service`（Ultralytics YOLO11 行人检测，替代 Pedestron/mmdet 旧服务）
> 版本：**v1.1.0** ｜ 更新日期：2026-10-22
> 代码位置（服务器）：`/home/deployer/yolo_service/` ｜ 端口：**1238**

---

## 1. 服务概述

### 1.1 背景

旧服务基于 [Pedestron](https://github.com/mightyCelu/Pedestron)（mmdetection 2.x，2022 年后停止维护），存在问题：

- 每次请求都 `init_detector` 重新加载 HRNet 模型（数秒 + 反复申请/释放显存），长期运行内存涨到 8GB+；
- 依赖 mmdet/mmcv 2.x，与新 CUDA/PyTorch 兼容性差；
- cascade HRNet 推理慢（实测约 4.9 秒/次）。

新服务改用 **Ultralytics YOLO11n**：模型启动加载一次常驻显存，单次推理 30~80ms，
接口协议与旧服务完全一致，业务方只需换端口即可切换。

### 1.2 部署形态

| 项目 | 旧服务（保留未动） | 新服务（本文档） |
|---|---|---|
| 位置 | `/home/deployer/Pedestron/demo/` | `/home/deployer/yolo_service/` |
| 端口 | 1237 | **1238** |
| Python 环境 | conda `open-mmlab`（mmdet 2.x） | conda `yolo11`（torch 2.5.1+cu121 + ultralytics 8.4.183） |
| 模型 | cascade_hrnet + epoch_19.pth.stu | yolo11n.pt（COCO person，conf≥0.5） |
| 进程管理 | `demo/start.sh`（nohup） | **systemd** `yolo-service.service`（开机自启 + 崩溃重启） |
| WSGI | Flask 内置 | waitress（8 线程），未安装则自动回退 Flask |

两服务并存、互不影响，可灰度切换。

---

## 2. 服务器环境

| 项目 | 内容 |
|---|---|
| 服务器 | <SERVER_IP>（主机名 user，Ubuntu 22.04，内核 6.8.0-52） |
| GPU | GTX 1660 6GB，驱动 535.230.02（支持到 CUDA 12.x） |
| Miniconda | `/home/deployer/miniconda3`（属 deployer，可写） |
| 磁盘 | `/` 414G（已用 14%）；`/home` 28G（2026-10-22 清理后已用 10%，可用 24G） |
| 外网 | 较慢（约 1MB/s），大包建议手动上传 whl |

> 2026-10-22 已清理 `/home`：删除约 3.5G 安装包/旧项目、清空 11G pip 缓存、清空 1.6G 旧日志，
> 大文件（`Pedestron.rar`、`torch-*.whl`）移至 `/backup/deployer/` 保留。日志轮转见 §9.2。

---

## 3. 环境安装（换机重装照此执行）

### 3.1 创建环境

```bash
/home/deployer/miniconda3/bin/conda create -n yolo11 python=3.10 -y
PY=/home/deployer/miniconda3/envs/yolo11/bin
```

### 3.2 安装 torch（先装，大文件可手动上传）

```bash
# 方式 A：直接走阿里云 PyTorch 镜像（网络好时）
$PY/pip install --timeout 60 torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu121/

# 方式 B：本地下载 whl 后上传（推荐）
#   本地：https://mirrors.aliyun.com/pytorch-wheels/cu121/
#   scp torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl deployer@<SERVER_IP>:/home/deployer/
$PY/pip install /home/deployer/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
$PY/pip install --no-deps torchvision==0.20.1+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

### 3.3 安装其余依赖

```bash
$PY/pip install --timeout 60 -i https://mirrors.aliyun.com/pypi/simple/ \
    -r /home/deployer/yolo_service/requirements.txt
```

> ⚠️ **顺序不能反**：先 torch 后 ultralytics。反了 pip 会从 pypi 拉入 CUDA 13 版 torch
> （torch 2.14.1+cu130），导致 `cuda: False`，甚至 cuDNN 初始化失败。

### 3.4 验证

```bash
$PY/python - <<'EOF'
import torch, torchvision, ultralytics
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('torchvision', torchvision.__version__, 'ultralytics', ultralytics.__version__)
torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 64, 64).cuda())
print('cudnn OK', torch.backends.cudnn.version())
EOF
# 期望：2.5.1+cu121 True / 0.20.1+cu121 8.4.183 / cudnn OK 91000
```

> 若出现 `torchvision==0.29 is incompatible with torch==2.5`，说明 site-packages 里残留了
> 旧版 dist-info，可 `rm -rf .../torchvision-0.29.1.dist-info`（不影响实际 0.20.1 模块）。

---

## 4. 目录结构（v1.1）

```
/home/deployer/yolo_service/
├── app.py                     # HTTP 层：路由、校验、统计、定时清理
├── config.py                  # 配置中心（YS_* 环境变量）
├── detector.py                # 模型单例 + GPU 串行推理
├── cache.py                   # 抓拍缓存、相似度去重、过期清理
├── callback.py                # 告警回调（异步 + 重试）
├── yolo_service.py            # 兼容入口（systemd/旧脚本仍可指向它）
├── yolo11n.pt                 # 模型权重（5.4MB）
├── requirements.txt / VERSION / README.md / CHANGELOG.md
├── DEPLOY.md                  # 本文件
├── start_yolo.sh              # 应急手动启动（不要与 systemd 同用）
├── deploy.sh                  # 一键部署/升级（备份 + 健康检查 + 回滚）
├── systemd/yolo-service.service
├── tools/{package.sh, make_manifest.py, audit_files.py}
├── tests/test_api.py
├── start/                     # 运行期：抓拍图缓存
├── results/                   # 运行期：标注结果图（用后即删）
├── service.log                # 应用日志（10MB × 5 轮转）
├── backup/                    # deploy.sh 自动备份的旧代码
└── yolo_service.env           # 可选：集中配置（KEY=VALUE）
```

### 4.1 权重

- 来源：`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`
- 追求精度可换 `yolo11s.pt`（1660 也能跑），或用 CrowdHuman 微调后替换；
  通过 `YS_MODEL=/path/to/xx.pt` 指定即可，无需改代码。

### 4.2 关键配置

全部通过环境变量覆盖，**默认值与旧服务一致**，完整表见 `README.md` §3。
生产环境建议写入 `/home/deployer/yolo_service/yolo_service.env`：

```ini
YS_PORT=1238
YS_CALLBACK_URL=http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo
YS_SCORE_THR=0.5
YS_DEDUP_THRESHOLD=0.99
YS_CALLBACK_ASYNC=1
YS_IMAGE_TTL_HOURS=24
YS_LOG_LEVEL=INFO
```

systemd 单元已通过 `EnvironmentFile=-.../yolo_service.env` 引入（文件不存在也不报错）。

---

## 5. 接口说明

### 5.1 `POST /api/portraitRecognition`

multipart/form-data：

| 字段 | 说明 |
|---|---|
| `devices` | 设备号，白名单 `^[A-Za-z0-9_.-]{1,64}$`（非法直接拒绝） |
| `file` | 抓拍图（jpg/jpeg/png/bmp，≤20MB） |

内部流程（v1.1）：

1. 保存为 `start/<毫秒时间戳>_<设备号>.<后缀>`（**避免同秒覆盖**），并校验可解码；
2. 取该设备**最新两张**做相似度比对（缩放到 1920×1080 灰度模板匹配）：
   - 得分 ≥0.99：画面未变化，跳过检测；
   - 得分 <0.99：YOLO 检测（仅 person 类，conf≥0.5）；
3. 人数 >0：标注结果图 base64 → 回调 `YS_CALLBACK_URL`（异步 + 重试 2 次）；
4. 结果图即删，缓存只保留最新 1 张；每张图处理完写一条 INFO 日志（含耗时）。

响应（HTTP 恒 200，业务码在 body）：

```json
{"code": "200", "msg": "ok"}
{"code": "400", "msg": "图片不足两张"}
{"code": "400", "msg": "devices 参数缺失或非法"}
```

### 5.2 `GET /api/clearPicture?devices=xxx`

删除该设备全部缓存图，返回 `{"code":"200","msg":"ok","removed":N}`。

### 5.3 `GET /api/health`

```json
{"code":"200","msg":"ok","device":"cuda:0","model":"yolo11n","version":"1.1.0",
 "uptime_s":3600,"requests":1024,"alarms":3,"cache_files":1,
 "gpu":{"used_mb":812,"total_mb":6144}}
```

### 5.4 `GET /api/stats`

请求数 / 检测数 / 去重跳过数 / 告警数 / 错误数 / 平均推理耗时 / 回调配置 / 生效配置。

### 5.5 业务方切换

把 `http://<SERVER_IP>:1237/api/portraitRecognition` 改成 `:1238`，其余不变。
建议先在一台摄像头灰度（`devices` 指向新服务），观察 `/api/stats` 与告警平台数据后再批量切。

---

## 6. 启停与验证

服务已注册为 systemd 单元，开机自启、崩溃 5 秒内自动重启。

```bash
sudo systemctl start   yolo-service
sudo systemctl stop    yolo-service
sudo systemctl restart yolo-service
sudo systemctl status  yolo-service
systemctl is-enabled yolo-service     # enabled
systemctl is-active  yolo-service     # active
```

验证三步：

```bash
ss -tlnp | grep 1238
curl -s http://127.0.0.1:1238/api/health
tail -30 /home/deployer/yolo_service/service.log
```

冒烟测试（不触发真实告警，噪声图不会检出人）：

```bash
/home/deployer/miniconda3/envs/yolo11/bin/python /home/deployer/yolo_service/tests/test_api.py \
    --url http://127.0.0.1:1238
```

### 6.1 单元文件

`/etc/systemd/system/yolo-service.service`（副本在 `systemd/`）：

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

改完执行 `sudo systemctl daemon-reload && sudo systemctl restart yolo-service`。

> 日志分两处：`journalctl -u yolo-service`（stdout）与 `service.log`（应用轮转日志），两者内容格式一致。

---

## 7. 升级 / 回滚 / 打包

### 7.1 服务器上直接升级

```bash
cd /home/deployer/yolo_service
bash deploy.sh            # 自动：备份 -> 复制代码 -> 依赖校验 -> 重启 -> 健康检查 -> 失败回滚
```

备份在 `backup/code_<时间>.tar.gz`，回滚：

```bash
tar -xzf backup/code_20261022_103000.tar.gz -C /home/deployer/yolo_service
sudo systemctl restart yolo-service
```

### 7.2 打包分发

```bash
bash tools/package.sh
# dist/yolo_service-1.1.0.tar.gz        代码+文档+权重+清单
# dist/yolo_service-1.1.0.tar.gz.sha256 校验文件
```

目标机器：

```bash
sha256sum -c yolo_service-1.1.0.tar.gz.sha256
tar -xzf yolo_service-1.1.0.tar.gz && cd yolo_service-1.1.0
bash deploy.sh
```

### 7.3 灰度实例（不影响 1238）

```bash
YS_PORT=1240 YS_START_DIR=./start_1240 YS_RESULTS_DIR=./results_1240 \
YS_LOG_FILE=./service_1240.log python app.py
python tests/test_api.py --url http://127.0.0.1:1240
```

---

## 8. 文件完整性与可疑删除审计

包内 `MANIFEST.sha256` 记录每个文件的 sha256 与大小，可随时校验：

```bash
cd /home/deployer/yolo_service
python3 tools/audit_files.py                         # 校验本服务
python3 tools/audit_files.py --extra /home/deployer/Pedestron   # 连旧业务目录一起扫
python3 tools/audit_files.py --deleted-only          # 只看“已删除仍被占用”的句柄
```

输出含义：

| 标记 | 含义 | 处理 |
|---|---|---|
| `MISSING` | 清单里有、磁盘上没有 | 从发布包重新解出该文件 |
| `MODIFIED` | 大小或 sha256 不符 | 确认是否人为改动，否则从包覆盖 |
| `EXTRA` | 清单外新增文件 | 通常是运行期产物（日志/缓存），确认无异常即可 |
| 删除句柄 | `/proc/*/fd` 中 `(deleted)` | 文件已删但进程仍占用，重启该进程释放空间 |

---

## 9. 运维经验与常见问题

### 9.1 已踩过的坑

1. **pip 卡住/极慢**：清华 tuna 在该机常超时，用阿里云 `-i https://mirrors.aliyun.com/pypi/simple/`；
   torch 大包手动上传 whl（§3.2）。
2. **装 ultralytics 反而重装 torch**：安装顺序问题（§3.3）。
3. **torch 被覆盖成 2.14.1+cu130**：`cuda: False`。修复：

   ```bash
   $PY/pip install --force-reinstall --no-deps /home/deployer/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
   $PY/pip install --force-reinstall --no-deps torchvision==0.20.1+cu121 \
       -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
   ```
   注意：运行中的服务用的是内存里的旧库，看似正常，**重启后才会暴露**，务必先修再重启。
4. **`CUDNN_STATUS_NOT_INITIALIZED`**：`site-packages/nvidia` 混入 cu13 库。
   `rm -rf .../site-packages/nvidia` 后用 whl 重装 torch，重启服务。
5. **snap 版 curl 读不了 /tmp 文件**：测试接口改用 `python -c "import requests..."`。
6. **pkill 自杀**：`pkill -f 'yolo_service.py'` 会匹配到 ssh 命令行自身，用 `pkill -f '[y]olo_service.py'`。
7. **端口被占用启动失败**：多为 nohup 残留，`pkill -f '[a]pp.py'` 后重启 systemd。

### 9.2 磁盘维护

```bash
# 查看各服务磁盘占用
du -sh /home/deployer/yolo_service/{start,results} /home/deployer/Pedestron/demo/{start,results,log}

# 手动强制轮转旧服务日志（正常情况下由 logrotate.timer 每日自动执行）
sudo logrotate -f /etc/logrotate.d/pedestron-legacy

# 查看轮转配置
cat /etc/logrotate.d/pedestron-legacy
```

**当前磁盘基线（2026-10-22 清理后）**：`/home` 2.5G / 28G（10%），可用 24G。

各服务日志策略：

| 服务 | 日志文件 | 策略 |
|---|---|---|
| 1238 yolo_service | `service.log` | 应用内 RotatingFileHandler，10MB × 5 |
| 1239 yolo-compare | `service.log` | 同上 |
| 1237 pedestron-legacy | `log/b_service.log` | logrotate：weekly 或超 100MB 轮转，保留 4 份并压缩 |

> ⚠️ 1237 曾用 `nohup ... > b.log` 启动，**非 O_APPEND 模式**，直接 `truncate` 会让文件偏移停在旧位置、产生 1.6G 的稀疏空洞（表观大小不降反升）。
> 已迁移到 systemd（`StandardOutput=append:`，O_APPEND），`truncate` / `copytruncate` 均可干净归零，不会再出现空洞。
> 历史日志留档：`log/b.log.tail_20261022.gz`。

### 9.3 排查顺序

1. `systemctl status yolo-service` / `journalctl -u yolo-service -n 80`
2. `service.log` 里搜 `ERROR|异常|回调失败`
3. `curl /api/health` 看 `device` 是否 `cuda:0`、`gpu.used_mb` 是否正常
4. `nvidia-smi` 看进程与显存
5. 回调不通再确认 `<RTMP_HOST>:8600` 可达性

---

## 10. 同机其他服务

| 端口 | 服务 | systemd 单元 | 说明 |
|---|---|---|---|
| 1237 | Pedestron 旧服务 | `pedestron-legacy.service` | enabled + active，2026-10-22 起由 systemd 管理（原 root nohup 进程已下线）；日志走 `log/b_service.log` + logrotate |
| 1238 | **本服务** yolo_service | `yolo-service.service` | enabled + active |
| 1239 | 接口对比页 | `yolo-compare.service` | `http://<SERVER_IP>:1239` |
| 3306 / 1194 | MySQL / OpenVPN（Docker） | — | 与本服务无关 |

```bash
sudo systemctl status yolo-service yolo-compare pedestron-legacy
```

---

## 11. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-21 | 1.0.0 | 初版部署：yolo11 环境 + yolo_service(1238) + 对比页(1239) |
| 2026-10-21 | 1.0.0 | 修复滞后安装脚本把 torch 覆盖为 2.14.1+cu130，重装 2.5.1+cu121 |
| 2026-10-22 | 1.0.0 | 修复 cu13 nvidia 包与 cu12 冲突导致 cuDNN 初始化失败 |
| 2026-10-22 | 1.0.0 | 三个 Web 服务改为 systemd 托管，开机自启 + 崩溃重启 |
| 2026-10-22 | 1.1.0 | 1237 从历史 root/nohup 进程迁移到 `pedestron-legacy.service`；日志改 `b_service.log` + logrotate（weekly/100MB，留 4 份）；旧 1.6G `b.log` 清空并留档 `.tail_20261022.gz`；`/home` 清理共释放约 17G |
| 2026-10-22 | **1.1.0** | 模块化重构；修复同秒覆盖、路径穿越、并发推理、日志/磁盘增长；新增 stats/预热/异步回调/waitress；补齐 tests、部署脚本、打包与文件审计 |
