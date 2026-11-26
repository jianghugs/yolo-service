# 变更记录

> Change log is kept in Chinese below (historical record). For docs, see [README](README.md) (English) / [README.zh-CN](README.zh-CN.md).

## 1.1.0 - 2026-10-22

### 结构
- 拆分为 `config.py`（配置中心）/ `detector.py`（模型与推理）/ `cache.py`（缓存与去重）/
  `callback.py`（告警回调）/ `app.py`（HTTP 层），`yolo_service.py` 保留为兼容入口
- 新增 `tests/test_api.py` 冒烟测试、`tools/package.sh` 打包、`tools/make_manifest.py` 清单、
  `tools/audit_files.py` 完整性审计、`deploy.sh` 一键部署（含备份与回滚）

### 修复
- **同秒文件名冲突**：文件名由「秒级时间戳」改为「毫秒时间戳 + 设备号」，避免同一秒内两次上传互相覆盖
- **路径穿越风险**：`devices` 参数加入 `^[A-Za-z0-9_.-]{1,64}$` 白名单校验，`../` 之类直接拒绝
- **文件句柄泄漏**：回调读取结果图改用 `with open(...)`，不再依赖 GC
- **并发推理风险**：模型推理加线程锁，Flask/waitress 多线程下不会并发压同一 GPU 模型
- **启动期异常崩溃**：`devices` / `file` 缺失、图片损坏、超大文件均返回 JSON 错误，不再抛 500 HTML
- **日志无限增长**：改用 RotatingFileHandler（10MB × 5）轮转，stdout 走 journald
- **磁盘无限增长**：新增每小时清理（`YS_IMAGE_TTL_HOURS`，默认 24h）+ 单设备上限保护
- **去重比较对象**：由「最旧两张」改为「最新两张」，缓存异常堆积时不会拿旧图做比较

### 优化
- 启动后用空白图预热一次，首个真实请求不再慢 1~2 秒
- 回调改为异步（可关）+ 失败重试 2 次 + 超时/返回码日志，不阻塞摄像头请求
- 新增 `GET /api/stats` 运行统计；`/api/health` 增加版本、运行时长、缓存数、显存
- 默认使用 waitress 生产 WSGI（未安装时自动回退 Flask 内置服务器）

### 兼容性
- `/api/portraitRecognition`、`/api/clearPicture` 的请求参数与响应体完全不变（`code`/`msg` 原样）
- 环境变量未设置时，默认值与旧版一致（端口 1238、阈值 0.5、去重 0.99、回调 <ALERT_HOST>:8600）

## 1.0.0 - 2026-10-21

- 首版：Ultralytics YOLO11n 替代 Pedestron，接口协议与旧 `b.py` 一致，端口 1238
- systemd 托管（`yolo-service.service`），开机自启 + 崩溃自动重启
