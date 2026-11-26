# -*- coding: utf-8 -*-
"""配置中心：所有参数均可用环境变量覆盖，便于同机多实例 / 灰度部署 / 迁移。

环境变量命名统一以 YS_ 开头（YOLO Service）。
"""
import os

# ---------- 基础路径 ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ---------- 版本 ----------
VERSION = '1.2.0'


def _env(name, default):
    v = os.environ.get(name)
    return default if v is None or v.strip() == '' else v.strip()


def _env_int(name, default):
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(_env(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    v = _env(name, '1' if default else '0').lower()
    return v in ('1', 'true', 'yes', 'on')


# ---------- 服务监听 ----------
HOST = _env('YS_HOST', '0.0.0.0')
PORT = _env_int('YS_PORT', 1238)
SERVER = _env('YS_SERVER', 'waitress')      # waitress | flask
THREADS = _env_int('YS_THREADS', 8)
MAX_UPLOAD_MB = _env_int('YS_MAX_UPLOAD_MB', 20)

# ---------- 模型 ----------
MODEL_PATH = _env('YS_MODEL', os.path.join(BASE_DIR, 'yolo11n.pt'))
DEVICE = _env('YS_DEVICE', '')              # 空 = 自动选择 cuda:0 / cpu
SCORE_THR = _env_float('YS_SCORE_THR', 0.5)  # 与原 Pedestron 保持一致
WARMUP = _env_bool('YS_WARMUP', True)

# ---------- 去重（dHash 感知哈希 + 汉明距离，比较对象为“最新两张”） ----------
DEDUP_HASH_SIZE = _env_int('YS_DEDUP_HASH_SIZE', 16)     # 哈希边长，位串 = size*size 位
DEDUP_HASH_DIST = _env_int('YS_DEDUP_HASH_DIST', 10)     # 汉明距离 <= 该值视为“画面未变化”

# ---------- 告警回调 ----------
CALLBACK_URL = _env('YS_CALLBACK_URL', 'http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo')
CALLBACK_TIMEOUT = _env_float('YS_CALLBACK_TIMEOUT', 10)
CALLBACK_RETRY = _env_int('YS_CALLBACK_RETRY', 2)      # 失败重试次数（不含首次）
CALLBACK_ASYNC = _env_bool('YS_CALLBACK_ASYNC', True)  # 异步回调，避免阻塞摄像头请求
CALLBACK_WORKERS = _env_int('YS_CALLBACK_WORKERS', 4)

# ---------- 存储 ----------
START_DIR = _env('YS_START_DIR', os.path.join(BASE_DIR, 'start'))
RESULTS_DIR = _env('YS_RESULTS_DIR', os.path.join(BASE_DIR, 'results'))
MAX_IMAGES_PER_DEVICE = _env_int('YS_MAX_IMAGES_PER_DEVICE', 20)
IMAGE_TTL_HOURS = _env_float('YS_IMAGE_TTL_HOURS', 24)
CLEANUP_INTERVAL = _env_int('YS_CLEANUP_INTERVAL', 3600)   # 秒，0 = 关闭定时清理

# ---------- 日志 ----------
LOG_FILE = _env('YS_LOG_FILE', os.path.join(BASE_DIR, 'service.log'))
LOG_LEVEL = _env('YS_LOG_LEVEL', 'INFO').upper()
LOG_MAX_BYTES = _env_int('YS_LOG_MAX_BYTES', 10 * 1024 * 1024)
LOG_BACKUP_COUNT = _env_int('YS_LOG_BACKUP_COUNT', 5)

# 允许的图片后缀
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
