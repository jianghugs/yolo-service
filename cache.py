# -*- coding: utf-8 -*-
"""抓拍图缓存与画面相似度判断。

关键点：
  * devices 参数会拼进文件名，必须校验，防止 `../` 之类的路径穿越
  * 设备归属用「文件名结构精确匹配」（<时间戳>_<devices>.<ext>），
    避免子串误匹配：设备 1 不再错误命中设备 21 / camera1 的文件
  * 相似度用 dHash 感知哈希 + 汉明距离（微秒级），替代原 1920x1080
    matchTemplate（几十毫秒），且带 mtime 缓存，同一文件不重复解码
  * 定时清理过期图片，避免 start/ 目录无限增长
"""
import logging
import os
import re
import threading
import time

import cv2
import numpy as np

import config

log = logging.getLogger('yolo_service.cache')

# 设备号白名单：字母数字下划线中划线点，1~64 位（兼容既有设备号 fig... / 纯数字 / webtest）
_DEVICE_RE = re.compile(r'^[A-Za-z0-9_.\-]{1,64}$')

# 缓存文件名结构：<毫秒时间戳>_<devices><允许的后缀>
_OWNER_RES = {}
_OWNER_LOCK = threading.Lock()


def is_valid_device(devices):
    return bool(devices) and bool(_DEVICE_RE.match(devices))


def _owner_re(devices):
    """按设备号生成文件名精确匹配正则（带缓存）。"""
    with _OWNER_LOCK:
        if devices not in _OWNER_RES:
            exts = '|'.join(re.escape(e) for e in config.ALLOWED_EXTS)
            _OWNER_RES[devices] = re.compile(
                r'^\d+_' + re.escape(devices) + r'(?:' + exts + r')$', re.IGNORECASE)
        return _OWNER_RES[devices]


def device_files(start_dir, devices):
    """返回该设备的缓存图列表，按 mtime 升序（旧 -> 新）。

    注意：精确匹配文件名结构，设备 1 不会再命中设备 21 / camera1 的文件。
    """
    pattern = _owner_re(devices)
    out = []
    try:
        names = os.listdir(start_dir)
    except FileNotFoundError:
        return out
    for name in names:
        if pattern.match(name):
            path = os.path.join(start_dir, name)
            if os.path.isfile(path):
                out.append(path)
    out.sort(key=lambda p: os.path.getmtime(p))
    return out


def remove_quiet(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- dHash 去重
# path -> (mtime, hash)；文件名带毫秒时间戳不会重复，缓存仅避免重复解码
_hash_cache = {}
_HASH_LOCK = threading.Lock()


def _dhash(img):
    """dHash：缩放到 (H+1)xW 灰度图，比较相邻像素亮度，得到位串。"""
    resized = cv2.resize(img, (config.DEDUP_HASH_SIZE + 1, config.DEDUP_HASH_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return int.from_bytes(np.packbits(diff).tobytes(), 'big')


def _file_hash(path):
    """读取图片并计算 dHash；带 mtime 缓存。失败返回 None。"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _HASH_LOCK:
        cached = _hash_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    img = cv2.imread(path)
    if img is None:
        return None
    h = _dhash(img)
    with _HASH_LOCK:
        if len(_hash_cache) > 512:          # 防止极端情况下无限增长
            _hash_cache.clear()
        _hash_cache[path] = (mtime, h)
    return h


def _hamming(h1, h2):
    return bin(h1 ^ h2).count('1')


def is_scene_changed(img1_path, img2_path, threshold=None):
    """两张图是否不同（True=画面变化，需要检测）。

    用 dHash 汉明距离判断：距离 <= threshold 视为「画面未变化」。
    任一图片读取/解码失败时返回 True（宁可多检测一次，也不要漏检）。
    """
    threshold = config.DEDUP_HASH_DIST if threshold is None else threshold
    h1 = _file_hash(img1_path)
    h2 = _file_hash(img2_path)
    if h1 is None or h2 is None:
        log.warning('读取失败 img1=%s img2=%s，按“画面已变化”处理', img1_path, img2_path)
        return True

    dist = _hamming(h1, h2)
    changed = dist > threshold
    log.info('dHash 距离 %d (阈值 %d) -> %s', dist, threshold,
             '画面变化，执行检测' if changed else '画面未变化，跳过检测')
    return changed


def prune_device(start_dir, devices, keep=1):
    """只保留最新 keep 张，其余删除（配合“最新两张比较”的去重策略）。"""
    files = device_files(start_dir, devices)
    if len(files) <= keep:
        return 0
    removed = 0
    for path in files[:-keep]:
        if remove_quiet(path):
            removed += 1
    return removed


def enforce_limit(start_dir, devices):
    """单设备缓存文件数上限保护（正常情况下不会触发）。"""
    files = device_files(start_dir, devices)
    extra = len(files) - config.MAX_IMAGES_PER_DEVICE
    if extra <= 0:
        return 0
    removed = 0
    for path in files[:extra]:
        if remove_quiet(path):
            removed += 1
    return removed


def cleanup_expired(directory, ttl_hours=None):
    """删除超过 TTL 的文件，返回删除数量。"""
    ttl_hours = config.IMAGE_TTL_HOURS if ttl_hours is None else ttl_hours
    if ttl_hours <= 0:
        return 0
    deadline = time.time() - ttl_hours * 3600
    removed = 0
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return 0
    for name in names:
        path = os.path.join(directory, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < deadline:
                if remove_quiet(path):
                    removed += 1
        except OSError:
            continue
    return removed


def count_files(directory):
    try:
        return len([n for n in os.listdir(directory) if os.path.isfile(os.path.join(directory, n))])
    except FileNotFoundError:
        return 0
