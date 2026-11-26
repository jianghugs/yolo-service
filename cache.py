# -*- coding: utf-8 -*-
"""抓拍图缓存与画面相似度判断。

关键点：
  * devices 参数会拼进文件名，必须校验，防止 `../` 之类的路径穿越
  * 文件名带毫秒时间戳，避免同一秒内多次上传互相覆盖（原实现只到秒）
  * 相似度沿用原 b.py 的 matchTemplate 方案，保持判定口径一致
  * 定时清理过期图片，避免 start/ 目录无限增长
"""
import logging
import os
import re
import time

import cv2
import numpy as np

import config

log = logging.getLogger('yolo_service.cache')

# 设备号白名单：字母数字下划线中划线点，1~64 位（兼容既有设备号 fig... / 纯数字 / webtest）
_DEVICE_RE = re.compile(r'^[A-Za-z0-9_.\-]{1,64}$')


def is_valid_device(devices):
    return bool(devices) and bool(_DEVICE_RE.match(devices))


def device_files(start_dir, devices):
    """返回该设备的缓存图列表，按 mtime 升序（旧 -> 新）。"""
    out = []
    try:
        names = os.listdir(start_dir)
    except FileNotFoundError:
        return out
    for name in names:
        if devices not in name:
            continue
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


def is_scene_changed(img1_path, img2_path, threshold=None):
    """两张图是否不同（True=画面变化，需要检测）。

    与原 b.py 一致：缩放到 1920x1080 后灰度模板匹配，得分 < threshold 视为变化。
    读取失败时返回 True（宁可多检测一次，也不要漏检）。
    """
    threshold = config.DEDUP_THRESHOLD if threshold is None else threshold
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        log.warning('读取失败 img1=%s img2=%s，按“画面已变化”处理', img1_path, img2_path)
        return True

    try:
        size = config.DEDUP_STD_SIZE
        gray1 = cv2.cvtColor(cv2.resize(img1, size), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.resize(img2, size), cv2.COLOR_BGR2GRAY)
        score = float(np.max(cv2.matchTemplate(gray1, gray2, cv2.TM_CCORR_NORMED)))
    except cv2.error as e:
        log.warning('相似度计算失败(%s)，按“画面已变化”处理', e)
        return True

    log.info('相似度 %.4f (阈值 %.2f) -> %s', score, threshold,
             '画面变化，执行检测' if score < threshold else '画面未变化，跳过检测')
    return score < threshold


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
