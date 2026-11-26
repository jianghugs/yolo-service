# -*- coding: utf-8 -*-
"""告警回调：与旧服务 payload 完全一致，增加失败重试与可选异步。

payload: {"deviceId": ..., "count": ..., "imageBase64": ...}
目标:   http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo
"""
import base64
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import config

log = logging.getLogger('yolo_service.callback')

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if not config.CALLBACK_ASYNC:
        return None
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=config.CALLBACK_WORKERS,
                                       thread_name_prefix='callback')
        return _pool


def _build_payload(device_id, count, image_base64):
    return json.dumps({
        'deviceId': device_id,
        'count': count,
        'imageBase64': image_base64,
    })


def _post(device_id, count, image_base64):
    payload = _build_payload(device_id, count, image_base64)
    headers = {'Content-Type': 'application/json'}
    attempts = max(1, config.CALLBACK_RETRY + 1)

    for i in range(1, attempts + 1):
        try:
            resp = requests.post(config.CALLBACK_URL, headers=headers, data=payload,
                                 timeout=config.CALLBACK_TIMEOUT)
            body = (resp.text or '')[:200]
            if resp.status_code < 500:
                log.info('回调成功 device=%s count=%s status=%s body=%s',
                         device_id, count, resp.status_code, body)
                return True
            log.warning('回调返回 %s（第 %d/%d 次）body=%s',
                        resp.status_code, i, attempts, body)
        except Exception as e:
            log.warning('回调异常（第 %d/%d 次）device=%s: %s', i, attempts, device_id, e)
        if i < attempts:
            time.sleep(min(2 * i, 5))
    log.error('回调最终失败 device=%s count=%s url=%s', device_id, count, config.CALLBACK_URL)
    return False


def report_alarm(device_id, count, image_base64):
    """上报告警。异步模式下立即返回，不阻塞摄像头请求。"""
    if not image_base64:
        log.error('回调跳过：图片数据为空 device=%s', device_id)
        return False
    pool = _get_pool()
    if pool is None:
        return _post(device_id, count, image_base64)
    try:
        pool.submit(_post, device_id, count, image_base64)
        return True
    except Exception as e:
        log.warning('提交异步回调失败，改为同步执行: %s', e)
        return _post(device_id, count, image_base64)


def encode_image(path):
    """读取结果图并转 base64；失败返回 None。"""
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii')
    except OSError as e:
        log.error('读取结果图失败 %s: %s', path, e)
        return None


def stats():
    pool = _get_pool()
    return {'async': bool(pool), 'workers': config.CALLBACK_WORKERS if pool else 0,
            'url': config.CALLBACK_URL, 'retry': config.CALLBACK_RETRY}
