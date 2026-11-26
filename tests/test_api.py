#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接口冒烟测试（对着已启动的服务跑，不依赖 pytest）。

用法：
    python tests/test_api.py                                  # 默认 127.0.0.1:1238
    python tests/test_api.py --url http://127.0.0.1:1240       # 灰度实例
    python tests/test_api.py --device unittest_abc --keep      # 保留测试图片

说明：测试图使用随机噪声，YOLO 不会检出人物，因此不会触发真实告警回调。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import requests

FAILED = []


def check(name, cond, detail=''):
    if cond:
        print('  [PASS] %s' % name)
    else:
        print('  [FAIL] %s %s' % (name, detail))
        FAILED.append(name)


def noise_image(w=1920, h=1080, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:1238')
    ap.add_argument('--device', default='')
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()

    url = args.url.rstrip('/')
    dev = args.device or ('unittest_%d' % int(time.time()))
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tmp')
    os.makedirs(tmpdir, exist_ok=True)

    print('目标服务: %s   测试设备号: %s' % (url, dev))

    # 1. 健康检查
    try:
        r = requests.get(url + '/api/health', timeout=10)
        d = r.json()
        check('健康检查', r.status_code == 200 and d.get('code') == '200', r.text[:200])
        print('        device=%s model=%s version=%s uptime=%ss'
              % (d.get('device'), d.get('model'), d.get('version'), d.get('uptime_s')))
    except Exception as e:
        check('健康检查', False, str(e))
        print('服务不可用，终止测试')
        return 1

    # 2. 第一张图 -> 图片不足两张
    p1 = os.path.join(tmpdir, '%s_1.jpg' % dev)
    cv2.imwrite(p1, noise_image(seed=1))
    with open(p1, 'rb') as f:
        r = requests.post(url + '/api/portraitRecognition',
                          data={'devices': dev}, files={'file': f}, timeout=120)
    d = r.json()
    check('首张图返回“图片不足两张”', d.get('code') == '400' and '不足' in d.get('msg', ''), r.text[:200])

    # 3. 同一张图再传 -> 去重命中，跳过检测仍返回 200
    s0 = requests.get(url + '/api/stats', timeout=10).json()
    with open(p1, 'rb') as f:
        r = requests.post(url + '/api/portraitRecognition',
                          data={'devices': dev}, files={'file': f}, timeout=120)
    d = r.json()
    check('重复画面去重返回 200', d.get('code') == '200', r.text[:200])
    s1 = requests.get(url + '/api/stats', timeout=10).json()
    check('去重后 detections 不增加',
          s1.get('detections', 0) == s0.get('detections', 0),
          '%s -> %s' % (s0.get('detections'), s1.get('detections')))
    print('        requests=%s detections=%s skipped=%s'
          % (s1.get('requests'), s1.get('detections'), s1.get('skipped')))

    # 4. 换一张不同画面 -> 触发检测
    p2 = os.path.join(tmpdir, '%s_2.jpg' % dev)
    cv2.imwrite(p2, noise_image(seed=99))
    with open(p2, 'rb') as f:
        r = requests.post(url + '/api/portraitRecognition',
                          data={'devices': dev}, files={'file': f}, timeout=180)
    d = r.json()
    check('画面变化触发检测返回 200', d.get('code') == '200', r.text[:200])
    s2 = requests.get(url + '/api/stats', timeout=10).json()
    check('detections 计数增加', s2.get('detections', 0) >= s1.get('detections', 0) + 1,
          '%s -> %s' % (s1.get('detections'), s2.get('detections')))
    print('        推理平均耗时 %sms' % s2.get('detect_ms_avg'))

    # 5. 非法 devices 校验
    with open(p2, 'rb') as f:
        r = requests.post(url + '/api/portraitRecognition',
                          data={'devices': '../etc'}, files={'file': f}, timeout=30)
    check('非法 devices 被拒绝', r.json().get('code') == '400', r.text[:200])

    # 6. 清空缓存
    r = requests.get(url + '/api/clearPicture', params={'devices': dev}, timeout=15)
    d = r.json()
    check('清空缓存返回 200', d.get('code') == '200', r.text[:200])
    print('        删除文件数 %s' % d.get('removed'))

    if not args.keep:
        for p in (p1, p2):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    print()
    if FAILED:
        print('测试失败 %d 项: %s' % (len(FAILED), ', '.join(FAILED)))
        return 1
    print('全部测试通过 ✅')
    return 0


if __name__ == '__main__':
    sys.exit(main())
