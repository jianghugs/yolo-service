# -*- coding: utf-8 -*-
"""检测器封装：模型单例 + GPU 串行锁 + 结果图落盘。

与原 demo1.py 的差异：
  * 模型只在进程启动时加载一次（原实现每次请求 init_detector，约 4 秒 + 显存反复申请）
  * 推理加锁，避免 Flask 多线程并发压同一个 GPU 模型导致显存抖动
  * 结果图用 imencode 写盘，兼容中文/特殊字符路径
"""
import logging
import os
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config

log = logging.getLogger('yolo_service.detector')


class PersonDetector(object):
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or config.MODEL_PATH
        self.device = device or config.DEVICE or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        self._lock = threading.Lock()

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError('模型文件不存在: %s' % self.model_path)

        log.info('loading model %s on %s ...', self.model_path, self.device)
        t0 = time.time()
        self.model = YOLO(self.model_path)
        if self.device.startswith('cuda'):
            self.model.to(self.device)
        log.info('model loaded in %.2fs', time.time() - t0)

    def warmup(self):
        """用一张空白图预热，避免首个真实请求慢 1~2 秒。"""
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with self._lock:
                self.model.predict(dummy, device=self.device,
                                   conf=config.SCORE_THR, classes=[0], verbose=False)
            log.info('warmup done')
        except Exception as e:                     # 预热失败不影响主流程
            log.warning('warmup failed: %s', e)

    def detect(self, image_path, result_dir):
        """检测行人数，返回 (结果图路径, 人数, 耗时毫秒)。"""
        t0 = time.time()
        with self._lock:
            results = self.model.predict(
                image_path,
                device=self.device,
                conf=config.SCORE_THR,
                classes=[0],          # COCO class 0 = person
                verbose=False,
            )
        r = results[0]
        count = len(r.boxes)
        elapsed_ms = int((time.time() - t0) * 1000)

        basename = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(result_dir, basename + '_result.jpg')
        plotted = r.plot()                        # BGR ndarray
        ok, buf = cv2.imencode('.jpg', plotted)
        if ok:
            buf.tofile(result_path)               # 兼容 Unicode 路径
        else:
            log.error('imencode failed for %s', image_path)
            result_path = None

        return result_path, count, elapsed_ms

    def gpu_memory(self):
        if not self.device.startswith('cuda'):
            return None
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            return {
                'used_mb': int((total_b - free_b) / 1024 / 1024),
                'total_mb': int(total_b / 1024 / 1024),
            }
        except Exception:
            return None
