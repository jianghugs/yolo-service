# -*- coding: utf-8 -*-
"""YOLO 行人检测服务（替代 Pedestron b.py）

接口协议与原 b.py 完全一致（仅端口默认 1238）：
  POST /api/portraitRecognition   form: devices, file   上报抓拍图 -> 去重 -> 数人 -> 回调告警
  GET  /api/clearPicture?devices=xxx                    清空指定设备的缓存图
  GET  /api/health                                      健康检查（新增）
  GET  /api/stats                                       运行统计（新增）

启动：
  python app.py                  # 默认 waitress，监听 0.0.0.0:1238
  YS_PORT=1240 python app.py     # 灰度实例
"""
import atexit
import base64
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

import callback
import cache
import config
from detector import PersonDetector


# ---------------------------------------------------------------- 日志
def setup_logging():
    root = logging.getLogger()
    if getattr(root, '_ys_configured', False):
        return
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    os.makedirs(os.path.dirname(config.LOG_FILE) or '.', exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        config.LOG_FILE, maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT, encoding='utf-8')
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('waitress').setLevel(logging.INFO)
    root._ys_configured = True


setup_logging()
log = logging.getLogger('yolo_service')

os.makedirs(config.START_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

STARTED_AT = time.time()
STATS = {
    'requests': 0,
    'detections': 0,
    'alarms': 0,
    'skipped': 0,
    'errors': 0,
    'detect_ms_total': 0,
}
_stats_lock = threading.Lock()

# ---------------------------------------------------------------- 模型（进程内单例）
DETECTOR = PersonDetector()
if config.WARMUP:
    DETECTOR.warmup()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_MB * 1024 * 1024


def _bump(field, delta=1):
    with _stats_lock:
        STATS[field] = STATS.get(field, 0) + delta


# ---------------------------------------------------------------- 后台维护
def _maintenance_loop():
    """定时清理过期抓拍图与残留结果图。"""
    interval = max(60, config.CLEANUP_INTERVAL)
    while True:
        time.sleep(interval)
        try:
            n1 = cache.cleanup_expired(config.START_DIR)
            n2 = cache.cleanup_expired(config.RESULTS_DIR, ttl_hours=1)
            if n1 or n2:
                log.info('定时清理完成：start=%d results=%d', n1, n2)
        except Exception as e:
            log.warning('定时清理异常: %s', e)


def _start_maintenance():
    if config.CLEANUP_INTERVAL <= 0:
        log.info('定时清理已关闭（YS_CLEANUP_INTERVAL=0）')
        return
    t = threading.Thread(target=_maintenance_loop, name='maintenance', daemon=True)
    t.start()
    log.info('定时清理已启动，间隔 %ds，TTL %.1fh', config.CLEANUP_INTERVAL, config.IMAGE_TTL_HOURS)


# ---------------------------------------------------------------- 业务
# 每设备一把锁：同一设备的「读缓存 -> 比较 -> 清理」串行化，避免并发请求竞态
_device_locks = {}
_device_locks_guard = threading.Lock()


def _device_lock(devices):
    with _device_locks_guard:
        if devices not in _device_locks:
            _device_locks[devices] = threading.Lock()
        return _device_locks[devices]


def _process(photo, devices):
    """检测 + 告警回调，返回 (count, elapsed_ms)。"""
    result_path, count, elapsed_ms = DETECTOR.detect(photo, config.RESULTS_DIR)
    _bump('detections')
    _bump('detect_ms_total', elapsed_ms)

    if count > 0 and result_path:
        image_b64 = callback.encode_image(result_path)
        if callback.report_alarm(devices, count, image_b64):
            _bump('alarms')
    if result_path:
        cache.remove_quiet(result_path)
    log.info('检测完成 device=%s count=%d 耗时=%dms', devices, count, elapsed_ms)
    return count, elapsed_ms


@app.route('/api/portraitRecognition', methods=['POST'])
def portrait_recognition():
    _bump('requests')
    t0 = time.time()

    devices = (request.form.get('devices') or request.args.get('devices') or '').strip()
    if not cache.is_valid_device(devices):
        _bump('errors')
        log.warning('非法 devices 参数: %r', devices)
        return jsonify({'code': '400', 'msg': 'devices 参数缺失或非法'})

    upload = request.files.get('file')
    if upload is None or not upload.filename:
        _bump('errors')
        log.warning('缺少文件上传 device=%s', devices)
        return jsonify({'code': '400', 'msg': '缺少 file 文件'})

    suffix = os.path.splitext(upload.filename)[1].lower()
    if suffix not in config.ALLOWED_EXTS:
        suffix = '.jpg'

    # 毫秒时间戳 + 设备号，避免同秒覆盖
    filename = '%d_%s%s' % (int(time.time() * 1000), devices, suffix)
    img_path = os.path.join(config.START_DIR, filename)
    upload.save(img_path)

    # 简单校验图片可解码，拦截损坏/伪造文件
    probe = cv2.imread(img_path)
    if probe is None:
        _bump('errors')
        cache.remove_quiet(img_path)
        log.warning('图片无法解码，已丢弃: %s', img_path)
        return jsonify({'code': '400', 'msg': '图片无法解码'})

    try:
        with _device_lock(devices):
            files = cache.device_files(config.START_DIR, devices)
            if len(files) < 2:
                log.info('device=%s 缓存不足两张（%d），等待下一张', devices, len(files))
                return jsonify({'code': '400', 'msg': '图片不足两张'})

            # 与最新两张比较：画面变化才检测，避免静置画面重复推理
            img1, img2 = files[-2], files[-1]
            if cache.is_scene_changed(img1, img2):
                _process(img2, devices)
            else:
                _bump('skipped')

            cache.prune_device(config.START_DIR, devices, keep=1)
            cache.enforce_limit(config.START_DIR, devices)
        log.info('请求处理完成 device=%s 总耗时=%dms', devices, int((time.time() - t0) * 1000))
        return jsonify({'code': '200', 'msg': 'ok'})
    except Exception as e:
        _bump('errors')
        log.exception('处理请求异常 device=%s: %s', devices, e)
        cache.remove_quiet(img_path)
        return jsonify({'code': '500', 'msg': 'internal error'})


@app.route('/api/detectPreview', methods=['POST'])
def detect_preview():
    """无副作用检测预览：返回标注图(base64)/人数/耗时。

    与 /api/portraitRecognition 的区别：不落盘、不去重、不回调，
    供检测控制台（1239）等以 WC_DETECT_VIA 方式复用本机 GPU。
    """
    _bump('requests')
    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({'code': '400', 'msg': '缺少 file 文件'})

    data = np.frombuffer(upload.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'code': '400', 'msg': '图片无法解码'})

    t0 = time.time()
    with DETECTOR._lock:
        results = DETECTOR.model.predict(image, device=DETECTOR.device,
                                         conf=config.SCORE_THR, classes=[0], verbose=False)
    elapsed_ms = int((time.time() - t0) * 1000)
    r = results[0]
    count = len(r.boxes)
    image_b64 = None
    if count > 0:
        ok, buf = cv2.imencode('.jpg', r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            image_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    _bump('detections')
    _bump('detect_ms_total', elapsed_ms)
    return jsonify({'code': '200', 'device': DETECTOR.device,
                    'count': count, 'elapsed_ms': elapsed_ms, 'image': image_b64})


@app.route('/api/clearPicture', methods=['GET'])
def clear_picture():
    devices = (request.args.get('devices') or '').strip()
    if not cache.is_valid_device(devices):
        return jsonify({'code': '400', 'msg': 'devices 参数缺失或非法'})

    removed = 0
    for path in cache.device_files(config.START_DIR, devices):
        if cache.remove_quiet(path):
            removed += 1
    log.info('清空缓存 device=%s 删除=%d', devices, removed)
    return jsonify({'code': '200', 'msg': 'ok', 'removed': removed})


@app.route('/api/health', methods=['GET'])
def health():
    with _stats_lock:
        snapshot = dict(STATS)
    return jsonify({
        'code': '200',
        'msg': 'ok',
        'device': DETECTOR.device,
        'model': DETECTOR.model_name,
        'version': config.VERSION,
        'uptime_s': int(time.time() - STARTED_AT),
        'requests': snapshot['requests'],
        'alarms': snapshot['alarms'],
        'cache_files': cache.count_files(config.START_DIR),
        'gpu': DETECTOR.gpu_memory(),
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    with _stats_lock:
        snapshot = dict(STATS)
    avg = 0.0
    if snapshot['detections']:
        avg = round(snapshot['detect_ms_total'] / float(snapshot['detections']), 1)
    return jsonify({
        'code': '200',
        'msg': 'ok',
        'uptime_s': int(time.time() - STARTED_AT),
        'requests': snapshot['requests'],
        'detections': snapshot['detections'],
        'skipped': snapshot['skipped'],
        'alarms': snapshot['alarms'],
        'errors': snapshot['errors'],
        'detect_ms_avg': avg,
        'callback': callback.stats(),
        'config': {
            'port': config.PORT,
            'score_thr': config.SCORE_THR,
            'dedup_hash_size': config.DEDUP_HASH_SIZE,
            'dedup_hash_dist': config.DEDUP_HASH_DIST,
            'start_dir': config.START_DIR,
            'results_dir': config.RESULTS_DIR,
        },
    })


@app.errorhandler(413)
def too_large(_):
    _bump('errors')
    return jsonify({'code': '413', 'msg': '图片超过 %dMB 限制' % config.MAX_UPLOAD_MB})


@app.errorhandler(404)
def not_found(_):
    return jsonify({'code': '404', 'msg': 'not found'})


@app.errorhandler(Exception)
def on_error(e):
    _bump('errors')
    log.exception('未捕获异常: %s', e)
    return jsonify({'code': '500', 'msg': 'internal error'})


# ---------------------------------------------------------------- 启动
def _on_exit():
    log.info('服务退出（运行 %.0f 秒）', time.time() - STARTED_AT)


def main():
    atexit.register(_on_exit)
    _start_maintenance()

    log.info('=' * 60)
    log.info('yolo_service v%s 启动 | device=%s model=%s', config.VERSION, DETECTOR.device, DETECTOR.model_name)
    log.info('监听 %s:%d | 回调 %s | 去重距离 %d | 检测阈值 %.2f',
             config.HOST, config.PORT, config.CALLBACK_URL,
             config.DEDUP_HASH_DIST, config.SCORE_THR)
    log.info('start=%s results=%s log=%s', config.START_DIR, config.RESULTS_DIR, config.LOG_FILE)

    if config.SERVER == 'waitress':
        try:
            from waitress import serve
        except ImportError:
            log.warning('未安装 waitress，回退到 Flask 内置服务器（生产环境建议 pip install waitress）')
        else:
            log.info('使用 waitress 生产服务器，threads=%d', config.THREADS)
            serve(app, host=config.HOST, port=config.PORT,
                  threads=config.THREADS, ident='yolo_service/%s' % config.VERSION)
            return

    app.run(host=config.HOST, port=config.PORT, threaded=True)


if __name__ == '__main__':
    main()
