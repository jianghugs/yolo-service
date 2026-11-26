# -*- coding: utf-8 -*-
"""
YOLO 行人检测服务（替代 Pedestron b.py）
接口协议与原 b.py 完全一致，仅端口改为 1238：
  POST /api/portraitRecognition  (form: devices, file)  上报抓拍图，识别行人数并回调告警
  GET  /api/clearPicture?devices=xxx                    清空指定设备的缓存图
区别：模型启动时加载一次，不再每次请求重新加载。
"""
import base64
import json
import os
import time

import cv2
import numpy as np
import requests
from flask import Flask, request
from flask_cors import CORS

from ultralytics import YOLO
import torch

BASEDIR = os.path.abspath(os.path.dirname(__file__))
START_DIR = os.path.join(BASEDIR, 'start')
RESULTS_DIR = os.path.join(BASEDIR, 'results')
os.makedirs(START_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 回调地址与原服务保持一致
CALLBACK_URL = "http://<ALERT_HOST>:8600/api/callback/reportAlarmInfo"
# 置信度阈值（原 Pedestron 用 0.5）
SCORE_THR = 0.5

app = Flask(__name__)
CORS(app)

# ---------- 模型：启动时加载一次，全局复用 ----------
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'loading model on {DEVICE} ...', flush=True)
MODEL = YOLO(os.path.join(BASEDIR, 'yolo11n.pt'))
if DEVICE == 'cuda:0':
    MODEL.to(DEVICE)
print('model loaded.', flush=True)


def detect_people(image_path):
    """检测图片中的行人，返回 (标注结果图路径, 人数)"""
    results = MODEL.predict(
        image_path,
        device=DEVICE,
        conf=SCORE_THR,
        classes=[0],        # COCO class 0 = person
        verbose=False,
    )
    r = results[0]
    count = int((r.boxes.cls == 0).sum().item())

    # 生成标注结果图
    basename = os.path.basename(image_path).split('.')[0]
    result_path = os.path.join(RESULTS_DIR, basename + "_result.jpg")
    plotted = r.plot()  # BGR ndarray，含检测框标注
    cv2.imwrite(result_path, plotted)
    return result_path, count


def Image_similarity_matching(img1_path, img2_path):
    # 1. 加载图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"读取失败: img1={img1_path}, img2={img2_path}")
        return True

    # 2. 统一缩放到一致尺寸（与原逻辑一致）
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"原始尺寸 - 图片1: {w1}x{h1}, 图片2: {w2}x{h2}")
    standard_size = (1920, 1080)
    img1_resized = cv2.resize(img1, standard_size)
    img2_resized = cv2.resize(img2, standard_size)

    # 3. 灰度模板匹配
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray1, gray2, cv2.TM_CCORR_NORMED)
    match_score = np.max(res)
    print(f"匹配得分: {match_score:.4f}")

    return match_score < 0.99


def get_file_list(file_path, devices):
    dir_list = []
    for mm in os.listdir(file_path):
        if devices in mm:
            dir_list.append(mm)
    if len(dir_list) <= 2:
        return
    dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
    return os.path.join(file_path, dir_list[0]), os.path.join(file_path, dir_list[1])


def reportAlarmInfo(deviceId, count, imageBase64):
    payload = json.dumps({
        "deviceId": deviceId,
        "count": count,
        "imageBase64": imageBase64
    })
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(CALLBACK_URL, headers=headers, data=payload, timeout=10)
        print(response.text)
    except Exception as e:
        print(f"回调失败: {e}")


def processingPictures(photo, devices):
    resultImgPath, count = detect_people(photo)
    if int(count) != 0:
        imageBase64 = base64.b64encode(open(resultImgPath, 'rb').read()).decode()
        reportAlarmInfo(devices, count, imageBase64)
    print(f'del:{resultImgPath},{photo},count={count}')
    try:
        os.remove(resultImgPath)
    except OSError:
        pass


@app.route('/api/portraitRecognition', methods=['POST'])
def register():
    devices = request.form.to_dict()['devices']
    img = request.files.get('file')
    suffix = '.' + img.filename.split('.')[-1]
    img_path = os.path.join(START_DIR, str(int(time.time())) + devices + suffix)
    img.save(img_path)
    print(img_path, devices)

    if get_file_list(START_DIR, devices):
        img1, img2 = get_file_list(START_DIR, devices)
        print(f'img1:{img1}', f'img2:{img2}')
        if Image_similarity_matching(img1, img2):
            processingPictures(img2, devices)
        else:
            pass
        try:
            os.remove(img1)
        except OSError:
            pass
        return {'code': '200', 'msg': 'ok'}
    else:
        return {'code': '400', 'msg': '图片不足两张'}


@app.route('/api/clearPicture', methods=['GET'])
def clearPicture():
    devices = request.args.get('devices')
    for mm in os.listdir(START_DIR):
        if devices in mm:
            try:
                os.remove(os.path.join(START_DIR, mm))
            except OSError:
                pass
    return {'code': '200', 'msg': 'ok'}


@app.route('/api/health', methods=['GET'])
def health():
    return {'code': '200', 'msg': 'ok', 'device': DEVICE, 'model': 'yolo11n'}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1238, threaded=True)
