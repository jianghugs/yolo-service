FROM python:3.10-slim

WORKDIR /app

# torch cu121 wheel 自带 CUDA 运行时，GPU 机器装 nvidia-container-toolkit 后 --gpus all 即可
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    || pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY app.py config.py detector.py cache.py callback.py yolo11n.pt ./

ENV YS_HOST=0.0.0.0 \
    YS_PORT=1238 \
    PYTHONUNBUFFERED=1

EXPOSE 1238

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:1238/api/health',timeout=3)"

CMD ["python", "app.py"]
