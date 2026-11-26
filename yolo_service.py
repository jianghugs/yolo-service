# -*- coding: utf-8 -*-
"""兼容入口：保留原文件名，`python yolo_service.py` 仍然可用（systemd / 旧脚本不受影响）。

实际实现已拆分到 app.py / detector.py / cache.py / callback.py / config.py。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app, main  # noqa: E402

__all__ = ['app', 'main']

if __name__ == '__main__':
    main()
