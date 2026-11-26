# -*- coding: utf-8 -*-
"""cache.py 单元测试：设备精确匹配、dHash 去重、过期清理。"""
import os
import sys
import time

import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache
import config


@pytest.fixture
def tmp_start(tmp_path):
    return str(tmp_path / 'start')


def _make_jpg(path, seed=0, size=(64, 64)):
    """生成一张随机噪声图并落盘。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------- 设备名校验
def test_valid_device():
    assert cache.is_valid_device('fig123')
    assert cache.is_valid_device('cam-01.a_2')
    assert cache.is_valid_device('1234567890')
    assert not cache.is_valid_device('')
    assert not cache.is_valid_device(None)
    assert not cache.is_valid_device('../etc')
    assert not cache.is_valid_device('a' * 65)


# ---------------------------------------------------------------- 设备精确匹配
def test_device_files_exact_match(tmp_start):
    os.makedirs(tmp_start, exist_ok=True)
    _make_jpg(os.path.join(tmp_start, '1700000000000_dev1.jpg'))
    _make_jpg(os.path.join(tmp_start, '1700000000001_dev1.png'))
    _make_jpg(os.path.join(tmp_start, '1700000000002_dev21.jpg'))     # 子串陷阱
    _make_jpg(os.path.join(tmp_start, '1700000000003_camera1.jpg'))   # 子串陷阱
    _make_jpg(os.path.join(tmp_start, 'unrelated.jpg'))

    files = cache.device_files(tmp_start, 'dev1')
    names = [os.path.basename(f) for f in files]
    assert names == ['1700000000000_dev1.jpg', '1700000000001_dev1.png']


def test_device_files_sorted_by_mtime(tmp_start):
    os.makedirs(tmp_start, exist_ok=True)
    a = _make_jpg(os.path.join(tmp_start, '1700000000000_a.jpg'))
    b = _make_jpg(os.path.join(tmp_start, '1700000000001_a.jpg'))
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    files = cache.device_files(tmp_start, 'a')
    assert files == [a, b]


def test_prune_device(tmp_start):
    os.makedirs(tmp_start, exist_ok=True)
    for i in range(5):
        _make_jpg(os.path.join(tmp_start, '170000000000%d_d.jpg' % i))
    removed = cache.prune_device(tmp_start, 'd', keep=1)
    assert removed == 4
    assert len(cache.device_files(tmp_start, 'd')) == 1


# ---------------------------------------------------------------- dHash 去重
def test_scene_same_image_unchanged(tmp_start):
    os.makedirs(tmp_start, exist_ok=True)
    p1 = _make_jpg(os.path.join(tmp_start, '1700000000000_a.jpg'), seed=7)
    p2 = _make_jpg(os.path.join(tmp_start, '1700000000001_a.jpg'), seed=7)
    assert cache.is_scene_changed(p1, p2) is False


def test_scene_different_image_changed(tmp_start):
    os.makedirs(tmp_start, exist_ok=True)
    p1 = _make_jpg(os.path.join(tmp_start, '1700000000000_a.jpg'), seed=1)
    p2 = _make_jpg(os.path.join(tmp_start, '1700000000001_a.jpg'), seed=99)
    assert cache.is_scene_changed(p1, p2) is True


def test_scene_broken_image_treated_as_changed(tmp_start):
    os.makedirs(tmp_start, exist_ok=True)
    p1 = _make_jpg(os.path.join(tmp_start, '1700000000000_a.jpg'), seed=1)
    bad = os.path.join(tmp_start, '1700000000001_a.jpg')
    with open(bad, 'wb') as f:
        f.write(b'not an image')
    assert cache.is_scene_changed(p1, bad) is True
    assert cache.is_scene_changed(bad, p1) is True


def test_hash_cache_hit(tmp_start):
    """同一文件第二次取哈希应命中缓存（mtime 不变）。"""
    os.makedirs(tmp_start, exist_ok=True)
    p = _make_jpg(os.path.join(tmp_start, '1700000000000_a.jpg'), seed=3)
    h1 = cache._file_hash(p)
    mtime = os.path.getmtime(p)
    h2 = cache._file_hash(p)
    assert h1 == h2
    os.utime(p, (mtime + 5, mtime + 5))
    h3 = cache._file_hash(p)     # mtime 变了会重新计算，但内容相同哈希一致
    assert h3 == h1


# ---------------------------------------------------------------- 清理
def test_cleanup_expired(tmp_start):
    os.makedirs(tmp_start, exist_ok=True)
    old = _make_jpg(os.path.join(tmp_start, 'old.jpg'))
    new = _make_jpg(os.path.join(tmp_start, 'new.jpg'))
    past = time.time() - 48 * 3600
    os.utime(old, (past, past))
    removed = cache.cleanup_expired(tmp_start, ttl_hours=24)
    assert removed == 1
    assert not os.path.exists(old)
    assert os.path.exists(new)
