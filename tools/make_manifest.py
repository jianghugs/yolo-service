#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成项目文件清单 MANIFEST.sha256（用于发布校验与“文件被删/被改”审计）。

用法：
    python tools/make_manifest.py [项目根目录] [-o MANIFEST.sha256]
"""
import argparse
import hashlib
import os
import sys

EXCLUDE_DIRS = {'__pycache__', 'dist', 'start', 'results', 'logs', 'backup',
                '.git', '.venv', 'venv', '.idea', '.vscode'}
EXCLUDE_FILES = {'service.log', 'MANIFEST.sha256', 'FILELIST.txt'}
EXCLUDE_SUFFIX = ('.pyc', '.log', '.tar.gz', '.sha256')


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in sorted(filenames):
            if name in EXCLUDE_FILES or name.endswith(EXCLUDE_SUFFIX):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            yield rel, full


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default=os.path.dirname(here))
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out = args.output or os.path.join(root, 'MANIFEST.sha256')

    rows = []
    for rel, full in iter_files(root):
        rows.append((rel, sha256(full), os.path.getsize(full)))

    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# yolo_service 文件清单  格式: sha256  字节数  相对路径\n')
        for rel, digest, size in rows:
            f.write('%s  %d  %s\n' % (digest, size, rel))

    total = sum(r[2] for r in rows)
    print('已生成 %s：%d 个文件，%.2f MB' % (out, len(rows), total / 1024.0 / 1024.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
