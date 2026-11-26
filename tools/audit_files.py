#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件完整性 & 可疑删除审计。

做三件事：
  1) 按 MANIFEST.sha256 校验项目文件：缺失 / 被修改 / 大小异常
  2) 列出项目里多出来的文件（非清单文件，排除运行期目录）
  3) 扫描 /proc/*/fd 中「已被删除但仍被进程占用」的句柄，找出磁盘上已删却仍占空间的文件

用法：
    python3 tools/audit_files.py                 # 校验当前项目
    python3 tools/audit_files.py --root /path    # 校验指定目录
    python3 tools/audit_files.py --extra /home/deployer/Pedestron   # 额外扫描业务目录
    python3 tools/audit_files.py --deleted-only   # 只看删除句柄
退出码：0 全部正常；1 存在问题
"""
import argparse
import hashlib
import os
import sys

EXCLUDE_DIRS = {'__pycache__', 'dist', 'start', 'results', 'logs', 'backup',
                '.git', '.venv', 'venv', '.idea', '.vscode'}
EXCLUDE_SUFFIX = ('.pyc', '.log', '.tar.gz', '.sha256')
IGNORE_DELETED = ('/memfd:', '/dev/shm/', '/SYSV', 'socket:', 'pipe:', '/dev/zero')

C_OK = '\033[32m'
C_WARN = '\033[33m'
C_ERR = '\033[31m'
C_END = '\033[0m'


def color(text, code):
    if not sys.stdout.isatty():
        return text
    return code + text + C_END


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 2)
            if len(parts) != 3:
                continue
            digest, size, rel = parts
            entries.append((rel, digest, int(size)))
    return entries


def iter_project_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in sorted(filenames):
            if name.endswith(EXCLUDE_SUFFIX):
                continue
            full = os.path.join(dirpath, name)
            yield os.path.relpath(full, root).replace(os.sep, '/'), full


def verify(root, manifest_path):
    entries = load_manifest(manifest_path)
    print('清单文件: %s（%d 项）' % (manifest_path, len(entries)))
    missing, modified, ok = [], [], 0

    for rel, digest, size in entries:
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            missing.append(rel)
            continue
        actual_size = os.path.getsize(full)
        if actual_size != size:
            modified.append((rel, '大小 %d -> %d' % (size, actual_size)))
            continue
        if sha256(full) != digest:
            modified.append((rel, '内容 sha256 不一致'))
            continue
        ok += 1

    listed = {e[0] for e in entries}
    extra = [rel for rel, _ in iter_project_files(root) if rel not in listed]

    print('  %s 校验通过: %d' % (color('OK', C_OK), ok))
    if missing:
        print('  %s 缺失文件: %d' % (color('MISSING', C_ERR), len(missing)))
        for rel in missing:
            print('      - %s' % rel)
    if modified:
        print('  %s 被修改文件: %d' % (color('MODIFIED', C_WARN), len(modified)))
        for rel, why in modified:
            print('      ~ %s（%s）' % (rel, why))
    if extra:
        print('  %s 清单外新增文件: %d' % (color('EXTRA', C_WARN), len(extra)))
        for rel in extra:
            print('      + %s' % rel)

    return not (missing or modified)


def scan_deleted_fds(paths):
    print('\n扫描「已删除但仍被占用」的文件句柄 ...')
    hits = []
    proc = '/proc'
    if not os.path.isdir(proc):
        print('  非 Linux 环境，跳过')
        return True

    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        fd_dir = os.path.join(proc, pid, 'fd')
        try:
            fds = os.listdir(fd_dir)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        try:
            with open(os.path.join(proc, pid, 'comm')) as f:
                comm = f.read().strip()
        except OSError:
            comm = '?'

        for fd in fds:
            link = os.path.join(fd_dir, fd)
            try:
                target = os.readlink(link)
            except OSError:
                continue
            if not target.endswith('(deleted)'):
                continue
            real = target[:-len(' (deleted)')].rstrip()
            if any(real.startswith(p) for p in IGNORE_DELETED):
                continue
            if paths and not any(real.startswith(p) for p in paths):
                continue
            try:
                size = os.path.getsize(link)
            except OSError:
                size = -1
            hits.append((pid, comm, real, size))

    if not hits:
        print('  %s 未发现业务文件被删除后仍被占用' % color('OK', C_OK))
        return True

    print('  %s 发现 %d 个句柄：' % (color('WARN', C_WARN), len(hits)))
    for pid, comm, real, size in hits:
        print('      pid=%-8s %-18s %10.1f MB  %s' % (pid, comm, size / 1024.0 / 1024.0, real))
    print('  说明：这些文件已从目录中删除，但进程未释放句柄，空间不会回收；重启对应进程即可释放。')
    return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=os.path.dirname(here))
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--extra', action='append', default=[])
    parser.add_argument('--deleted-only', action='store_true')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    manifest = args.manifest or os.path.join(root, 'MANIFEST.sha256')

    print('=' * 72)
    print('yolo_service 文件完整性审计   目录: %s' % root)
    print('=' * 72)

    integrity_ok = True
    if not args.deleted_only:
        if os.path.isfile(manifest):
            integrity_ok = verify(root, manifest)
        else:
            print('%s 未找到清单 %s，跳过完整性校验（可运行 tools/make_manifest.py 生成）'
                  % (color('WARN', C_WARN), manifest))
            files = list(iter_project_files(root))
            print('  当前项目文件数: %d' % len(files))

    scan_paths = [root] + [os.path.abspath(p) for p in args.extra]
    deleted_ok = scan_deleted_fds(scan_paths)

    print('\n结论: %s' % ('全部正常' if (integrity_ok and deleted_ok) else '存在问题，请查看上方明细'))
    return 0 if (integrity_ok and deleted_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
