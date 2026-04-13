#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通过 adb 从设备目录拉取 bag 文件。

支持两种常用场景：
1) 无参数直接运行，默认拉取目录里最新的 bag 文件
2) 需要时可通过 --name 指定文件，例如 70.bag

示例：
    python pull_bag.py
    python pull_bag.py --latest
    python pull_bag.py --name 70.bag
    python pull_bag.py --latest --out .
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REMOTE_DIR_DEFAULT = "/userdata/log/current/sense/bag/followme_bag/0"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _adb_prefix(adb_bin: str, serial: str | None) -> list[str]:
    prefix = [adb_bin]
    if serial:
        prefix.extend(["-s", serial])
    return prefix


def list_remote_bags(adb_bin: str, serial: str | None, remote_dir: str) -> list[str]:
    cmd = _adb_prefix(adb_bin, serial) + ["shell", "ls", "-1", remote_dir]
    res = _run(cmd)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "adb shell ls failed")
    bags = [line.strip() for line in res.stdout.splitlines() if line.strip().endswith(".bag")]
    return sorted(bags)


def latest_bag(adb_bin: str, serial: str | None, remote_dir: str) -> str:
    cmd = _adb_prefix(adb_bin, serial) + ["shell", "ls", "-t", f"{remote_dir}/*.bag"]
    res = _run(cmd)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "no bag file found")
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("no bag file found")
    # ls -t 第一行是最新修改的文件
    first = lines[0]
    return Path(first).name


def pull_file(adb_bin: str, serial: str | None, remote_dir: str, bag_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"{remote_dir.rstrip('/')}/{bag_name}"
    cmd = _adb_prefix(adb_bin, serial) + ["pull", remote_path, str(out_dir)]
    res = _run(cmd)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "adb pull failed")
    return out_dir / bag_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull bag files from device via adb")
    parser.add_argument("--adb", default="adb", help="adb binary path (default: adb)")
    parser.add_argument("--serial", default=None, help="adb device serial")
    parser.add_argument("--remote-dir", default=REMOTE_DIR_DEFAULT, help="remote bag directory")
    parser.add_argument("--out", default=".", help="local output directory")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--name", help="specific bag file name to pull, e.g. 70.bag")
    group.add_argument("--latest", action="store_true", help="pull the latest bag in remote directory")

    args = parser.parse_args()
    # 默认行为：无参数时自动拉取最新 bag
    if not args.name and not args.latest:
        args.latest = True
    return args


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out).resolve()

    try:
        if args.name:
            bag_name = args.name
        else:
            bag_name = latest_bag(args.adb, args.serial, args.remote_dir)

        local_path = pull_file(args.adb, args.serial, args.remote_dir, bag_name, out_dir)
        print(f"[OK] pulled: {bag_name}")
        print(f"[OK] local: {local_path}")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERR] {exc}", file=sys.stderr)
        try:
            bags = list_remote_bags(args.adb, args.serial, args.remote_dir)
            if bags:
                print("[INFO] remote bags:")
                for name in bags:
                    print(f"  - {name}")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

