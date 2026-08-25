#!/usr/bin/env python3
"""远程编译 dtof.bin → 上板 → Armed → 拉取 log 并画出 inference cost 分布。

默认打开 UI：进度条、步骤列表、远端全程日志、服务器对比图和测速图。
无界面模式: py test_speed.py --cli

所有中间产物和图表都落在 tmp/。
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COST_RE = re.compile(
    r"frame_id:\s*(\d+).*?inference cost:\s*([0-9.]+)\s*ms",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"^\[\[STEP\]\]\s+(\S+)\s*$")
REMOTE_LOG = "/userdata/log/current/sense/dtof_depth/0/log0.txt"
ARMED_ENV = "export LD_LIBRARY_PATH=/app/lib:/sense/lib:/usr/hobot/lib:$LD_LIBRARY_PATH"
ARMED_CMD = "/app/bin/as_center_tool switch_mode on Armed"
DEFAULT_REMOTE_DIR = "/sense/models/dtof_depth/20260521/hbdnn"
SSH_HOST = "oycq@192.168.81.129"
REMOTE_QUANT_DIR = "/home/oycq/quant_dtof"
REMOTE_COMPILE_SH = "/tmp/dtof_speed_compile.sh"
COMPILE_TIMEOUT_S = 20 * 60
SCP_TIMEOUT_S = 120
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

STEP_LABELS: dict[str, str] = {
    "upload_net": "上传 net.py",
    "export_onnx": "导出 ONNX",
    "adapt_onboard_io": "适配上板接口",
    "hb_mapper": "hb_mapper 编译 bin",
    "hb_perf": "hb_perf 性能测试",
    "compare_quant": "量化对比 compare_quant",
    "compile_done": "远端编译结束",
    "pull_artifacts": "拉回 bin / 图片",
    "adb_push": "推包上板",
    "wait_reboot": "等待设备重启",
    "switch_armed": "切换 Armed",
    "collect": "采集 inference cost",
    "plot": "绘制测速图",
}

FULL_STEPS = [
    "upload_net",
    "export_onnx",
    "adapt_onboard_io",
    "hb_mapper",
    "hb_perf",
    "compare_quant",
    "pull_artifacts",
    "adb_push",
    "wait_reboot",
    "switch_armed",
    "collect",
    "plot",
]


class Reporter:
    def step(self, step_id: str, label: str, index: int, total: int) -> None:
        print(f"\n===== [{index}/{total}] {label} =====")

    def image(self, title: str, path: Path) -> None:
        print(f"[image] {title}: {path}")

    def done(self, ok: bool, message: str) -> None:
        print(message)


def run_adb(args: list[str], timeout: float | None = 30, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["adb", *args]
    print(">", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def adb_ok() -> None:
    out = run_adb(["devices"]).stdout
    lines = [ln for ln in out.splitlines()[1:] if ln.strip() and "device" in ln]
    if not lines:
        raise SystemExit("未检测到 adb device，请先连接设备。")
    print(out.rstrip())


def adb_shell(cmd: str, timeout: float | None = 30, check: bool = True) -> str:
    r = run_adb(["shell", cmd], timeout=timeout, check=check)
    return (r.stdout or "") + (r.stderr or "")


def detect_remote_dir() -> str:
    out = adb_shell("ls -d /sense/models/dtof_depth/*/hbdnn 2>/dev/null", check=False)
    dirs = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("/")]
    if dirs:
        print(f"检测到模型目录: {dirs[0]}")
        return dirs[0]
    print(f"未检测到模型目录，使用默认: {DEFAULT_REMOTE_DIR}")
    return DEFAULT_REMOTE_DIR


def wait_for_device(timeout_s: float) -> None:
    print("等待 adb 上线...")
    run_adb(["wait-for-device"], timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["adb", "shell", "echo ok"],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except subprocess.TimeoutExpired:
            time.sleep(0.2)
            continue
        if r.returncode == 0 and "ok" in (r.stdout or ""):
            print("adb shell 已通。")
            return
        time.sleep(0.2)
    raise SystemExit("设备超时未就绪。")


def switch_armed(timeout_s: float = 25) -> bool:
    print("log0 已有数据，立即切 Armed ...")
    deadline = time.time() + timeout_s
    n = 0
    while time.time() < deadline:
        n += 1
        print(f"[{n}] /app/bin/as_center_tool switch_mode on Armed")
        try:
            out = adb_shell(f"{ARMED_ENV}; {ARMED_CMD}", timeout=12, check=False)
        except subprocess.TimeoutExpired:
            time.sleep(0.3)
            continue
        print(out.rstrip())
        if re.search(r"->\s*\([^)]*Armed", out):
            print("心跳已包含 Armed。")
            return True
        if "failed" not in out.lower() and ("armed" in out.lower() or "mode ctrl" in out.lower()):
            print("Armed 切换成功。")
            return True
        time.sleep(0.3)
    print("Armed 未确认成功，仍按时间窗采集。")
    return False


def parse_costs(text: str, skip_warmup: int) -> tuple[np.ndarray, np.ndarray]:
    frame_ids: list[int] = []
    costs: list[float] = []
    for m in COST_RE.finditer(text):
        frame_ids.append(int(m.group(1)))
        costs.append(float(m.group(2)))
    if not costs:
        return np.array([]), np.array([])
    f = np.array(frame_ids)
    c = np.array(costs, dtype=float)
    if skip_warmup > 0 and len(c) > skip_warmup:
        f = f[skip_warmup:]
        c = c[skip_warmup:]
    return f, c


def stats_dict(costs: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(costs)),
        "mean": float(np.mean(costs)),
        "std": float(np.std(costs)),
        "min": float(np.min(costs)),
        "max": float(np.max(costs)),
        "p50": float(np.percentile(costs, 50)),
        "p90": float(np.percentile(costs, 90)),
        "p95": float(np.percentile(costs, 95)),
        "p99": float(np.percentile(costs, 99)),
    }


def plot_costs(frame_ids: np.ndarray, costs: np.ndarray, title: str, out_png: Path) -> dict[str, float]:
    s = stats_dict(costs)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    ax.plot(frame_ids, costs, lw=0.8, color="#2563eb", label="inference cost")
    ax.axhline(s["mean"], color="#dc2626", ls="--", lw=1.2, label=f"均值 {s['mean']:.3f} ms")
    ax.axhline(s["p95"], color="#d97706", ls=":", lw=1.2, label=f"P95 {s['p95']:.3f} ms")
    ax.set_xlabel("frame_id")
    ax.set_ylabel("inference cost (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    bins = max(20, min(80, int(np.sqrt(len(costs)) * 2)))
    ax.hist(costs, bins=bins, color="#93c5fd", edgecolor="#1d4ed8", alpha=0.9, label="分布")
    ax.axvline(s["mean"], color="#dc2626", ls="--", lw=1.4, label=f"均值 {s['mean']:.3f} ms")
    ax.axvline(s["p50"], color="#059669", ls="-.", lw=1.2, label=f"中位 {s['p50']:.3f} ms")
    ax.axvline(s["p95"], color="#d97706", ls=":", lw=1.2, label=f"P95 {s['p95']:.3f} ms")
    ax.set_xlabel("inference cost (ms)")
    ax.set_ylabel("帧数")
    xmax = max(s["p99"] * 1.2, s["mean"] + 3 * s["std"], 4.0)
    ax.set_xlim(max(0.0, s["min"] - 0.3), xmax)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    box = (
        f"n={int(s['n'])}\n"
        f"mean={s['mean']:.3f} ms\n"
        f"std={s['std']:.3f} ms\n"
        f"min={s['min']:.3f}  max={s['max']:.3f}\n"
        f"p50={s['p50']:.3f}  p90={s['p90']:.3f}\n"
        f"p95={s['p95']:.3f}  p99={s['p99']:.3f}"
    )
    ax.text(
        0.98,
        0.97,
        box,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#94a3b8", alpha=0.92),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return s


def write_stats(path: Path, s: dict[str, float], extra: str) -> None:
    lines = [extra, ""]
    for k in ("n", "mean", "std", "min", "max", "p50", "p90", "p95", "p99"):
        v = s[k]
        if k == "n":
            lines.append(f"{k}: {int(v)}")
        else:
            lines.append(f"{k}: {v:.4f} ms")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def adb_exec_sh(cmd: str, timeout: float, quiet: bool = False) -> str:
    if not quiet:
        print(f"> adb exec-out sh -c {cmd!r}")
    r = subprocess.run(
        ["adb", "exec-out", "sh", "-c", cmd],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return (r.stdout or b"").decode("utf-8", errors="ignore")


def fetch_cost_text() -> str:
    cmd = f"tail -n 4000 {REMOTE_LOG} 2>/dev/null | grep -F 'inference cost:' || true"
    try:
        return adb_exec_sh(cmd, timeout=30)
    except subprocess.TimeoutExpired:
        print("抽取日志超时，改用更短 tail 重试...")
        return adb_exec_sh(f"tail -n 800 {REMOTE_LOG} 2>/dev/null || true", timeout=20)


def wait_for_log_costs(timeout_s: float) -> None:
    print("等待 log0 出现 inference cost（Armed 前会偏慢）...")
    cmd = f"tail -n 80 {REMOTE_LOG} 2>/dev/null | grep -F 'inference cost:' || true"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            text = adb_exec_sh(cmd, timeout=5, quiet=True)
        except subprocess.TimeoutExpired:
            time.sleep(0.2)
            continue
        if COST_RE.search(text):
            print("log0 已有推理数据，开始 Armed。")
            return
        time.sleep(0.25)
    raise SystemExit("超时仍未在 log0 中看到 inference cost。")


def collect_armed_window(settle_s: float, collect_s: float) -> tuple[str, int]:
    print(f"Armed 成功，{settle_s:.0f}s 后开始统计...")
    time.sleep(settle_s)
    frames, _ = parse_costs(fetch_cost_text(), skip_warmup=0)
    last_id = int(frames[-1]) if frames.size else 0
    print(f"统计窗口: frame_id > {last_id}，采 {collect_s:.0f}s ...")
    time.sleep(collect_s)
    return fetch_cost_text(), last_id


def shell_alive() -> bool:
    try:
        r = subprocess.run(
            ["adb", "shell", "echo ok"],
            capture_output=True,
            text=True,
            timeout=1.2,
        )
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and "ok" in (r.stdout or "")


def reboot_and_wait_offline(timeout_s: float = 20) -> None:
    print("reboot ...")
    print("> adb reboot")
    subprocess.Popen(
        ["adb", "reboot"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not shell_alive():
            print("设备已掉线。")
            return
        time.sleep(0.15)


def deploy(bin_path: Path, remote_dir: str, reboot: bool = True) -> None:
    remote = f"{remote_dir.rstrip('/')}/dtof.bin"
    print(f"推入 {bin_path} -> {remote}")
    run_adb(["push", str(bin_path), remote], timeout=60)
    print("sync ...")
    adb_shell("sync", timeout=20)
    if reboot:
        reboot_and_wait_offline()
    else:
        print("已跳过 reboot。")


def ssh_base(host: str) -> list[str]:
    return ["ssh", *SSH_OPTS, host]


def scp_base() -> list[str]:
    return ["scp", *SSH_OPTS]


def write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


def remote_compile_script(quant_dir: str) -> str:
    root = quant_dir.replace("\\", "/").rstrip("/")
    return "\n".join(
        [
            "#!/bin/bash",
            "set -eu",
            "export PATH=\"$HOME/.local/bin:$PATH\"",
            "export PYTHONUNBUFFERED=1",
            f"ROOT=\"{root}\"",
            "BUILD=\"$ROOT/build\"",
            "run() { if command -v stdbuf >/dev/null 2>&1; then stdbuf -oL -eL \"$@\"; else \"$@\"; fi; }",
            "echo \"[[STEP]] export_onnx\"",
            "rm -rf \"$BUILD\"",
            "mkdir -p \"$BUILD\"",
            "cd \"$BUILD\"",
            "run python3.10 \"$ROOT/export_onnx.py\"",
            "echo \"[[STEP]] adapt_onboard_io\"",
            "run python3.10 \"$ROOT/adapt_onboard_io.py\"",
            "echo \"[[STEP]] hb_mapper\"",
            "run hb_mapper makertbin -c \"$ROOT/compile.yaml\" --model-type onnx",
            "test -f \"$BUILD/output/dtof.bin\"",
            "ls -lh \"$BUILD/output/dtof.bin\"",
            "echo \"[[STEP]] hb_perf\"",
            "run hb_perf output/dtof.bin",
            "echo \"[[STEP]] compare_quant\"",
            "run python3.10 \"$ROOT/compare_quant.py\"",
            "test -f \"$BUILD/compare_quant.png\"",
            "ls -lh \"$BUILD/compare_quant.png\"",
            "echo \"[[STEP]] compile_done\"",
            "",
        ]
    )


def stream_proc(
    cmd: list[str],
    log_path: Path,
    timeout: float,
    on_line: Callable[[str], None] | None = None,
) -> None:
    print(">", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n>>> {' '.join(cmd)}\n")
        logf.flush()
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None

        def _tee() -> None:
            for raw in p.stdout:
                line = raw.replace("\r", "")
                print(line, end="" if line.endswith("\n") else "\n")
                logf.write(line if line.endswith("\n") else line + "\n")
                logf.flush()
                if on_line:
                    on_line(line.rstrip("\n"))

        t = threading.Thread(target=_tee, daemon=True)
        t.start()
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            t.join(timeout=5)
            raise SystemExit(f"命令超时 ({timeout:.0f}s): {' '.join(cmd)}")
        t.join(timeout=5)
    if rc != 0:
        raise SystemExit(f"命令失败 exit={rc}: {' '.join(cmd)}")


def scp_file(src: str, dst: str, log_path: Path, timeout: float = SCP_TIMEOUT_S) -> None:
    stream_proc([*scp_base(), src, dst], log_path, timeout=timeout)


def make_stepper(reporter: Reporter, steps: list[str]) -> Callable[[str], None]:
    total = max(len(steps), 1)
    index = {sid: i + 1 for i, sid in enumerate(steps)}

    def go(step_id: str) -> None:
        label = STEP_LABELS.get(step_id, step_id)
        reporter.step(step_id, label, index.get(step_id, total), total)

    return go


def compile_remote(
    net_path: Path,
    bin_path: Path,
    out_dir: Path,
    host: str,
    quant_dir: str,
    timeout: float,
    reporter: Reporter,
    stepper: Callable[[str], None],
) -> Path:
    if not net_path.is_file():
        raise SystemExit(f"找不到 net.py: {net_path}")
    log_path = out_dir / "compile.log"
    remote_net = f"{quant_dir.rstrip('/')}/net.py"
    remote_bin = f"{quant_dir.rstrip('/')}/build/output/dtof.bin"
    remote_png = f"{quant_dir.rstrip('/')}/build/compare_quant.png"

    stepper("upload_net")
    print(f"上传 {net_path} -> {host}:{remote_net}")
    scp_file(str(net_path), f"{host}:{remote_net}", log_path)

    local_sh = out_dir / "remote_compile.sh"
    write_lf(local_sh, remote_compile_script(quant_dir))
    print(f"上传编译脚本 -> {host}:{REMOTE_COMPILE_SH}")
    scp_file(str(local_sh), f"{host}:{REMOTE_COMPILE_SH}", log_path)

    def on_line(line: str) -> None:
        m = STEP_RE.match(line.strip())
        if m and m.group(1) in STEP_LABELS and m.group(1) != "compile_done":
            stepper(m.group(1))

    print(f"远端编译 {host}:{quant_dir} ...")
    stream_proc(
        [*ssh_base(host), "bash", REMOTE_COMPILE_SH],
        log_path,
        timeout=timeout,
        on_line=on_line,
    )

    stepper("pull_artifacts")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"拉取 {host}:{remote_bin} -> {bin_path}")
    scp_file(f"{host}:{remote_bin}", str(bin_path), log_path)
    if not bin_path.is_file() or bin_path.stat().st_size <= 0:
        raise SystemExit(f"拉取后本地 bin 无效: {bin_path}")
    stamped = out_dir / "dtof.bin"
    if bin_path.resolve() != stamped.resolve():
        stamped.write_bytes(bin_path.read_bytes())
    print(f"编译完成: {bin_path}  ({bin_path.stat().st_size} bytes)")

    local_png = out_dir / "compare_quant.png"
    print(f"拉取 {host}:{remote_png} -> {local_png}")
    try:
        scp_file(f"{host}:{remote_png}", str(local_png), log_path)
    except SystemExit as e:
        print(f"量化对比图拉取失败: {e}")
    if local_png.is_file() and local_png.stat().st_size > 0:
        reporter.image("量化对比", local_png)
    return bin_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="远程编译 dtof.bin 并上板统计 inference cost")
    p.add_argument("--bin", default="tmp/dtof.bin", help="本地 dtof.bin；远程编译成功后也会写到这里")
    p.add_argument("--net", default="net.py", help="要上传到远端的本地 net.py")
    p.add_argument("--ssh", default=SSH_HOST, help="量化机 SSH，形如 user@host")
    p.add_argument("--quant-dir", default=REMOTE_QUANT_DIR, help="远端 quant_dtof 目录")
    p.add_argument("--remote-dir", default="", help="板端 hbdnn 目录，空则自动检测")
    p.add_argument("--out", default="tmp", help="输出根目录，实际写入 <out>/<时间戳>/")
    p.add_argument("--settle-s", type=float, default=2.0, help="Armed 成功后再等多少秒开始统计")
    p.add_argument("--collect-s", type=float, default=5.0, help="开始统计后采集秒数")
    p.add_argument("--boot-timeout", type=float, default=90.0, help="重启后等待设备超时")
    p.add_argument("--log-timeout", type=float, default=60.0, help="等待 log0 出现 inference cost 的超时")
    p.add_argument("--compile-timeout", type=float, default=COMPILE_TIMEOUT_S, help="远端编译超时秒数")
    p.add_argument("--skip-warmup", type=int, default=0, help="丢掉前 N 帧再统计（实时窗口默认 0）")
    p.add_argument("--skip-compile", action="store_true", help="不上传 net.py、不远程编译，直接用 --bin")
    p.add_argument("--compile-only", action="store_true", help="只远程编译并拉回 bin，不上板")
    p.add_argument("--plot-only", action="store_true", help="只拉 log / 画图，不推包重启")
    p.add_argument("--skip-push", action="store_true", help="不推包不重启，只切 Armed 并采 log")
    p.add_argument("--log", default="", help="本地 log 路径（有此参数时不访问 adb）")
    p.add_argument("--no-reboot", action="store_true", help="推包 sync 后不重启（模型可能不会生效）")
    p.add_argument("--cli", action="store_true", help="不打开 UI，只在终端跑")
    return p.parse_args()


def print_stats(s: dict[str, float], png: Path) -> None:
    print("\n===== inference cost (ms) =====")
    print(f"n     : {int(s['n'])}")
    print(f"mean  : {s['mean']:.3f}")
    print(f"std   : {s['std']:.3f}")
    print(f"min   : {s['min']:.3f}")
    print(f"max   : {s['max']:.3f}")
    print(f"p50   : {s['p50']:.3f}")
    print(f"p90   : {s['p90']:.3f}")
    print(f"p95   : {s['p95']:.3f}")
    print(f"p99   : {s['p99']:.3f}")
    print(f"图    : {png}")


def finish_plot(
    text: str,
    skip_warmup: int,
    bin_path: Path,
    out_dir: Path,
    extra: str,
    min_frame_id: int = 0,
    reporter: Reporter | None = None,
) -> int:
    frames, costs = parse_costs(text, skip_warmup)
    if min_frame_id > 0 and frames.size:
        mask = frames > min_frame_id
        frames, costs = frames[mask], costs[mask]
    if costs.size == 0:
        print("日志中没有 inference cost，无法画图。请确认已切到 Armed 且 dtof_depth 在跑。")
        return 2
    title = f"dtof inference cost  ({bin_path.name}, n={len(costs)})"
    png = out_dir / "inference_cost.png"
    s = plot_costs(frames, costs, title, png)
    write_stats(out_dir / "stats.txt", s, extra=extra)
    print_stats(s, png)
    if reporter is not None:
        reporter.image("测速分布", png)
    return 0


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def pipeline_steps(args: argparse.Namespace) -> list[str]:
    if args.log:
        return ["plot"]
    if args.plot_only:
        return ["collect", "plot"]
    if args.skip_push:
        return ["switch_armed", "collect", "plot"]
    if args.skip_compile:
        steps = ["adb_push"]
        if not args.no_reboot:
            steps.append("wait_reboot")
        steps.extend(["switch_armed", "collect", "plot"])
        return steps
    steps = [
        "upload_net",
        "export_onnx",
        "adapt_onboard_io",
        "hb_mapper",
        "hb_perf",
        "compare_quant",
        "pull_artifacts",
    ]
    if args.compile_only:
        return steps
    steps.append("adb_push")
    if not args.no_reboot:
        steps.append("wait_reboot")
    steps.extend(["switch_armed", "collect", "plot"])
    return steps


def run_pipeline(args: argparse.Namespace, reporter: Reporter) -> int:
    if args.skip_compile and args.compile_only:
        raise SystemExit("--skip-compile 和 --compile-only 不能一起用。")
    if args.compile_only and (args.plot_only or args.skip_push or args.log):
        raise SystemExit("--compile-only 不能和 --plot-only / --skip-push / --log 一起用。")

    root = Path(__file__).resolve().parent
    bin_path = resolve_path(root, args.bin)
    net_path = resolve_path(root, args.net)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_path(root, args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    stepper = make_stepper(reporter, pipeline_steps(args))
    need_compile = not (args.skip_compile or args.plot_only or args.skip_push or args.log)
    if need_compile:
        compile_remote(
            net_path,
            bin_path,
            out_dir,
            host=args.ssh,
            quant_dir=args.quant_dir,
            timeout=args.compile_timeout,
            reporter=reporter,
            stepper=stepper,
        )
        if args.compile_only:
            reporter.done(True, f"只编译，已写入 {bin_path}")
            return 0

    if args.log:
        stepper("plot")
        log_path = resolve_path(root, args.log)
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        (out_dir / "log0.txt").write_text(text, encoding="utf-8")
        remote_dir = args.remote_dir or DEFAULT_REMOTE_DIR
        code = finish_plot(
            text,
            args.skip_warmup,
            bin_path,
            out_dir,
            extra=f"bin={bin_path}\nremote_dir={remote_dir}\nlog={log_path}",
            reporter=reporter,
        )
        reporter.done(code == 0, "完成" if code == 0 else "画图失败")
        return code

    adb_ok()
    remote_dir = args.remote_dir or detect_remote_dir()
    min_frame_id = 0
    text = ""

    if args.plot_only:
        stepper("collect")
        text = fetch_cost_text()
        (out_dir / "log0.txt").write_text(text, encoding="utf-8")
    else:
        if not args.skip_push:
            if not bin_path.is_file():
                raise SystemExit(f"找不到模型文件: {bin_path}")
            stepper("adb_push")
            deploy(bin_path, remote_dir, reboot=not args.no_reboot)
        else:
            print("跳过推包/重启。")
        if not args.skip_push and not args.no_reboot:
            stepper("wait_reboot")
        wait_for_device(args.boot_timeout)
        wait_for_log_costs(args.log_timeout)
        stepper("switch_armed")
        switch_armed()
        stepper("collect")
        text, min_frame_id = collect_armed_window(args.settle_s, args.collect_s)
        (out_dir / "log0.txt").write_text(text, encoding="utf-8")
        print(f"已写入 {out_dir / 'log0.txt'} ，窗口起点 frame_id > {min_frame_id}")

    stepper("plot")
    code = finish_plot(
        text,
        args.skip_warmup,
        bin_path,
        out_dir,
        extra=(
            f"bin={bin_path}\nremote_dir={remote_dir}\nlog={REMOTE_LOG}\n"
            f"min_frame_id={min_frame_id}\nssh={args.ssh}\nquant_dir={args.quant_dir}"
        ),
        min_frame_id=min_frame_id,
        reporter=reporter,
    )
    reporter.done(code == 0, "测速完成" if code == 0 else "测速结束，但没有统计到 inference cost")
    return code


class LineTee:
    def __init__(self, orig, on_line: Callable[[str], None]) -> None:
        self.orig = orig
        self.on_line = on_line
        self.buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        try:
            self.orig.write(s)
            self.orig.flush()
        except Exception:
            pass
        self.buf += s.replace("\r", "\n")
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line:
                self.on_line(line)
        return len(s)

    def flush(self) -> None:
        try:
            self.orig.flush()
        except Exception:
            pass


def _tk_font(root, size: int = 10, bold: bool = False):
    import tkinter as tk
    from tkinter import font as tkfont

    weight = "bold" if bold else "normal"
    for family in ("Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "TkDefaultFont"):
        try:
            return tkfont.Font(root=root, family=family, size=size, weight=weight)
        except tk.TclError:
            continue
    return tkfont.nametofont("TkDefaultFont")


class SpeedUI(Reporter):
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.args = args
        self.exit_code = 1
        self._photos: list = []
        self.steps = pipeline_steps(args)
        self._step_done: set[str] = set()
        self._running = False

        self.root = tk.Tk()
        self.root.title("dToF 测速")
        self.root.geometry("1480x900")
        self.root.minsize(1100, 700)
        title_font = _tk_font(self.root, 13, True)
        ui_font = _tk_font(self.root, 10)
        log_font = _tk_font(self.root, 9)

        top = ttk.Frame(self.root, padding=(12, 10, 12, 6))
        top.pack(fill="x")
        self.step_var = tk.StringVar(value="准备中")
        tk.Label(top, textvariable=self.step_var, font=title_font, anchor="w").pack(anchor="w")
        self.prog = ttk.Progressbar(top, mode="determinate", maximum=max(len(self.steps), 1))
        self.prog.pack(fill="x", pady=(8, 0))
        self.pct_var = tk.StringVar(value="0%")
        ttk.Label(top, textvariable=self.pct_var).pack(anchor="e")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=6)

        left = ttk.Frame(body, padding=4)
        ttk.Label(left, text="步骤").pack(anchor="w")
        self.step_list = tk.Listbox(
            left, width=28, activestyle="none", exportselection=False, font=ui_font
        )
        self.step_list.pack(fill="both", expand=True)
        for i, sid in enumerate(self.steps, 1):
            self.step_list.insert("end", f"  {i:>2}/{len(self.steps)}  {STEP_LABELS.get(sid, sid)}")
        body.add(left, weight=1)

        mid = ttk.Frame(body, padding=4)
        ttk.Label(mid, text="全程日志").pack(anchor="w")
        self.log = ScrolledText(
            mid,
            wrap="none",
            font=log_font,
            state="disabled",
            background="#0f172a",
            foreground="#e2e8f0",
        )
        self.log.pack(fill="both", expand=True)
        body.add(mid, weight=3)

        right = ttk.Frame(body, padding=4)
        ttk.Label(right, text="图片").pack(anchor="w")
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.img_labels: dict[str, tk.Label] = {}
        for title in ("量化对比", "测速分布"):
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            canvas = tk.Canvas(frame, background="#1e293b", highlightthickness=0)
            vsb = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            hsb = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
            inner = ttk.Frame(canvas)
            inner.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            lab = tk.Label(inner, text="等待图片", bg="#1e293b", fg="#94a3b8", font=ui_font)
            lab.pack(anchor="nw")
            self.img_labels[title] = lab
        body.add(right, weight=3)

        bottom = ttk.Frame(self.root, padding=(12, 4, 12, 10))
        bottom.pack(fill="x")
        self.status_var = tk.StringVar(value="正在启动")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")
        ttk.Button(bottom, text="重新运行", command=self._rerun).pack(side="right")

        self.root.after(200, self._start)

    def _ui(self, fn: Callable[[], None]) -> None:
        self.root.after(0, fn)

    def log_line(self, line: str) -> None:
        def _append() -> None:
            self.log.configure(state="normal")
            self.log.insert("end", line + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")

        self._ui(_append)

    def step(self, step_id: str, label: str, index: int, total: int) -> None:
        def _set() -> None:
            self.step_var.set(f"[{index}/{total}] {label}")
            self.prog["maximum"] = total
            self.prog["value"] = index
            self.pct_var.set(f"{int(index / max(total, 1) * 100)}%")
            self.status_var.set(f"正在：{label}")
            if step_id in self.steps:
                idx = self.steps.index(step_id)
                for i, sid in enumerate(self.steps):
                    prefix = "✓" if sid in self._step_done or i < idx else ("►" if i == idx else " ")
                    self.step_list.delete(i)
                    self.step_list.insert(i, f"{prefix} {i + 1:>2}/{total}  {STEP_LABELS.get(sid, sid)}")
                self._step_done.add(step_id)
                self.step_list.selection_clear(0, "end")
                self.step_list.selection_set(idx)
                self.step_list.see(idx)

        self._ui(_set)

    def image(self, title: str, path: Path) -> None:
        def _show() -> None:
            lab = self.img_labels.get(title)
            if lab is None:
                return
            photo = _load_photo(path, max_w=900, max_h=820)
            if photo is None:
                lab.configure(text=f"无法显示: {path}")
                return
            self._photos.append(photo)
            lab.configure(image=photo, text="")
            for i in range(self.nb.index("end")):
                if self.nb.tab(i, "text") == title:
                    self.nb.select(i)
                    break

        self._ui(_show)

    def done(self, ok: bool, message: str) -> None:
        def _set() -> None:
            self.status_var.set(message)
            if ok:
                self.prog["value"] = self.prog["maximum"]
                self.pct_var.set("100%")
                self.step_var.set(message)
            else:
                self.step_var.set("失败: " + message)

        self._ui(_set)

    def _start(self) -> None:
        if self._running:
            return
        self._running = True
        self.status_var.set("运行中…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _rerun(self) -> None:
        if self._running:
            return
        self._step_done.clear()
        for i, sid in enumerate(self.steps):
            self.step_list.delete(i)
            self.step_list.insert(i, f"  {i + 1:>2}/{len(self.steps)}  {STEP_LABELS.get(sid, sid)}")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.prog["value"] = 0
        self.pct_var.set("0%")
        self.step_var.set("重新运行…")
        self._start()

    def _worker(self) -> None:
        tee_out = LineTee(sys.__stdout__, self.log_line)
        tee_err = LineTee(sys.__stderr__, self.log_line)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            self.exit_code = run_pipeline(self.args, self)
        except SystemExit as e:
            msg = str(e) if e.code not in (0, None) else "已退出"
            if e.code not in (0, None):
                print(msg)
                self.done(False, msg)
                self.exit_code = int(e.code) if isinstance(e.code, int) else 1
            else:
                self.done(True, "完成")
                self.exit_code = 0
        except Exception:
            traceback.print_exc()
            self.done(False, "异常退出，见左侧日志")
            self.exit_code = 1
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self._running = False

    def mainloop(self) -> int:
        self.root.mainloop()
        return self.exit_code


def _load_photo(path: Path, max_w: int, max_h: int):
    try:
        from PIL import Image, ImageTk
    except ImportError:
        try:
            import tkinter as tk

            return tk.PhotoImage(file=str(path))
        except Exception:
            return None
    try:
        im = Image.open(path)
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(im)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    if args.cli:
        return run_pipeline(args, Reporter())
    try:
        ui = SpeedUI(args)
    except Exception as e:
        print(f"UI 启动失败: {e}", file=sys.stderr)
        traceback.print_exc()
        print("改用命令行模式继续。", file=sys.stderr)
        return run_pipeline(args, Reporter())
    return ui.mainloop()


if __name__ == "__main__":
    sys.exit(main())
