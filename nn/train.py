#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/train.py

不接收任何参数，只支持：
  py -3 nn\\train.py

数据格式（来自 nn/train_data）：
- input_00001.npy: (30,40,64) float32
- output_00001.npy: (30,40) float32，单位米；无效为 0（或 <=0）

网络输出（见 net.py）：
- 每个像素输出 64 个 logits（softmax 后为 64 个类别概率）
  - 类别 0：GT 无效（真值 > 最大量程 / < 最小量程 / NA 等）
  - 类别 1~63：距离区间概率（注意：为了让 bin0 专用于“无效”，有效距离从 bin1 开始）
    - 类别 k 表示区间 [k*0.15m, (k+1)*0.15m)

Loss（“熵”）：
- 对每个像素：取 GT 类别对应的概率 p_k，loss = -log(p_k)
- 无效 GT 会落在类别 0，因此也参与 loss（学习“无效概率”）
- 等价实现：`torch.nn.functional.cross_entropy(logits, target_class)`

2026-01：全量 batch 训练（把所有样本堆成一个“大 patch”一次训完），训练轮数见 EPOCHS。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from net import Network


# 固定配置（不从命令行读取）
EPOCHS = 1000
LR = 1e-3
EPS = 1e-6
SHUFFLE = False  # 全量 batch 下打乱只有“batch 内顺序变化”，对本网络通常无意义

# 分类配置：固定 64 输出
# - bin0: 无效 GT 概率
# - bin1..63: 有效距离区间
NUM_BINS = 64
BIN_M = 0.15
# 最小/最大量程（GT 不在范围内则视为无效，落到 bin0）
# - 为了避免有效深度落到 bin0（从而和 invalid 混在一起），这里把最小量程设置为 1 个 bin：
#   depth < 0.15m 视为无效，depth in [0.15m, 9.60m) 视为有效。
MIN_RANGE_M = BIN_M
MAX_RANGE_M = float(NUM_BINS) * float(BIN_M)

# 是否使用 CUDA（不从命令行读取）
# - True: 若本机有可用 CUDA，则使用 GPU；否则自动回退到 CPU（会打印提示）
# - False: 强制使用 CPU
USE_CUDA = True

H, W, C = 30, 40, 64


def find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    inputs = sorted(train_dir.glob("input_*.npy"))
    pairs: List[Tuple[Path, Path]] = []
    for ip in inputs:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            pairs.append((ip, op))
    return pairs


def load_pair(ip: Path, op: Path) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)  # (H,W,64)
    y = np.load(str(op)).astype(np.float32, copy=False)  # (H,W)
    if x.shape != (H, W, C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (H, W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


def to_torch_input(x_hw_c: np.ndarray, device: torch.device) -> torch.Tensor:
    # (H,W,C) -> (1,C,H,W)
    t = torch.from_numpy(x_hw_c).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device=device, dtype=torch.float32)


def to_torch_target_depth(y_hw: np.ndarray, device: torch.device) -> torch.Tensor:
    # (H,W) -> (1,1,H,W)
    t = torch.from_numpy(y_hw).unsqueeze(0).unsqueeze(0).contiguous()
    return t.to(device=device, dtype=torch.float32)

def depth_to_bin_index(depth_m: torch.Tensor) -> torch.Tensor:
    """depth(m) -> bin index (long).

    depth_m: (N,1,H,W) float32
    return:  (N,H,W) int64
      - 0: invalid (<=min_range / >=max_range / NA)
      - 1..63: valid bins
    """
    if depth_m.ndim != 4 or depth_m.shape[1] != 1:
        raise ValueError(f"expect depth_m shape (N,1,H,W), got {tuple(depth_m.shape)}")

    d = depth_m[:, 0, :, :]  # (N,H,W)
    # 有效距离范围： [MIN_RANGE_M, MAX_RANGE_M)
    # - 注意上边界是开区间：depth == MAX_RANGE_M 视为 invalid
    max_m = float(MAX_RANGE_M)
    finite = torch.isfinite(d)
    valid = finite & (d >= float(MIN_RANGE_M)) & (d < max_m)

    # 有效：floor(d/bin_m) 得到 1..63（因为 d>=0.15）
    idx = torch.floor(d / float(BIN_M)).to(dtype=torch.int64)
    # 无效：置为 0
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    # 安全裁剪到 [0,63]
    idx = torch.clamp(idx, 0, int(NUM_BINS) - 1)
    return idx


def stack_all_pairs(pairs: List[Tuple[Path, Path]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """把所有样本堆成一个大 batch：
    - inp: (N,C,H,W)
    - gt:  (N,1,H,W)
    """
    xs: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    for ip, op in pairs:
        x, gt_depth = load_pair(ip, op)
        xs.append(to_torch_input(x, device).squeeze(0))  # (C,H,W)
        gts.append(to_torch_target_depth(gt_depth, device).squeeze(0))  # (1,H,W)
    inp = torch.stack(xs, dim=0).contiguous()
    gt = torch.stack(gts, dim=0).contiguous()
    return inp, gt


def main() -> int:
    here = Path(__file__).resolve().parent
    train_dir = here / "train_data"
    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")

    pairs = find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    cuda_ok = torch.cuda.is_available()
    if USE_CUDA and not cuda_ok:
        print("[device] USE_CUDA=True but torch.cuda.is_available()=False, fallback to CPU.")
    device = torch.device("cuda" if (USE_CUDA and cuda_ok) else "cpu")
    print(f"[device] {device}")
    print(f"[data] {train_dir}  pairs={len(pairs)}")

    net = Network(in_channels=C, out_bins=NUM_BINS).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # 全量 batch：一次性加载并堆叠
    t_load0 = time.time()
    try:
        inp_all, gt_all = stack_all_pairs(pairs, device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to stack all training data into one big batch (possibly OOM). "
            "Try: run on CPU or reduce files under nn/train_data."
        ) from e
    dt_load = time.time() - t_load0
    print(f"[batch] inp={tuple(inp_all.shape)} gt={tuple(gt_all.shape)}  (load+stack {dt_load:.2f}s)")

    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        if SHUFFLE and inp_all.shape[0] > 1:
            perm = torch.randperm(inp_all.shape[0], device=inp_all.device)
            inp = inp_all[perm]
            gt = gt_all[perm]
        else:
            inp = inp_all
            gt = gt_all

        net.train()
        logits = net(inp)  # (N,64,H,W)

        # target bin: (N,H,W) long, bin0=invalid
        target = depth_to_bin_index(gt)
        # loss 归一化：所有像素都参与（包含 invalid）
        denom = float(target.numel())
        denom_t = torch.tensor(denom, device=target.device, dtype=torch.float32).clamp(min=1.0)

        # per-pixel cross entropy (sum then normalize by valid count)
        # 等价于：loss = -log softmax(logits)[..., target]
        ce_sum = torch.nn.functional.cross_entropy(
            logits,
            target,
            reduction="sum",
        )
        loss = ce_sum / denom_t

        # 训练时简单算个 top1 accuracy（仅有效像素）
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)  # (N,H,W)
            correct_all = (pred == target)
            acc_all = correct_all.to(dtype=torch.float32).mean()

            # 额外输出一个 valid-only accuracy（便于观察有效距离的学习情况）
            d = gt[:, 0, :, :]
            valid_gt = torch.isfinite(d) & (d >= float(MIN_RANGE_M)) & (d < float(MAX_RANGE_M))
            denom_valid = torch.clamp(valid_gt.sum().to(dtype=torch.float32), min=1.0)
            acc_valid = ((pred == target) & valid_gt).sum().to(dtype=torch.float32) / denom_valid

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1
        dt = time.time() - t0
        print(
            f"[ep {ep+1:03d}/{EPOCHS}] step {step:06d}  "
            f"ce={loss.item():.6f}  "
            f"acc_all={float(acc_all.detach().cpu().item()):.3f}  "
            f"acc_valid={float(acc_valid.detach().cpu().item()):.3f}  "
            f"({dt:.1f}s)  batch=N={inp.shape[0]}"
        )

    # 训练完保存一份
    ckpt = here / "model_last.pt"
    torch.save({"state_dict": net.state_dict()}, str(ckpt))
    print(f"[save] {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


