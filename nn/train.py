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
- 每个像素输出 64 个概率（net 内部已做 softmax）
  - 类别 0~62：有效距离概率（非等间距，按 log_{1.06}(GT) 四舍五入映射）
  - 类别 63：GT 无效（NA / <=0 / >35m）

Loss（“熵”）：
- 对每个像素：取 GT 类别对应的概率 p_k，loss = -log(p_k)
- 无效 GT 会落在类别 63，因此也参与 loss（学习“无效概率”）
- 注：net 输出的是概率而非 logits，因此不能用 `cross_entropy`，直接对概率做 -log 即可。

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
EPOCHS = 2000
LR = 1e-3
EPS = 1e-6
SHUFFLE = False  # 全量 batch 下打乱只有“batch 内顺序变化”，对本网络通常无意义

# 分类配置：固定 64 输出
# - bin0..62: 有效距离（按 log_{1.06}(GT) 四舍五入）
# - bin63: 无效（NA / <=0 / >35m）
NUM_BINS = 64
VALID_BINS = 63
INVALID_BIN = 63
MAX_VALID_M = 35.0
LOG_BASE = 1.06

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
      - 0..62: valid bins  (round(log_{1.06}(depth)))
      - 63: invalid (NA / <=0 / >35m)
    """
    if depth_m.ndim != 4 or depth_m.shape[1] != 1:
        raise ValueError(f"expect depth_m shape (N,1,H,W), got {tuple(depth_m.shape)}")

    d = depth_m[:, 0, :, :]  # (N,H,W)
    finite = torch.isfinite(d)
    invalid = (~finite) | (d <= 0.0) | (d > float(MAX_VALID_M))
    valid = ~invalid

    # idx_valid = round(log_{base}(d))
    # - 用 ln 计算：log_b(x) = ln(x) / ln(b)
    ln_base = float(np.log(LOG_BASE))
    if not np.isfinite(ln_base) or ln_base <= 0.0:
        raise ValueError(f"bad LOG_BASE={LOG_BASE}")

    # 仅在 valid 区域计算 log，避免 d<=0 产生 -inf/nan 干扰
    idx_f = torch.zeros_like(d, dtype=torch.float32)
    idx_f = torch.where(
        valid,
        torch.log(torch.clamp(d, min=EPS)) / float(ln_base),
        idx_f,
    )
    idx = torch.round(idx_f).to(dtype=torch.int64)
    idx = torch.clamp(idx, 0, int(VALID_BINS) - 1)
    idx = torch.where(invalid, torch.full_like(idx, int(INVALID_BIN)), idx)
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
        probs = net(inp)  # (N,64,H,W) probabilities (softmax done in net)

        # target bin: (N,H,W) long, bin63=invalid
        target = depth_to_bin_index(gt)
        # loss 归一化：所有像素都参与（包含 invalid）
        denom = float(target.numel())
        denom_t = torch.tensor(denom, device=target.device, dtype=torch.float32).clamp(min=1.0)

        # per-pixel negative log-likelihood on probs:
        # loss = -log(probs[target]) averaged over all pixels (including invalid bin)
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,H,W)
        nll_sum = -torch.log(torch.clamp(p_t, min=EPS)).sum()
        loss = nll_sum / denom_t

        # 训练时简单算个 top1 accuracy（仅有效像素）
        with torch.no_grad():
            pred = torch.argmax(probs, dim=1)  # (N,H,W)
            correct_all = (pred == target)
            acc_all = correct_all.to(dtype=torch.float32).mean()

            # 额外输出一个 valid-only accuracy（便于观察有效距离的学习情况）
            d = gt[:, 0, :, :]
            valid_gt = torch.isfinite(d) & (d > 0.0) & (d <= float(MAX_VALID_M))
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


